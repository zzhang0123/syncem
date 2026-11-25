"""Ultra-relativistic continuum kernel (private, re-exported by ``kernels``).

LABEL: ``eq: directional continuum``, ``eq: channel kernel`` (channel integral
of the continuum kernel).
See :class:`ContinuumKernel` for units, assumptions and what is not certified.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..bessel import _modified_tail_bound
from ..constants import C_CGS, E_ESU, M_E
from ..ultrarel import F, F_tail_bound, G
from ._harmonic_cells import CHUNK_BUDGET, static_int, sum_point_blocks
from ._kernel_helpers import (
    Modes,
    ProjectedModes,
    check_scalar_point,
    legendre_norm,
    phase_coordinate,
    phase_weights,
    resolve_index,
    sine_from_cosine,
)
from .errors import ErrorTerm

# Above this argument F and G are below 1e-300 (F ~ sqrt(x) exp(-x)). Clamping
# keeps the K-integral finite when B_perp -> 0 sends x -> inf.
_X_CLAMP = 1e4


class ContinuumKernel(eqx.Module):
    """Ultra-relativistic beaming continuum kernel (``eq: directional continuum``).

    ``K_I(nu) = A_theta F(x)``, ``K_Q(nu) = -A_theta G(x)``, ``K_V = 0``, with
    ``B_perp = B sqrt(1 - eta^2)``, ``A_theta = sqrt(3) e^3 B_perp / (4 pi m_e c^2)``
    [erg/s/Hz/sr per electron], ``x = nu / (a_B gamma^2)``,
    ``a_B = 3 e B_perp / (4 pi m_e c)``. These spectral factors are per electron
    with uniform pitch cosine. The default ``pitch="isotropic"`` uses this closure.
    With ``pitch="joint"``, the angular kernel is ``2 delta(mu-eta) K(eta)``.
    Its projection acts on joint pitch and inclination moments. A direct
    evaluation instead needs the conditional pitch density per unit cosine at
    ``mu=eta``, given the other emitter coordinates including Faraday depth.
    The beaming limit requires ``gamma sin(theta) >> 1`` and a density that
    varies little over a pitch-angle interval of order ``1/gamma`` about
    the viewing angle. Circular polarisation vanishes at this order.
    ``channel_modes`` sums ``w_n R_j(nu_n) K_S(nu_n)
    w_b(tau_n)`` over the channel's Gauss-Legendre nodes (the integral in
    ``eq: channel kernel``), in erg/s/sr per electron for ``unit_peak``.

    ``x < x_min`` at a node with ``B_perp > 0`` is an ``equinox.error_if``
    (``F`` has a divergent slope at zero), and ``B_perp = 0`` gives zero kernel.
    ``x`` is clamped at 1e4 where ``F, G < 1e-300``. ``n_nodes_F`` and
    ``tail_cutoff`` are the static ``F``/``G`` quadrature controls of
    ``syncem.ultrarel``. :meth:`tail_bound` reports the finite-tail bound of
    that quadrature for the ``numerical`` slot. ``n_eta`` (static, keyword)
    is the Gauss-Legendre count of the ``eta`` projection. For joint pitch it
    must exceed both retained angular degrees.

    Memory: ``chunk_budget`` (static, keyword, default ``CHUNK_BUDGET =
    2^20``) bounds the ``F``/``G`` cosh-integral values of one ``lax.map``
    step: ``angular_projection`` runs its ``eta`` nodes in blocks of
    :meth:`eta_block` (``2 n_ch n_nu n_nodes_F`` values per node), and
    :meth:`samples_per_step` caps the samples of a caller that vmaps
    ``channel_modes`` (``direct_channel_average``). One ``eta`` node or one
    sample is the floor. Forward tangents multiply the working set, and nested
    ``jacfwd`` callers divide the budget with :meth:`for_tangents`
    (``build_basis`` does). Blocking changes only the summation order.

    Not certified: the discrepancy from the harmonic reference (``physical_error``
    is ``unbounded`` unless ``E_phys`` is supplied, and main.tex sec: ultrarelativistic
    states no uniform bound), the channel quadrature error (``n_nu`` of the
    channels) and the ``F``/``G`` discretisation error (only the tail is bounded).
    """

    pitch: str = eqx.field(static=True, default="isotropic", kw_only=True)
    n_nodes_F: int = eqx.field(static=True)
    tail_cutoff: float = eqx.field(static=True)
    x_min: float = eqx.field(static=True)
    E_phys: ErrorTerm | None
    n_eta: int = eqx.field(static=True)
    chunk_budget: int = eqx.field(static=True)

    name: ClassVar[str] = "continuum"
    LABEL: ClassVar[tuple[str, ...]] = (
        "eq: directional continuum",
        "eq: channel kernel",
    )
    components: ClassVar[tuple[str, ...]] = ("I", "Q")

    required_closures: tuple[str, ...] = eqx.field(
        static=True, init=False, default=("uniform_mu",)
    )

    def __init__(
        self,
        *,
        pitch="isotropic",
        n_nodes_F=128,
        tail_cutoff=50.0,
        x_min=1e-6,
        E_phys=None,
        n_eta=48,
        chunk_budget=CHUNK_BUDGET,
    ):
        if pitch not in ("isotropic", "joint"):
            raise ValueError("pitch must be 'isotropic' or 'joint'")
        self.pitch = pitch
        self.required_closures = ("uniform_mu",) if pitch == "isotropic" else ()
        for name, value in (("n_nodes_F", n_nodes_F), ("n_eta", n_eta)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be a static integer")
        if n_nodes_F < 16 or n_eta < 2:
            raise ValueError("n_nodes_F >= 16 and n_eta >= 2 are required")
        if not (float(tail_cutoff) > 0.0) or not (float(x_min) > 0.0):
            raise ValueError("tail_cutoff and x_min must be positive")
        if E_phys is not None and not isinstance(E_phys, ErrorTerm):
            raise ValueError("E_phys must be an ErrorTerm or None")
        self.n_nodes_F = int(n_nodes_F)
        self.tail_cutoff = float(tail_cutoff)
        self.x_min = float(x_min)
        self.E_phys = E_phys
        self.n_eta = int(n_eta)
        self.chunk_budget = static_int(chunk_budget, "chunk_budget", 1)

    # -- working-set budget -----------------------------------------------------

    def _values_per_point(self, channels) -> int:
        """``F`` and ``G`` integrand values of one ``(gamma, B, eta)`` point."""
        return 2 * int(np.prod(np.shape(channels.nodes))) * self.n_nodes_F

    def eta_block(self, channels) -> int:
        """``eta`` nodes per ``lax.map`` step of ``angular_projection`` (static)."""
        per_eta = self._values_per_point(channels)
        return max(1, min(self.n_eta, self.chunk_budget // per_eta))

    def samples_per_step(self, channels, requested: int) -> int:
        """Samples per step of a ``vmap`` of ``channel_modes`` within ``chunk_budget``."""
        per_sample = self._values_per_point(channels)
        return max(1, min(int(requested), self.chunk_budget // per_sample))

    def for_tangents(self, tangents: int) -> "ContinuumKernel":
        """Copy with ``chunk_budget // tangents`` (at least one) for nested
        ``jacfwd`` callers. It changes the summation order only."""
        budget = max(1, self.chunk_budget // max(1, int(tangents)))
        return dataclasses.replace(self, chunk_budget=budget)

    # -- kernel -----------------------------------------------------------------

    def _geometry(self, gamma, B, eta):
        b_perp = B * sine_from_cosine(eta)
        positive = b_perp > 0.0
        safe = jnp.where(positive, b_perp, 1.0)
        amplitude = jnp.sqrt(3.0) * E_ESU**3 * safe / (4.0 * jnp.pi * M_E * C_CGS**2)
        a_B = 3.0 * E_ESU * safe / (4.0 * jnp.pi * M_E * C_CGS)
        return positive, amplitude, a_B * gamma**2

    def _argument(self, nu, positive, scale):
        x = jnp.asarray(nu) / scale
        x = eqx.error_if(
            x,
            jnp.any(positive & (x < self.x_min)),
            "continuum kernel argument x = nu/(a_B gamma^2) below x_min",
        )
        return jnp.where(positive, jnp.minimum(x, _X_CLAMP), 1.0)

    def kernels(self, nu, gamma, B, eta) -> tuple[jax.Array, jax.Array]:
        """``(K_I, K_Q)`` [erg/s/Hz/sr per electron] at frequencies ``nu`` (Hz)."""
        positive, amplitude, scale = self._geometry(gamma, B, eta)
        x = self._argument(nu, positive, scale)
        f = F(x, n_nodes=self.n_nodes_F, tail_cutoff=self.tail_cutoff)
        g = G(x, n_nodes=self.n_nodes_F, tail_cutoff=self.tail_cutoff)
        zero = jnp.zeros_like(x)
        return (
            jnp.where(positive, amplitude * f, zero),
            jnp.where(positive, -amplitude * g, zero),
        )

    def _response(self, channels):
        nodes = jnp.asarray(channels.nodes)
        full = channels(nodes)  # (n_ch, n_ch, n_nu)
        own = jnp.diagonal(full, axis1=0, axis2=1).T
        return nodes, jnp.asarray(channels.weights), own

    def _conditional_density(self, density):
        if self.pitch == "isotropic":
            if density is not None:
                raise ValueError(
                    "pitch_density requires ContinuumKernel(pitch='joint')"
                )
            return None
        if density is None:
            raise ValueError("joint continuum evaluation needs pitch_density at mu=eta")
        density = jnp.asarray(density)
        if density.ndim or jnp.iscomplexobj(density):
            raise ValueError("pitch_density must be a real scalar")
        return eqx.error_if(
            density.astype(float),
            ~jnp.isfinite(density) | (density < 0),
            "pitch_density must be finite and nonnegative",
        )

    def channel_modes(
        self,
        channels,
        gamma,
        B,
        mu,
        eta,
        *,
        phase=None,
        depth_ref=0.0,
        s_depth=1.0,
        pitch_density=None,
    ):
        """Channel modes after integration over pitch at the given viewing cosine.

        For joint pitch, supply the conditional density per unit mu at mu=eta.
        The positional mu belongs to the common kernel protocol.
        """
        density = self._conditional_density(pitch_density)
        gamma, B, mu, eta = check_scalar_point(gamma, B, mu, eta)
        nodes, weights, response = self._response(channels)
        k_I, k_Q = self.kernels(nodes, gamma, B, eta)
        measure = weights * response
        w = phase_weights(phase, phase_coordinate(nodes), depth_ref, s_depth)
        modes = Modes(
            I=jnp.sum(measure * k_I, axis=-1),
            V=jnp.zeros(channels.n_ch, dtype=measure.dtype),
            P=jnp.einsum("cn,wcn->wc", measure * k_Q, w),
        )

        if density is None:
            return modes
        return jax.tree.map(lambda value: (2.0 * value) * density, modes)

    def angular_projection(
        self,
        channels,
        gamma,
        B,
        *,
        phase=None,
        depth_ref=0.0,
        s_depth=1.0,
        truncation,
        quadrature=None,
    ):
        if quadrature not in (None, "tensor", "product"):
            raise ValueError("quadrature must be None, 'tensor' or 'product'")
        index = resolve_index(truncation, self.components)
        gamma, B, _, _ = check_scalar_point(gamma, B, 0.0, 0.0)
        degree = max(index.truncation.L_eta, index.truncation.L_mu)
        if self.pitch == "joint" and self.n_eta <= degree:
            raise ValueError(
                f"joint continuum projection needs n_eta > max(L_mu, L_eta) = {degree}"
            )
        eta_host, weights_host = np.polynomial.legendre.leggauss(self.n_eta)
        eta_nodes, eta_weights = jnp.asarray(eta_host), jnp.asarray(weights_host)
        nodes, weights, response = self._response(channels)
        measure = weights * response  # (n_ch, n_nu)
        w = phase_weights(phase, phase_coordinate(nodes), depth_ref, s_depth)

        def at_eta(eta):
            k_I, k_Q = self.kernels(nodes, gamma, B, eta)
            return jnp.sum(measure * k_I, axis=-1), jnp.einsum(
                "cn,wcn->wc", measure * k_Q, w
            )

        # The beaming delta evaluates both Legendre factors at the same cosine.
        pairs = np.asarray(index.pairs)
        l, k = pairs.T
        legendre = np.polynomial.legendre.legvander(eta_host, degree)
        pitch_factor = legendre[:, l] if self.pitch == "joint" else (l == 0)
        pair_factors = pitch_factor * legendre[:, k]
        pair_weights = (
            eta_weights[:, None] * (2 * legendre_norm(index.pairs)) * pair_factors
        )

        def block(args, mask):  # eta nodes of one lax.map step, reduced over eta
            I_eta, P_eta = jax.vmap(at_eta)(args[0])  # (b, n_ch), (b, n_w, n_ch)
            kw = args[1] * mask[:, None]
            return jnp.einsum("ec,ek->ck", I_eta, kw), jnp.einsum(
                "ewc,ek->wck", P_eta, kw
            )

        I_proj, P_proj = sum_point_blocks(
            block, (eta_nodes, pair_weights), self.eta_block(channels)
        )
        return ProjectedModes(I=I_proj, V=jnp.zeros_like(I_proj), P=P_proj)

    def tail_bound(self, channels, gamma, B, eta, *, pitch_density=None) -> ErrorTerm:
        """Finite-tail bound of the ``F``/``G`` quadrature, ``(n_ch, 4)`` per electron.

        Columns ``I, Q, U, V`` carry ``sum_n w_n |R_j| A F_tail``, the ``G``
        analogue twice and zero. Joint pitch multiplies this bound by twice the
        conditional density at mu=eta. Excludes discretisation and roundoff error.
        """
        gamma, B, _, eta = check_scalar_point(gamma, B, 0.0, eta)
        nodes, weights, response = self._response(channels)
        positive, amplitude, scale = self._geometry(gamma, B, eta)
        x = self._argument(nodes, positive, scale)
        f_tail = F_tail_bound(x, tail_cutoff=self.tail_cutoff)
        g_tail = x * _modified_tail_bound(2.0 / 3.0, x, tail_cutoff=self.tail_cutoff)
        measure = weights * jnp.abs(response) * amplitude
        f_col = jnp.where(positive, jnp.sum(measure * f_tail, axis=-1), 0.0)
        g_col = jnp.where(positive, jnp.sum(measure * g_tail, axis=-1), 0.0)
        value = jnp.stack([f_col, g_col, g_col, jnp.zeros_like(f_col)], axis=-1)
        density = self._conditional_density(pitch_density)
        if density is not None:
            value = (2.0 * value) * density
        return ErrorTerm(
            value,
            "bound",
            "finite upper limit of the F/G cosh integrals (syncem.ultrarel); "
            "excludes quadrature discretisation and roundoff",
            "E_num",
        )

    def physical_error(self, channels) -> ErrorTerm:
        if self.E_phys is not None:
            return self.E_phys
        return ErrorTerm.unbounded(
            "ultra-relativistic beaming continuum replaces the harmonic "
            "reference; no uniform discrepancy bound (main.tex sec: ultrarelativistic)",
            "E_phys",
        )

    def truncation_error(
        self, support, channels, *, samples=None, reference=None, phase=None
    ):
        return ErrorTerm.not_applicable(
            "continuum kernel has no harmonic sum; the F/G tail bound is reported "
            "by tail_bound()"
        )

    def describe(self) -> tuple:
        return (
            ("name", self.name),
            ("label", self.LABEL),
            ("pitch", self.pitch),
            ("n_nodes_F", self.n_nodes_F),
            ("tail_cutoff", self.tail_cutoff),
            ("x_min", self.x_min),
            ("n_eta", self.n_eta),
            ("chunk_budget", self.chunk_budget),
            ("E_phys", None if self.E_phys is None else self.E_phys.kind),
        )


def joint_pitch_basis(basis):
    """Whether the stored basis projects a distribution on the beaming diagonal."""
    record = getattr(getattr(basis, "provenance", None), "kernel", ())
    return ("name", "continuum") in record and ("pitch", "joint") in record


def check_basis_pitch(basis, kernel):
    """A kernel used to estimate errors must retain the basis's pitch model."""
    if joint_pitch_basis(basis) != (
        getattr(kernel, "name", None) == "continuum"
        and getattr(kernel, "pitch", None) == "joint"
    ):
        raise ValueError("kernel pitch model must match the one that built the basis")


def sample_pitch_density(kernel, density, size):
    """Broadcast a conditional pitch density on the marginal sample grid."""
    if density is None:
        return None
    if getattr(kernel, "pitch", None) != "joint":
        raise ValueError("pitch_density requires ContinuumKernel(pitch='joint')")
    density = jnp.asarray(density)
    if density.shape not in ((), (size,)):
        raise ValueError("pitch_density must be scalar or have one value per sample")
    return jnp.broadcast_to(density, (size,))


__all__ = ["ContinuumKernel"]
