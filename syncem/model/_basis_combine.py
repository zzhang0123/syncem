"""Columns and kernel terms of a ``from_basis`` channel set (private helper of ``basis``).

The channel kernel ``H_{X,j} = sum_m L_{X,m} R_j(nu_m)`` (``eq: channel
kernel``), and every Taylor coefficient of it, is linear in ``R_j``. For
``R_j = sum_k c_jk phi_k`` the basis columns are therefore
``C_j = sum_k c_jk C_k`` with ``C_k`` the columns of the reference responses
``phi_k``: ``combine_columns`` contracts ``(n_ref, n)`` columns with the real
``(n_ch, n_ref)`` coefficients. An error term ``e_k`` of the reference
channels becomes ``|c| @ e`` (``combine_term``): ``|sum_k c_jk x_k| <=
sum_k |c_jk| |x_k|``, so a ``bound`` stays a ``bound``. A per-column envelope
``(4 n_ref, n_real)`` is combined per Stokes row and column. Units are those
of the reference terms times the (dimensionless) coefficients. Nothing here
checks the reference terms themselves. ``combine_basis`` (re-exported by
``syncem.model.basis``) assembles the combined ``SpectralBasis``.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax.numpy as jnp

from .channels import Channels
from .errors import ErrorTerm

COMBINED_NOTE = (
    "instrument channels: |c| @ the term of the {n_ref} reference channels "
    "(Channels.from_basis; triangle inequality)"
)


def combine_columns(coefficients, I_basis, V_basis, P_basis):
    """``(c @ I, c @ V, c @ P)`` for reference columns ``(n_ref, n0|n2)``."""
    c = jnp.asarray(coefficients)
    return c @ I_basis, c @ V_basis, c.astype(P_basis.dtype) @ P_basis


def combine_term(term, coefficients, n_ref, n_real):
    """``|c| @ term`` over the reference channels (valueless terms unchanged)."""
    if term.value is None:
        return term
    absc = jnp.abs(jnp.asarray(coefficients))
    n_ch = absc.shape[0]
    value = jnp.asarray(term.value)
    if value.shape == (4 * n_ref, n_real):
        per_column = value.reshape(n_ref, 4, n_real)
        out = jnp.einsum("jk,kxa->jxa", absc, per_column).reshape(4 * n_ch, n_real)
    else:
        try:
            per_channel = jnp.broadcast_to(value, (n_ref, 4))
        except ValueError:
            raise ValueError(
                f"cannot combine an error term of shape {value.shape} over "
                f"{n_ref} reference channels"
            ) from None
        out = absc @ per_channel
    note = COMBINED_NOTE.format(n_ref=n_ref)
    return ErrorTerm(
        out,
        term.kind,
        f"{term.note}; {note}" if term.note else note,
        term.manuscript_term,
    )


def combine_basis(reference_basis, channels, *, E_phys=None):
    """The basis of ``channels = Channels.from_basis(phi, c)`` from that of ``phi``.

    ``reference_basis`` is a :class:`SpectralBasis` on the reference
    channels ``phi``: ``channels.reference_channels`` must equal
    ``reference_basis.channels`` under ``equinox.tree_equal`` (an eager
    check: the same arrays and static fields, callables compared by
    identity and tables by value). The columns are ``c @`` the reference
    columns (``eq: channel kernel`` is linear in ``R_j``), shapes
    ``(n_ch, n0)``, ``(n_ch, n2)``. Each kernel term is ``|c| @`` the
    reference term, which keeps its kind (a per-column envelope
    ``(4 n_ref, n_real)`` becomes ``(4 n_ch, n_real)``). ``E_phys`` (an
    ``ErrorTerm`` for the instrument channels) replaces the combined
    physical term. Index, reference, support, phase and
    ``certified_orders`` (orders validated by finite-difference tests) are
    the reference basis's. The provenance records the combined channels and
    a note. Units: those of the reference basis. A kernel-level ``E_phys``
    (declared on the kernel) is read per reference channel and combined by
    ``|c|``. Pass ``E_phys=`` for values that refer to the instrument
    channels. The ``numerical`` term of the result is ``|c| @`` the
    reference estimate.
    ``basis_convergence`` of the result, and ``RemainderInputs.from_samples``
    probes on it, integrate the combined responses directly over the hull
    support. On the product route that rule converges more slowly than the
    reference columns (one cell spans the flat zeros of the inner
    references), so such an estimate describes the hull rule rather than
    the combined columns. The remainder is linear in ``R_j``, so ``|c| @``
    supplied reference inputs (``rho_ang``, ``H``) is also valid. Not
    certified: the reference terms, and the representation of an instrument
    by the combination (``bandpass.fit_bandpass``).
    """
    from .basis import _TERMS, KernelTerms, SpectralBasis  # basis imports this module

    if not isinstance(reference_basis, SpectralBasis):
        raise ValueError("reference_basis must be a SpectralBasis")
    if not isinstance(channels, Channels) or channels.family != "basis":
        raise ValueError("channels must come from Channels.from_basis")
    # describe() names a callable by __qualname__ and a table by its length,
    # so it cannot tell two different references apart. Compare the trees.
    if not bool(eqx.tree_equal(channels.reference_channels, reference_basis.channels)):
        raise ValueError(
            "channels.reference_channels must be the channels of reference_basis"
        )
    if E_phys is not None and not isinstance(E_phys, ErrorTerm):
        raise ValueError("E_phys must be an ErrorTerm or None")
    c = channels.coefficients
    n_ref, n_real = reference_basis.n_ch, reference_basis.index.n_real
    I_basis, V_basis, P_basis = combine_columns(
        c, reference_basis.I_basis, reference_basis.V_basis, reference_basis.P_basis
    )
    old = reference_basis.kernel_terms
    terms = {
        name: combine_term(getattr(old, name), c, n_ref, n_real) for name in _TERMS
    }
    if E_phys is not None:
        terms["physical_kernel"] = E_phys
    note = (
        f"from_basis: C_j = sum_k c_jk C_k over {n_ref} reference channels "
        "(the channel kernel is linear in R_j); kernel terms are |c| @ the "
        "reference terms; no radiation integral of the combined responses"
    )
    provenance = dataclasses.replace(
        reference_basis.provenance,
        channels=tuple(channels.describe()),
        notes=reference_basis.provenance.notes + (note,),
    )
    return SpectralBasis(
        index=reference_basis.index,
        truncation=reference_basis.truncation,
        channels=channels,
        reference=reference_basis.reference,
        support=reference_basis.support,
        I_basis=I_basis,
        V_basis=V_basis,
        P_basis=P_basis,
        kernel_terms=KernelTerms(**terms),
        provenance=provenance,
        certified_orders=reference_basis.certified_orders,
        phase=reference_basis.phase,
    )


__all__ = ["COMBINED_NOTE", "combine_basis", "combine_columns", "combine_term"]
