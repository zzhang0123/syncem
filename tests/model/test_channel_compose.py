"""Composed channel responses: ``Channels.from_basis``, ``Channels.from_callable``,
``combine_basis`` and ``bandpass.fit_bandpass`` with its response-error term.

The linearity checks compare ``from_basis`` columns with the columns of a
directly integrated channel whose response is the same combination; on the
tensor angular route the nodes do not depend on the channel support, so the
two agree to rounding. The product route integrates over cells bounded by
the channel support and agrees to quadrature accuracy only.
"""

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

from syncem.model.bandpass import fit_bandpass
from syncem.model.basis import build_basis, combine_basis
from syncem.model.bounds import RemainderInputs
from syncem.model.channels import Channels
from syncem.model.errors import ErrorTerm
from syncem.model.harmonic import HarmonicKernel
from syncem.model.index import Truncation
from syncem.model.kernels import required_m_max
from syncem.model._budget_terms import propagate_envelope

from _basis_fixtures import SMALL_SUPPORT
from _harmonic_oracles import B0, GAMMA0, NU_STAR, S_DEPTH
from syncem.model.moments import Reference

CENTRE = 2.0 * NU_STAR
REF_CENTRES = CENTRE * np.array([0.85, 1.0, 1.15])
REF_WIDTHS = CENTRE * np.full(3, 0.3)
COEFF = np.array([[0.5, 1.0, 0.25], [0.0, -0.3, 0.8]])


def references():
    return Channels.bump(REF_CENTRES, REF_WIDTHS)


def bump_formula(c, w):
    def fn(nu):
        t = (nu - c) / w
        return jnp.exp(1.0 - 1.0 / jnp.maximum(1.0 - t * t, 1e-3))

    return fn


def combination_callables(ref, coefficients):
    """One callable per row: ``nu -> sum_k c_jk phi_k(nu)`` (elementwise)."""
    return [
        (lambda nu, row=jnp.asarray(row): jnp.tensordot(row, ref(nu), axes=1))
        for row in coefficients
    ]


def harmonic_setup(channels, m_extra=2):
    m_max = required_m_max(SMALL_SUPPORT, channels) + m_extra
    kernel = HarmonicKernel(m_max, n_outer=16, n_inner=16, n_mu=24, n_eta=24)
    reference = Reference(
        GAMMA0, B0, depth_ref=3.0 * S_DEPTH, scales=(GAMMA0, B0, S_DEPTH)
    )
    return kernel, reference


def columns_rel_err(a, b):
    a, b = np.asarray(a.response_matrix()), np.asarray(b.response_matrix())
    scale = np.max(np.abs(np.stack([a, b])), axis=(0, 1))
    keep = scale > 0
    return float(np.max(np.abs(a - b)[:, keep] / scale[keep]))


# -- from_basis: the response ------------------------------------------------------


def test_from_basis_response_is_the_linear_combination():
    ref = references()
    channels = Channels.from_basis(ref, COEFF)
    nu = np.linspace(0.2 * CENTRE, 1.8 * CENTRE, 777)
    assert channels.family == "basis" and channels.n_ch == 2
    assert channels.smoothness == ref.smoothness
    assert channels.normalisation == ref.normalisation
    assert_allclose(
        np.asarray(channels(nu)),
        COEFF @ np.asarray(ref(nu)),
        rtol=1e-15,
        atol=1e-300,
    )
    support = np.asarray(channels.support)
    ref_support = np.asarray(ref.support)
    # Row 0 uses every reference; row 1 has c_10 = 0, so its hull starts at ref 1.
    assert support[0, 0] == ref_support[0, 0] and support[0, 1] == ref_support[2, 1]
    assert support[1, 0] == ref_support[1, 0] and support[1, 1] == ref_support[2, 1]
    hash(channels.describe())
    record = dict(channels.describe())
    assert record["n_reference"] == 3
    assert record["reference"] == ref.describe()


def test_from_basis_flattens_nested_bases_and_validates():
    ref = references()
    inner = Channels.from_basis(ref, np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]]))
    outer = Channels.from_basis(inner, np.array([[0.5, -1.0]]))
    assert outer.reference_channels.family == "bump"
    assert_allclose(np.asarray(outer.coefficients), [[0.5, 0.0, -1.0]], rtol=1e-15)
    one = Channels.from_basis(ref, np.array([0.0, 1.0, 0.0]))
    assert one.n_ch == 1
    nu = np.linspace(0.5 * CENTRE, 1.5 * CENTRE, 101)
    assert_allclose(np.asarray(one(nu))[0], np.asarray(ref(nu))[1], rtol=0, atol=0)
    for bad in (
        np.ones((2, 2)),  # wrong reference count
        np.zeros((1, 3)),  # a row without any reference
        np.array([[np.nan, 1.0, 0.0]]),
        np.array([[1j, 1.0, 0.0]]),
        np.ones((1, 1, 3)),
    ):
        with pytest.raises(ValueError):
            Channels.from_basis(ref, bad)
    with pytest.raises(ValueError):
        Channels.from_basis("not channels", np.ones((1, 3)))


def test_from_basis_select_and_edge_jump():
    ref = Channels.tophat(REF_CENTRES, REF_WIDTHS)
    channels = Channels.from_basis(ref, COEFF)
    assert channels.smoothness == 0
    assert channels.edge_jump() == pytest.approx(np.max(np.abs(COEFF).sum(axis=1)))
    second = channels.select(1)
    nu = np.linspace(0.5 * CENTRE, 1.5 * CENTRE, 55)
    assert second.n_ch == 1
    assert_allclose(np.asarray(second(nu))[0], np.asarray(channels(nu))[1], rtol=0)
    assert_allclose(np.asarray(second.support)[0], np.asarray(channels.support)[1])


# -- from_callable: the response ---------------------------------------------------


def test_from_callable_reproduces_the_bump_values_and_derivatives():
    bump = Channels.bump([CENTRE], [0.4 * CENTRE])
    fn = bump_formula(CENTRE, 0.4 * CENTRE)
    channels = Channels.from_callable(fn, np.asarray(bump.support)[0], smoothness=99)
    assert channels.family == "callable" and channels.smoothness == 99
    lo, hi = np.asarray(bump.support)[0]
    nu = np.concatenate(
        [
            np.linspace(0.5 * CENTRE, 1.5 * CENTRE, 301),
            [lo, hi, np.nextafter(lo, np.inf), np.nextafter(hi, -np.inf)],
        ]
    )
    assert_allclose(np.asarray(channels(nu)), np.asarray(bump(nu)), rtol=0, atol=0)

    def derivative(ch, order):
        f = lambda x: ch(x)[0]  # noqa: E731
        for _ in range(order):
            f = jax.grad(f)
        return np.asarray(jax.vmap(f)(jnp.asarray(nu)))

    for order in (1, 2, 3):
        got, want = derivative(channels, order), derivative(bump, order)
        assert np.all(np.isfinite(got))
        assert_allclose(got, want, rtol=1e-14, atol=0)
    assert channels.edge_jump() == 0.0
    jitted = jax.jit(lambda x: channels(x))(nu)
    assert_allclose(np.asarray(jitted), np.asarray(channels(nu)), rtol=1e-13)


@pytest.mark.parametrize(
    "support",
    [
        (0.0, 1e9),
        (-1e8, 1e9),
        (2e9, 1e9),
        (1e9, 1e9),
        (1e9, np.inf),
        (np.nan, 1e9),
        (-np.inf, 1e9),
    ],
)
def test_from_callable_refuses_a_support_not_in_positive_frequency(support):
    with pytest.raises(ValueError, match="support"):
        Channels.from_callable(lambda nu: jnp.ones_like(nu), support, smoothness=0)


def test_from_callable_shape_and_declaration_checks():
    fn = bump_formula(CENTRE, 0.4 * CENTRE)
    support = (0.6 * CENTRE, 1.4 * CENTRE)
    with pytest.raises(ValueError, match="callable"):
        Channels.from_callable(3.0, support, smoothness=0)
    with pytest.raises(ValueError, match="support"):
        Channels.from_callable(fn, np.ones((2, 3)), smoothness=0)
    with pytest.raises(ValueError):
        Channels.from_callable([fn, fn], support, smoothness=0)  # 2 fns, 1 support
    with pytest.raises(ValueError, match="smoothness"):
        Channels.from_callable(fn, support, smoothness=-1)
    with pytest.raises(ValueError, match="shape"):
        Channels.from_callable(lambda nu: jnp.ones(3), support, smoothness=0)
    with pytest.raises(ValueError, match="finite"):
        Channels.from_callable(lambda nu: jnp.log(nu - nu), support, smoothness=0)
    with pytest.raises(ValueError, match="real"):
        Channels.from_callable(lambda nu: 1j * nu, support, smoothness=0)
    # A response that does not vanish at its edges is at most C^0.
    flat = lambda nu: jnp.ones_like(nu)  # noqa: E731
    with pytest.raises(ValueError, match="edge"):
        Channels.from_callable(flat, support, smoothness=1)
    tophat = Channels.from_callable(flat, support, smoothness=0)
    assert tophat.edge_jump() == pytest.approx(1.0)
    two = Channels.from_callable(fn, [support, support], smoothness=99)
    assert two.n_ch == 2
    hash(two.describe())


def test_from_callable_refutes_a_declared_smoothness_at_the_edges():
    """A response extended by zero is C^k only if R .. R^(k) vanish at the edges."""
    lo, hi = 0.6 * CENTRE, 1.4 * CENTRE
    parabola = lambda nu: 4 * (nu - lo) * (hi - nu) / (hi - lo) ** 2  # noqa: E731
    assert Channels.from_callable(parabola, (lo, hi), smoothness=0).edge_jump() < 1e-6
    for declared in (1, 3, 99):
        with pytest.raises(ValueError, match=r"R\^\(1\).*smoothness=0"):
            Channels.from_callable(
                parabola, (lo, hi), smoothness=declared, derivative_bound={3: 0.0}
            )
    w = 0.4 * CENTRE
    cosine = lambda nu: 0.5 * (1 + jnp.cos(jnp.pi * (nu - CENTRE) / w))  # noqa: E731
    support = (CENTRE - w, CENTRE + w)
    assert Channels.from_callable(cosine, support, smoothness=1).smoothness == 1
    with pytest.raises(ValueError, match=r"R\^\(2\).*smoothness=1"):
        Channels.from_callable(cosine, support, smoothness=2)
    with pytest.raises(ValueError, match="JAX"):
        Channels.from_callable(
            lambda nu: np.exp(-(((nu - CENTRE) / w) ** 2)), support, smoothness=0
        )


def test_from_callable_derivative_bound_is_recorded_and_checked():
    w = 0.4 * CENTRE
    fn = bump_formula(CENTRE, w)
    support = (CENTRE - w, CENTRE + w)
    nu = jnp.linspace(support[0], support[1], 4001)[1:-1]
    third = jax.grad(jax.grad(jax.grad(fn)))
    sampled = float(np.max(np.abs(np.asarray(jax.vmap(third)(nu)))))
    declared = Channels.from_callable(
        fn, support, smoothness=99, derivative_bound={3: 2.0 * sampled}
    )
    assert dict(declared.describe())["derivative_bound"] == ((3, (2.0 * sampled,)),)
    assert declared.derivative_bound_reason(3) is None
    assert "derivative_bound" in declared.derivative_bound_reason(2)
    undeclared = Channels.from_callable(fn, support, smoothness=99)
    assert "derivative_bound" in undeclared.derivative_bound_reason(3)
    assert references().derivative_bound_reason(3) is None
    with pytest.raises(ValueError, match="derivative_bound"):
        Channels.from_callable(
            fn, support, smoothness=99, derivative_bound={3: 0.1 * sampled}
        )
    for bad in ({3: -1.0}, {3: np.nan}, {-1: 1.0}, {3: [1.0, 2.0]}, 5.0):
        with pytest.raises(ValueError, match="derivative_bound"):
            Channels.from_callable(fn, support, smoothness=99, derivative_bound=bad)


# -- the basis -------------------------------------------------------------------------


def test_from_basis_columns_equal_direct_columns_on_the_tensor_route():
    ref = references()
    combined = Channels.from_basis(ref, COEFF)
    direct = Channels.from_callable(
        combination_callables(ref, COEFF),
        np.asarray(combined.support),
        smoothness=99,
    )
    kernel, reference = harmonic_setup(combined)
    kw = dict(support=SMALL_SUPPORT, quadrature="tensor", convergence=False)
    truncation = Truncation(1, 1, 2)
    by_basis = build_basis(kernel, combined, truncation, reference, **kw)
    by_direct = build_basis(kernel, direct, truncation, reference, **kw)
    by_reference = build_basis(kernel, ref, truncation, reference, **kw)
    assert columns_rel_err(by_basis, by_direct) < 1e-12
    assert_allclose(
        np.asarray(by_basis.I_basis), COEFF @ np.asarray(by_reference.I_basis), rtol=0
    )
    assert_allclose(
        np.asarray(by_basis.P_basis), COEFF @ np.asarray(by_reference.P_basis), rtol=0
    )
    assert by_basis.channels is combined
    assert any("from_basis" in note for note in by_basis.provenance.notes)
    assert by_basis.provenance.channels == combined.describe()


def test_from_basis_product_route_matches_direct_cells_for_single_references():
    """Rows with one reference each: the product cells of the two routes coincide.

    For a general combination the direct product route integrates one cell
    over the hull, across the flat zeros of the inner references, and
    converges slowly (16, 32, 64, 128 outer/inner nodes: 0.56, 0.32, 0.015,
    0.012 of the largest column against the from_basis route at 64 nodes,
    which moves by 2e-4 from 32 to 64); the tensor test above checks the
    linearity there.
    """
    ref = references()
    single = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    combined = Channels.from_basis(ref, single)
    direct = Channels.from_callable(
        combination_callables(ref, single), np.asarray(combined.support), smoothness=99
    )
    assert_allclose(np.asarray(direct.support), np.asarray(ref.support)[1:])
    kernel, reference = harmonic_setup(ref)
    kw = dict(support=SMALL_SUPPORT, convergence=False)
    truncation = Truncation(1, 1, 2)
    by_basis = build_basis(kernel, combined, truncation, reference, **kw)
    by_direct = build_basis(kernel, direct, truncation, reference, **kw)
    assert columns_rel_err(by_basis, by_direct) < 1e-12


def test_kernel_terms_combine_by_absolute_coefficients():
    ref = references()
    combined = Channels.from_basis(ref, COEFF)
    kernel, reference = harmonic_setup(combined)
    kw = dict(support=SMALL_SUPPORT, convergence="angular")
    truncation = Truncation(1, 1, 1)
    by_reference = build_basis(kernel, ref, truncation, reference, **kw)
    by_basis = build_basis(kernel, combined, truncation, reference, **kw)
    again = combine_basis(by_reference, combined)
    assert_allclose(
        np.asarray(again.response_matrix()), np.asarray(by_basis.response_matrix())
    )
    absC = np.abs(COEFF)
    n_real = by_reference.index.n_real
    ref_num = np.asarray(by_reference.kernel_terms.numerical.value).reshape(
        3, 4, n_real
    )
    want = np.einsum("jk,kxa->jxa", absC, ref_num).reshape(8, n_real)
    numerical = by_basis.kernel_terms.numerical
    assert numerical.kind == by_reference.kernel_terms.numerical.kind == "estimate"
    assert_allclose(np.asarray(numerical.value), want, rtol=1e-14)
    assert "|c|" in numerical.note
    truncation_term = by_basis.kernel_terms.harmonic_truncation
    assert truncation_term.kind == "bound"
    assert_allclose(np.asarray(truncation_term.value), np.zeros((2, 4)))
    assert by_basis.kernel_terms.physical_kernel.kind == "unbounded"
    declared = ErrorTerm(np.full((2, 4), 1e-30), "bound", "declared E_phys")
    with_phys = combine_basis(by_reference, combined, E_phys=declared)
    assert with_phys.kernel_terms.physical_kernel is declared
    other = Channels.from_basis(Channels.bump(REF_CENTRES, 0.9 * REF_WIDTHS), COEFF)
    with pytest.raises(ValueError, match="reference"):
        combine_basis(by_reference, other)
    with pytest.raises(ValueError):
        combine_basis(by_reference, ref)


def test_combine_basis_refuses_a_reference_that_describe_cannot_tell_apart():
    """Two callables with one ``__qualname__`` on one support share ``describe()``."""
    w = 0.65 * CENTRE
    support = np.asarray(Channels.bump([CENTRE], [w]).support)[0]
    built_on = Channels.from_callable(bump_formula(CENTRE, w), support, smoothness=99)
    other = Channels.from_callable(
        bump_formula(CENTRE, 0.9 * w), support, smoothness=99
    )
    assert built_on.describe() == other.describe()
    kernel, reference = harmonic_setup(built_on)
    kw = dict(support=SMALL_SUPPORT, convergence=False)
    reference_basis = build_basis(
        kernel, built_on, Truncation(1, 1, 1), reference, **kw
    )
    same = combine_basis(reference_basis, Channels.from_basis(built_on, [2.0]))
    assert_allclose(
        np.asarray(same.I_basis), 2.0 * np.asarray(reference_basis.I_basis), rtol=0
    )
    with pytest.raises(ValueError, match="reference_basis"):
        combine_basis(reference_basis, Channels.from_basis(other, [2.0]))
    grid = np.linspace(0.6 * CENTRE, 1.4 * CENTRE, 5)
    first = Channels.from_table(grid, [0.0, 1.0, 0.5, 0.2, 0.0], smoothness=0)
    second = Channels.from_table(grid, [0.0, 1.0, 0.7, 0.2, 0.0], smoothness=0)
    assert first.describe() == second.describe()
    table_basis = build_basis(
        kernel, first, Truncation(1, 1, 0), reference, allow_nonsmooth=True, **kw
    )
    with pytest.raises(ValueError, match="reference_basis"):
        combine_basis(table_basis, Channels.from_basis(second, [1.0]))


def test_from_callable_reproduces_the_bump_basis():
    bump = Channels.bump([CENTRE], [0.65 * CENTRE])
    channels = Channels.from_callable(
        bump_formula(CENTRE, 0.65 * CENTRE), np.asarray(bump.support), smoothness=99
    )
    kernel, reference = harmonic_setup(bump)
    kw = dict(support=SMALL_SUPPORT, convergence=False)
    truncation = Truncation(1, 1, 2)
    direct = build_basis(kernel, bump, truncation, reference, **kw)
    by_callable = build_basis(kernel, channels, truncation, reference, **kw)
    assert columns_rel_err(direct, by_callable) < 1e-13


def _remainder_inputs(basis):
    n_ch, n_lk = basis.n_ch, basis.index.n_lk
    return RemainderInputs(
        rho_ang=np.full((n_ch, 3), 1e-30),
        H=np.full((n_ch, 3, n_lk), 1e-28),
        absolute_moments=np.full((2, n_lk), 1e-3),
    )


def test_callable_remainder_is_a_bound_only_with_a_declared_derivative_bound():
    w = 0.65 * CENTRE
    bump = Channels.bump([CENTRE], [w])
    support = np.asarray(bump.support)[0]
    fn = bump_formula(CENTRE, w)
    kernel, reference = harmonic_setup(bump)
    truncation = Truncation(1, 1, 2)  # N + 1 = 3
    kw = dict(support=SMALL_SUPPORT, convergence=False)
    third = jax.grad(jax.grad(jax.grad(fn)))
    grid = jnp.linspace(support[0], support[1], 4001)[1:-1]
    sup3 = 2.0 * float(np.max(np.abs(np.asarray(jax.vmap(third)(grid)))))
    cases = {
        "bump family": (bump, "bound"),
        "callable without bound": (
            Channels.from_callable(fn, support, smoothness=99),
            "estimate",
        ),
        "callable with order 2 only": (
            Channels.from_callable(
                fn, support, smoothness=99, derivative_bound={2: 1e300}
            ),
            "estimate",
        ),
        "callable with order N + 1": (
            Channels.from_callable(
                fn, support, smoothness=99, derivative_bound={3: sup3}
            ),
            "bound",
        ),
    }
    for name, (channels, kind) in cases.items():
        basis = build_basis(kernel, channels, truncation, reference, **kw)
        term = _remainder_inputs(basis).basis_remainder(basis, 1.0)
        assert term.kind == kind, name
        if channels.family == "callable":
            assert "derivative_bound" in term.note, name
    # A callable reference inside from_basis passes its declaration through.
    ref = Channels.from_callable(fn, support, smoothness=99)
    combined = Channels.from_basis(ref, np.array([[2.0]]))
    assert "derivative_bound" in combined.derivative_bound_reason(3)
    basis = build_basis(kernel, combined, truncation, reference, **kw)
    assert _remainder_inputs(basis).basis_remainder(basis, 1.0).kind == "estimate"


# -- bandpass fit ----------------------------------------------------------------------


def _fine_references(n=9, width=0.12, span=0.5):
    centres = CENTRE * (1.0 + np.linspace(-span, span, n))
    return Channels.bump(centres, np.full(n, width * CENTRE))


def test_fit_bandpass_recovers_an_exact_combination():
    ref = _fine_references()
    rng = np.random.default_rng(3)
    true = np.abs(rng.standard_normal((2, ref.n_ch)))
    # The outer references carry no weight, so the reference set is wider
    # than the bandpass and covers the interpolant's last grid interval.
    true[:, [0, -1]] = 0.0
    grid = np.linspace(0.3 * CENTRE, 1.7 * CENTRE, 1201)
    table = true @ np.asarray(ref(grid))
    fit = fit_bandpass(ref, grid, table)
    assert_allclose(np.asarray(fit.coefficients), true, rtol=1e-9, atol=1e-12)
    assert fit.channels.family == "basis" and fit.channels.n_ch == 2
    peak = np.max(np.abs(table), axis=1, keepdims=True)
    at_samples = np.abs(table - np.asarray(fit.channels(grid))) / peak
    assert float(np.max(at_samples)) < 1e-12
    # Between samples the measured response is the linear interpolant, whose
    # error h^2 max|R''| / 8 (about 3e-4 of the peak here) is the residual.
    h = grid[1] - grid[0]
    second = np.abs(np.diff(table, 2, axis=1)) / h**2
    interpolation = h**2 * np.max(second, axis=1) / 8 / peak[:, 0]
    assert np.all(np.asarray(fit.relative_residual) <= 1.1 * interpolation)
    assert float(np.max(fit.uncovered)) == 0.0
    term = fit.response_uncertainty()
    assert term.kind == "estimate" and term.value.shape == (8, 4 * ref.n_ch)
    assert np.all(np.isfinite(np.asarray(term.value)))
    operator = np.asarray(fit.operator())
    assert operator.shape == (8, 4 * ref.n_ch)
    assert_allclose(operator, np.kron(np.asarray(fit.coefficients), np.eye(4)))


def _tilted_bandpass(grid):
    """A smooth, asymmetric bandpass with a ripple; zero outside its band."""
    t = (grid - CENTRE) / (0.35 * CENTRE)
    inside = np.abs(t) < 1
    shape = np.zeros_like(grid)
    shape[inside] = (1 + 0.3 * t[inside] + 0.05 * np.sin(9 * t[inside])) * (
        1 - t[inside] ** 2
    ) ** 2
    return shape


def test_fit_bandpass_covering_dominates_the_residual():
    ref = _fine_references()
    grid = np.linspace(0.5 * CENTRE, 1.5 * CENTRE, 401)
    fit = fit_bandpass(ref, grid, _tilted_bandpass(grid))
    phi = np.asarray(ref(np.asarray(fit.check_grid)))
    cover = np.asarray(fit.covering) @ phi
    residual = np.abs(np.asarray(fit.residual))
    assert np.all(cover >= residual * (1 - 1e-12))
    assert 1e-6 < float(np.max(fit.relative_residual)) < 0.2
    # Between the check points the covering is an estimate: test it on a grid
    # four times denser that the fit did not see.
    dense = np.linspace(0.45 * CENTRE, 1.55 * CENTRE, 16001)
    measured = np.interp(dense, grid, _tilted_bandpass(grid), left=0.0, right=0.0)
    represented = np.asarray(fit.channels(dense))[0]
    off_grid = np.abs(measured - represented)
    assert np.all(np.asarray(fit.covering) @ np.asarray(ref(dense)) >= 0.99 * off_grid)
    hash(fit.describe())


def test_fit_bandpass_uncovered_residual_is_unbounded_and_inputs_are_checked():
    ref = _fine_references(n=3, span=0.1)
    grid = np.linspace(0.3 * CENTRE, 1.7 * CENTRE, 301)
    wide = np.where(np.abs(grid - CENTRE) < 0.6 * CENTRE, 1.0, 0.0)
    fit = fit_bandpass(ref, grid, wide)
    assert float(np.max(fit.uncovered)) > 0.5
    term = fit.response_uncertainty()
    assert term.kind == "unbounded" and "not covered" in term.note
    inside = (grid > 0.9 * CENTRE) & (grid < 1.1 * CENTRE)
    cut = np.where(inside, 1.0, 0.0)[inside.argmax() - 2 :]  # starts and ends at 0
    assert fit_bandpass(ref, grid[inside.argmax() - 2 :], cut).open_edges == ()
    truncated = fit_bandpass(ref, grid[inside], np.ones(int(inside.sum())))
    assert truncated.open_edges == (0,)
    term = truncated.response_uncertainty()
    assert term.kind == "unbounded" and "beyond the table grid" in term.note
    with pytest.raises(ValueError):
        fit_bandpass(ref, grid, wide[:-1])
    with pytest.raises(ValueError):
        fit_bandpass(ref, grid[::-1], wide)
    with pytest.raises(ValueError):
        fit_bandpass(ref, grid, np.where(wide > 0, np.nan, 0.0))
    negative = Channels.from_callable(
        lambda nu: jnp.sin(nu / CENTRE * 20.0),
        (0.5 * CENTRE, 1.5 * CENTRE),
        smoothness=0,
    )
    with pytest.raises(ValueError, match="nonnegative"):
        fit_bandpass(negative, grid, wide)


# -- propagation of the response-error term -------------------------------------------


def test_propagate_accepts_a_response_uncertainty_term():
    rng = np.random.default_rng(4)
    stokes = np.abs(rng.standard_normal((3, 4)))
    total = ErrorTerm(np.full((3, 4), 1e-3), "bound", "budget")
    R = rng.standard_normal((5, 12))
    dR = np.abs(rng.standard_normal((5, 12))) * 1e-2
    plain = propagate_envelope(total, jnp.asarray(stokes), R, dR)
    as_term = propagate_envelope(
        total, jnp.asarray(stokes), R, ErrorTerm(dR, "estimate", "fit residual")
    )
    assert plain.kind == "bound" and as_term.kind == "estimate"
    # (R + dR) s_true - R s = R (s_true - s) + dR s_true, |s_true| <= |s| + e.
    envelope = np.full(12, 1e-3)
    want = np.abs(R) @ envelope + dR @ (stokes.ravel() + envelope)
    assert_allclose(np.asarray(plain.value), want, rtol=1e-14)
    assert_allclose(np.asarray(as_term.value), np.asarray(plain.value), rtol=1e-15)
    assert "fit residual" in as_term.note
    unbounded = propagate_envelope(
        total, jnp.asarray(stokes), R, ErrorTerm.unbounded("no covering")
    )
    assert unbounded.kind == "unbounded" and "no covering" in unbounded.note
    # A valueless term is never read as an exact response.
    skipped = propagate_envelope(
        total, jnp.asarray(stokes), R, ErrorTerm.not_applicable("no fit")
    )
    assert skipped.kind == "unbounded" and "not_applicable" in skipped.note
    with pytest.raises(ValueError):
        propagate_envelope(
            total, jnp.asarray(stokes), R, ErrorTerm(dR[:, :5], "bound", "")
        )
