"""Named assumptions: ``declare``, the ``assumptions=`` keyword and the
uncoupled case of ``eq: uncoupled moments``.

Oracles: an independent NumPy counter of free parameters (distinct
nonconstant projections of the kept rows of ``m`` onto every free group, two
reals per ``e^{2i phi}`` entry); the counts measured with a raw
``Factorisation`` in the package survey of 2026-10-27 (``(2, 2, 2)``: 33,
25, 86; zero ``fixed_table`` on ``eta`` or ``phi``: 77, 73; ``(0, 2, 2)``
with ``(I, Q)``: 26, 25, 29, 25, 19); the manuscript count below ``eq:
uncoupled moments`` (``N = 2``, ``L_mu = L_eta = 8``: 410 polarisation
moments, 41 angular coefficients, 10 non-angular moments); exact weighted
sums over product populations; and the maps built without the keyword.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.special import eval_legendre

from _linear_fixtures import setup as linear_setup
from syncem.model.assumptions import (
    ASSUMPTIONS,
    VARS,
    Assume,
    Closure,
    Factorisation,
    ParameterMap,
    Parameters,
    angles_independent,
    angular_table,
    declare,
    isotropic_pitch,
    known_angles,
    known_field,
    pitch_symmetric,
    uniform_azimuth,
    uniform_field_direction,
)
from syncem.model.errors import ErrorTerm
from syncem.model.fit import (
    LogDensity,
    StokesData,
    fit_bfgs,
    fit_combinations,
    fit_linear,
)
from syncem.model.index import MomentIndex, Truncation
from syncem.model.moments import JointMoments, PopulationSamples, Reference
from syncem.model.predict import predict

REF = Reference(5.2, 2.1, 0.3, scales=(1.3, 0.7, 0.9))
POSITION = {"phi": 0, "mu": 1, "eta": 2, "gamma": 3, "B": 4, "depth": 5}
ANGLES, NON_ANGULAR = ("mu", "eta"), ("gamma", "B", "phi", "depth")
NEW = (
    "angles_independent",
    "known_angles",
    "uniform_field_direction",
    "uniform_azimuth",
    "known_field",
)
PARAMS = {"gaussian_screen": {"mean": 0.4, "sigma": 0.6}, "known_field": {"B": 2.4}}


def coefficients(entries):
    """``(3, 3)`` ``a_lk`` with ``a_00 = 1`` and the given ``{(l, k): value}``."""
    a = np.zeros((3, 3))
    a[0, 0] = 1.0
    for (l, k), value in entries.items():
        a[l, k] = value
    return a


def spec(name):
    return Assume(name, **PARAMS.get(name, {}))


def rest(v):
    return tuple(u for u in VARS if u != v)


@pytest.fixture(scope="module")
def index_882():
    return MomentIndex.build(Truncation(8, 8, 2))


@pytest.fixture(scope="module")
def index_022():
    return MomentIndex.build(Truncation(0, 2, 2), components=("I", "Q"))


# -- free counts ------------------------------------------------------------------


def count_free_np(index, groups, closed=(), drop=lambda row: False):
    """Independent counter: distinct nonconstant projections of kept rows."""
    rows = [(0, *r) for r in index.h0] + [(2, *r) for r in index.h2]
    rows = [r for r in rows if not drop(r)]
    total = 0
    for group in groups:
        if group in closed:
            continue
        projections = {tuple(row[POSITION[v]] for v in group) for row in rows}
        projections.discard((0,) * len(group))
        for p in projections:
            complex_entry = "phi" in group and p[group.index("phi")] == 2
            total += 2 if complex_entry else 1
    return total


ORACLE = {
    "angles_independent": ((ANGLES, NON_ANGULAR), (), lambda r: False),
    "known_angles": ((ANGLES, NON_ANGULAR), (ANGLES,), lambda r: False),
    "uniform_field_direction": ((("eta",), rest("eta")), (("eta",),), lambda r: r[2]),
    "uniform_azimuth": ((("phi",), rest("phi")), (("phi",),), lambda r: r[0] == 2),
    "known_field": ((("B",), rest("B")), (("B",),), lambda r: False),
}
SURVEY_222 = {"angles_independent": 33, "known_angles": 25, "known_field": 86}
STATIC_222 = {"uniform_field_direction": 57, "uniform_azimuth": 53}
COUNTS_882 = {
    "angles_independent": 105,
    "known_angles": 25,
    "uniform_field_direction": 153,
    "uniform_azimuth": 485,
    "known_field": 734,
}
AFFINE = {
    "angles_independent": False,
    "known_angles": True,
    "uniform_field_direction": True,
    "uniform_azimuth": True,
    "known_field": True,
}


@pytest.mark.parametrize("name", NEW)
def test_new_counts_at_222_match_survey_and_counter(index_222, name):
    pm = declare(index_222, spec(name))
    expected = {**SURVEY_222, **STATIC_222}[name]
    groups, closed, drop = ORACLE[name]
    assert pm.n_free() == expected == count_free_np(index_222, groups, closed, drop)
    assert len(set(pm.labels())) == expected and pm.record().n_free == expected
    assert pm.is_affine() is AFFINE[name] and pm.name == name


@pytest.mark.parametrize("name", NEW)
def test_new_counts_at_882_match_counter(index_882, name):
    pm = declare(index_882, spec(name))
    assert pm.n_free() == COUNTS_882[name] == count_free_np(index_882, *ORACLE[name])
    assert pm.is_affine() is AFFINE[name]


def test_manuscript_count_below_uncoupled_moments(index_882):
    """410 polarisation moments = products of 41 angular and 10 non-angular."""
    h2 = index_882.h2
    assert len(h2) == index_882.n2 == 410
    assert len({row[:2] for row in h2}) == 41
    assert len({row[2:] for row in h2}) == 10
    pm = angles_independent(index_882)
    assert pm.n_free() == 80 + 25  # all (l, k) != (0, 0) of I, Q, V; 5 + 2 x 10


def test_counts_with_i_q_only(index_022):
    survey = {"angles_independent": 26, "known_angles": 25, "known_field": 29}
    static = {"uniform_field_direction": 25, "uniform_azimuth": 11}
    for name, expected in {**survey, **static}.items():
        pm = declare(index_022, spec(name))
        assert pm.n_free() == expected == count_free_np(index_022, *ORACLE[name])


def _zero_table_map(index, variable):
    plain = ParameterMap.build(index, Factorisation(((variable,), rest(variable))))
    n = len(plain.tables_spec[plain.factorisation.groups.index((variable,))])
    zero = Closure((variable,), "fixed_table", (jnp.zeros(n),))
    return ParameterMap.build(
        index, Factorisation(((variable,), rest(variable))), (zero,)
    )


@pytest.mark.parametrize(
    "variable,builder,survey",
    [("eta", uniform_field_direction, 77), ("phi", uniform_azimuth, 73)],
)
def test_static_drop_removes_only_dead_parameters(index_222, variable, builder, survey):
    """A zero ``fixed_table`` keeps 20 parameters that multiply a zero factor."""
    zero = _zero_table_map(index_222, variable)
    static = builder(index_222)
    assert zero.n_free() == survey and static.n_free() == survey - 20
    P0, c0 = (np.asarray(x) for x in zero.affine_pieces())
    P1, c1 = (np.asarray(x) for x in static.affine_pieces())
    live = np.any(P0 != 0.0, axis=0)
    assert int(np.sum(~live)) == 20
    assert tuple(np.asarray(zero.labels())[live]) == static.labels()
    assert_array_equal(P0[:, live], P1)
    assert_array_equal(c0, c1)


@pytest.mark.parametrize("name", [n for n in ASSUMPTIONS if n not in NEW])
@pytest.mark.parametrize("which", ["222", "882"])
def test_registry_names_build_the_named_constructors(index_222, index_882, name, which):
    index = index_222 if which == "222" else index_882
    pm = declare(index, spec(name))
    direct = ASSUMPTIONS[name](index, **PARAMS.get(name, {}))
    assert pm.n_free() == direct.n_free() and pm.record() == direct.record()
    assert pm.name == name and pm.labels() == direct.labels()


def test_existing_counts_through_declare(index_222, index_882):
    at_222 = {"no_assumption": 153, "isotropic_pitch": 57, "independent_screen": 115}
    at_222 |= {"field_reversal_symmetric": 129, "fully_independent": 12}
    for name, n in at_222.items():
        assert declare(index_222, name).n_free() == n
    at_882 = {"no_assumption": 1305, "pitch_symmetric": 769, "fully_independent": 24}
    for name, n in at_882.items():
        assert declare(index_882, name).n_free() == n
    combined = declare(index_222, ("angles_independent", "independent_screen"))
    assert combined.n_free() == 27  # survey value


# -- known angles against isotropic pitch + uniform field direction ------------------


@pytest.mark.parametrize("which", ["222", "882", "022"])
def test_isotropic_known_angles_equal_uniform_closures(
    index_222, index_882, index_022, which
):
    index = {"222": index_222, "882": index_882, "022": index_022}[which]
    table = known_angles(index, angular_table(index))
    closures = isotropic_pitch(index).assume(uniform_field_direction(index))
    assert table.n_free() == closures.n_free() == 25
    assert table.labels() == closures.labels()
    for a, b in zip(table.affine_pieces(), closures.affine_pieces()):
        assert_array_equal(np.asarray(a), np.asarray(b))
    theta = table.unflatten(jnp.asarray(np.random.default_rng(3).normal(size=25)))
    m_table, m_closures = table(theta, REF), closures(theta, REF)
    assert_array_equal(
        np.asarray(m_table.to_vector()), np.asarray(m_closures.to_vector())
    )
    assert table.is_affine() and closures.is_affine()
    assert_array_equal(np.asarray(known_angles(index).closures[0].hyper[0]), 0.0)


@pytest.mark.parametrize("which", ["222", "882"])
def test_zero_angular_table_leaves_the_azimuth_free(index_222, index_882, which):
    """``a_lk = None`` is isotropic pitch and uniform ``eta`` only: the 20
    ``h = 2`` parameters stay free, and an isotropic field direction also
    needs ``uniform_azimuth`` (5 free parameters, the counter's value)."""
    index = index_222 if which == "222" else index_882
    zero = known_angles(index)
    assert zero.n_free() == 25
    assert sum("phi" in label for label in zero.labels()) == 20
    isotropic = known_angles(index).assume(uniform_azimuth(index))
    by_name = declare(index, ("known_angles", "uniform_azimuth"))
    closures = declare(
        index, ("isotropic_pitch", "uniform_field_direction", "uniform_azimuth")
    )
    groups = (("gamma", "B", "depth"), ANGLES, ("phi",))
    counted = count_free_np(index, groups, (ANGLES, ("phi",)), lambda r: r[0] == 2)
    assert isotropic.n_free() == by_name.n_free() == closures.n_free() == 5 == counted
    assert isotropic.labels() == by_name.labels() == closures.labels()
    assert not any("phi" in label for label in isotropic.labels())
    for a, b in zip(isotropic.affine_pieces(), closures.affine_pieces()):
        assert_array_equal(np.asarray(a), np.asarray(b))
    theta = isotropic.unflatten(jnp.asarray(np.random.default_rng(4).normal(size=5)))
    assert_array_equal(np.asarray(isotropic(theta, REF).m2), 0.0)


def test_angular_table_layout_and_coefficients(index_222):
    a = np.arange(9.0).reshape(3, 3) / 10.0
    a[0, 0] = 1.0
    table = np.asarray(angular_table(index_222, a))
    spec_ = angles_independent(index_222).tables_spec[1]
    assert spec_ == ((0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2))
    expected = [a[l, k] / ((2 * l + 1) * (2 * k + 1)) for l, k in spec_]
    assert_allclose(table, expected, rtol=0, atol=0)
    pm = known_angles(index_222, a_lk=a)
    assert_array_equal(np.asarray(pm.closures[0].hyper[0]), table)
    assert pm.record().hyper == tuple(float(x) for x in table)


# -- product and correlated populations ------------------------------------------------


def _outer(parts):
    sizes = [len(p["w"]) for p in parts]
    out, w = {}, np.ones(int(np.prod(sizes)))
    for part, idx in zip(parts, np.meshgrid(*map(np.arange, sizes), indexing="ij")):
        for key, value in part.items():
            if key == "w":
                w = w * np.asarray(value)[idx.ravel()]
            else:
                out[key] = np.asarray(value)[idx.ravel()]
    out["w"] = w
    return out


def _rest_atoms(rng, n, keep):
    t = rng.uniform(-1, 1, n)
    values = dict(
        gamma=5.0 + 1.5 * t,
        B=2.0 + 0.8 * t**2 + 0.1 * rng.standard_normal(n),
        mu=np.clip(0.6 * t + 0.2 * rng.standard_normal(n), -0.95, 0.95),
        eta=np.clip(-0.5 * t + 0.3 * rng.standard_normal(n), -0.95, 0.95),
        phi=0.3 + 1.7 * t,
        depth=1.5 * t + 0.2 * rng.standard_normal(n),
    )
    return {**{k: values[k] for k in keep}, "w": rng.uniform(0.2, 3.0, n)}


def _samples(pop):
    keys = ("gamma", "B", "mu", "eta", "phi", "depth")
    return PopulationSamples(*(pop[k] for k in keys), weights=pop["w"])


def _population(name, seed=5):
    rng = np.random.default_rng(seed)
    if name in ("angles_independent", "known_angles"):
        t = rng.uniform(-1, 1, 5)
        angles = dict(
            mu=0.8 * t, eta=np.clip(0.3 - 0.6 * t**2, -1, 1), w=rng.uniform(0.5, 2, 5)
        )
        return _outer([angles, _rest_atoms(rng, 6, NON_ANGULAR)])
    if name == "uniform_field_direction":
        x, w = leggauss(5)  # exact for P_k(eta), k <= 9
        return _outer([dict(eta=x, w=w), _rest_atoms(rng, 6, rest("eta"))])
    if name == "uniform_azimuth":
        phi = dict(phi=np.arange(4) * np.pi / 2, w=np.ones(4))
        return _outer([phi, _rest_atoms(rng, 6, rest("phi"))])
    return _outer(
        [dict(B=np.array([2.4]), w=np.ones(1)), _rest_atoms(rng, 6, rest("B"))]
    )


def _a_lk(pop, index):
    t = index.truncation
    w = pop["w"] / pop["w"].sum()
    return np.array(
        [
            [
                (2 * l + 1)
                * (2 * k + 1)
                * np.sum(w * eval_legendre(l, pop["mu"]) * eval_legendre(k, pop["eta"]))
                for k in range(t.L_eta + 1)
            ]
            for l in range(t.L_mu + 1)
        ]
    )


def _declared(name, index, pop):
    if name == "known_angles":
        return Assume("known_angles", a_lk=_a_lk(pop, index))
    return spec(name)


@pytest.mark.parametrize("which", ["222", "882"])
@pytest.mark.parametrize("name", NEW)
def test_product_population_reproduces_every_moment(index_222, index_882, name, which):
    index = index_222 if which == "222" else index_882
    pop = _population(name)
    joint = JointMoments.from_samples(_samples(pop), index, REF)
    assumption = _declared(name, index, pop)
    measured = JointMoments.from_samples(
        _samples(pop), index, REF, assumptions=assumption, discrepancy="measured"
    )
    scale = np.max(np.abs(np.asarray(joint.to_vector())))
    assert_allclose(measured.to_vector(), joint.to_vector(), rtol=0, atol=1e-13 * scale)
    assert measured.discrepancy.kind == "measured"
    assert np.max(np.asarray(measured.discrepancy.value)) <= 1e-13 * scale
    assert measured.assumptions == (declare(index, assumption).record(),)
    assert measured.assumptions[0].name == name
    if name == "uniform_azimuth":
        assert np.max(np.abs(np.asarray(joint.m2))) < 1e-14
        assert_array_equal(np.asarray(measured.m2), 0.0)


def _correlated(seed=8, n=400):
    """``mu`` follows ``gamma``; the other variables are independent."""
    rng = np.random.default_rng(seed)
    gamma = 5.0 + 1.5 * rng.uniform(-1, 1, n)
    mu = np.clip(0.5 * (gamma - 5.0) + 0.1 * rng.standard_normal(n), -0.98, 0.98)
    pop = dict(
        gamma=gamma,
        B=2.0 + 0.5 * rng.uniform(-1, 1, n),
        mu=mu,
        eta=rng.uniform(-0.9, 0.9, n),
        phi=rng.uniform(0, 2 * np.pi, n),
        depth=rng.uniform(-1, 1, n),
        w=rng.uniform(0.5, 2.0, n),
    )
    return pop


@pytest.mark.parametrize("name", ["angles_independent", "known_angles"])
def test_correlated_population_gives_a_measured_discrepancy(index_222, name):
    pop = _correlated()
    joint = JointMoments.from_samples(_samples(pop), index_222, REF)
    assumption = _declared(name, index_222, pop)
    pm = declare(index_222, assumption)
    factorised = pm(pm.project(joint), REF)
    difference = np.abs(np.asarray(joint.to_vector() - factorised.to_vector()))
    assert np.max(difference) > 1e-2
    measured = JointMoments.from_samples(
        _samples(pop), index_222, REF, assumptions=assumption, discrepancy="measured"
    )
    assert measured.discrepancy.kind == "measured"
    assert_allclose(measured.discrepancy.value, difference, rtol=0, atol=1e-15)
    plain = JointMoments.from_samples(
        _samples(pop), index_222, REF, assumptions=assumption
    )
    assert plain.discrepancy is None
    assert_allclose(plain.to_vector(), factorised.to_vector(), rtol=0, atol=1e-15)


# -- composition and order -----------------------------------------------------------------

COMBINATIONS = [
    ("known_angles", "known_field"),
    ("uniform_azimuth", "isotropic_pitch", "uniform_field_direction"),
    ("independent_screen", "angles_independent"),
    ("pitch_symmetric", "known_angles"),
    ("field_reversal_symmetric", "uniform_azimuth", "known_field"),
    ("gaussian_screen", "known_angles"),
]


@pytest.mark.parametrize("names", COMBINATIONS, ids="+".join)
def test_declare_is_order_independent(index_222, names):
    maps = [
        declare(index_222, tuple(spec(n) for n in order))
        for order in itertools.permutations(names)
    ]
    first = maps[0]
    order = tuple(ASSUMPTIONS)
    assert first.name == "+".join(sorted(names, key=order.index))
    theta = first.unflatten(
        jnp.asarray(np.random.default_rng(1).normal(size=first.n_free()))
    )
    m = np.asarray(first(theta, REF).to_vector())
    for pm in maps[1:]:
        assert pm.record() == first.record()
        assert pm.labels() == first.labels() and pm.tables_spec == first.tables_spec
        assert_array_equal(np.asarray(pm(theta, REF).to_vector()), m)
    assert first.discrepancy is None and first.record().discrepancy_kind == "unbounded"


def test_declare_equals_assume_and_carries_a_supplied_discrepancy(index_222):
    by_name = declare(index_222, ("isotropic_pitch", "uniform_field_direction"))
    by_hand = isotropic_pitch(index_222).assume(uniform_field_direction(index_222))
    assert by_name.record() == by_hand.record()
    term = ErrorTerm(jnp.full(index_222.n_real, 1e-3), "bound", "declared allowance")
    carried = declare(
        index_222, ("isotropic_pitch", "uniform_field_direction"), discrepancy=term
    )
    assert carried.discrepancy is term and carried.record().discrepancy_kind == "bound"
    assert carried.labels() == by_hand.labels()
    single = declare(index_222, "known_angles", discrepancy=term)
    assert single.record().discrepancy_kind == "bound" and single.name == "known_angles"


# -- JIT and autodiff ----------------------------------------------------------------------


@pytest.mark.parametrize("name", NEW)
def test_jit_and_grad_through_call(index_222, name):
    pm = declare(index_222, spec(name))
    rng = np.random.default_rng(7)
    theta = pm.unflatten(jnp.asarray(0.4 * rng.standard_normal(pm.n_free())))
    probe = rng.standard_normal(index_222.n_real)

    def scalar(flat):
        return jnp.dot(probe, pm(pm.unflatten(flat), REF).to_vector())

    flat = pm.flatten(theta)
    assert_allclose(jax.jit(scalar)(flat), scalar(flat), rtol=1e-13)
    grad = np.asarray(jax.jit(jax.grad(scalar))(flat))
    assert np.all(np.isfinite(grad))
    step = 1e-6
    for i in range(0, flat.size, max(1, flat.size // 5)):
        e = jnp.zeros(flat.size).at[i].set(step)
        fd = (scalar(flat + e) - scalar(flat - e)) / (2 * step)
        assert_allclose(grad[i], fd, rtol=1e-6, atol=1e-8)
    if pm.is_affine():
        P, _ = pm.affine_pieces(reference=REF)
        assert_allclose(grad, probe @ np.asarray(P), rtol=1e-12, atol=1e-12)


def test_grad_through_closure_values(index_222):
    """``a_lk`` and ``B`` are traced leaves of the rebuilt maps."""
    rng = np.random.default_rng(2)
    rest_theta = known_angles(index_222).unflatten(jnp.asarray(rng.normal(size=25)))
    probe = rng.standard_normal(index_222.n_real)

    def of_a(a):
        pm = known_angles(index_222, a_lk=a)
        return jnp.dot(probe, pm(rest_theta, REF).to_vector())

    a = np.eye(3) * 0.2
    a[0, 0] = 1.0
    grad = np.asarray(jax.jit(jax.grad(of_a))(jnp.asarray(a)))
    assert grad.shape == (3, 3) and np.all(np.isfinite(grad))
    for l, k in [(0, 1), (1, 1), (2, 0), (2, 2)]:
        e = np.zeros((3, 3))
        e[l, k] = 1e-6
        fd = (of_a(jnp.asarray(a + e)) - of_a(jnp.asarray(a - e))) / 2e-6
        assert_allclose(grad[l, k], fd, rtol=1e-6, atol=1e-9)
    theta_B = known_field(index_222, 2.0).unflatten(jnp.asarray(rng.normal(size=86)))

    def of_B(B):
        return jnp.dot(probe, known_field(index_222, B)(theta_B, REF).to_vector())

    g = float(jax.jit(jax.grad(of_B))(2.0))
    fd = float((of_B(2.0 + 1e-6) - of_B(2.0 - 1e-6)) / 2e-6)
    assert np.isfinite(g) and abs(g - fd) <= 1e-6 * max(1.0, abs(fd))


# -- fits with the keyword ------------------------------------------------------------------


@pytest.fixture(scope="module")
def known_angles_problem():
    index, basis, C = linear_setup((2, 2, 2), n_ch=24, seed=3)
    a = np.zeros((3, 3))
    a[0, 0], a[2, 0], a[0, 2], a[1, 1] = 1.0, 0.4, -0.3, 0.2
    pm = known_angles(index, a_lk=a)
    rng = np.random.default_rng(11)
    theta = pm.unflatten(jnp.asarray(0.3 * rng.standard_normal(pm.n_free())))
    clean = 2.0 * np.asarray(C) @ np.asarray(pm(theta, basis.reference).to_vector())
    clean = clean.reshape(-1, 4)
    sigma = 1e-3 * np.abs(clean).max()
    noisy = clean + sigma * rng.standard_normal(clean.shape)
    data = StokesData(noisy, np.full(clean.size, sigma**2))
    return basis, data, pm, a


def test_fit_linear_keyword_equals_parameter_map(known_angles_problem):
    basis, data, pm, a = known_angles_problem
    by_map = fit_linear(basis, data, pm)
    by_name = fit_linear(basis, data, assumptions=Assume("known_angles", a_lk=a))
    assert by_name.rank == by_map.rank == 26
    assert_array_equal(np.asarray(by_name.amplitude), np.asarray(by_map.amplitude))
    assert_array_equal(
        np.asarray(pm.flatten(by_name.theta)), np.asarray(pm.flatten(by_map.theta))
    )
    assert by_name.labels == by_map.labels and float(by_name.chi2) == float(by_map.chi2)
    sigma_A = float(np.sqrt(np.asarray(by_name.covariance)[0, 0]))
    assert abs(float(by_name.amplitude) - 2.0) < 5.0 * sigma_A
    terms = dict(by_name.prediction.budget.assumption)
    assert terms["known_angles"].kind == "unbounded"
    assert [r.name for r in by_name.provenance.assumptions] == ["known_angles"]


def test_fit_combinations_keyword_equals_parameter_map(known_angles_problem):
    basis, data, pm, a = known_angles_problem
    by_map = fit_combinations(basis, data, max_sigma=np.inf, parameter_map=pm)
    by_name = fit_combinations(
        basis, data, max_sigma=np.inf, assumptions=Assume("known_angles", a_lk=a)
    )
    for field in ("x_hat", "covariance", "singular_values", "jacobian", "stokes_hat"):
        assert_array_equal(
            np.asarray(getattr(by_name, field)), np.asarray(getattr(by_map, field))
        )
    assert by_name.labels == by_map.labels and by_name.n_retained == by_map.n_retained
    assert by_name.provenance.assumptions == by_map.provenance.assumptions
    assert pm.record() in by_name.provenance.assumptions
    isotropic = fit_combinations(
        basis, data, max_sigma=np.inf, assumptions="known_angles"
    )
    reference = fit_combinations(
        basis, data, max_sigma=np.inf, parameter_map=known_angles(basis.index)
    )
    assert_array_equal(np.asarray(isotropic.x_hat), np.asarray(reference.x_hat))


def test_allowance_replaces_the_named_term(known_angles_problem):
    basis, data, pm, a = known_angles_problem
    fit = fit_linear(basis, data, assumptions=Assume("known_angles", a_lk=a))
    allowance = ErrorTerm(np.full((basis.n_ch, 4), 1e-6), "bound", "declared")
    pred = predict(
        basis,
        fit.moments,
        amplitude=fit.amplitude,
        assumption_allowances={"known_angles": allowance},
    )
    term = dict(pred.budget.assumption)["known_angles"]
    assert term.kind == "bound"
    assert_allclose(np.asarray(term.value), 1e-6)
    names = [name for name, _ in pred.budget.terms()]
    assert "assumption:known_angles" in names
    plain = predict(basis, fit.moments, amplitude=fit.amplitude)
    assert "assumption:known_angles" in plain.budget.unbounded()


def test_combined_and_measured_budget_terms(known_angles_problem):
    """A combined name is one record, ``unbounded`` until measured or supplied."""
    basis, data, pm, a = known_angles_problem
    pop = _correlated(n=60)
    combined = ("known_angles", Assume("known_field", B=2.4))
    moments = JointMoments.from_samples(
        _samples(pop), basis.index, basis.reference, assumptions=combined
    )
    budget = predict(basis, moments, amplitude=1.0).budget
    assert "assumption:known_angles+known_field" in budget.unbounded()
    measured = JointMoments.from_samples(
        _samples(pop),
        basis.index,
        basis.reference,
        assumptions=combined,
        discrepancy="measured",
    )
    term = dict(predict(basis, measured, amplitude=1.0).budget.assumption)
    assert term["known_angles+known_field"].kind == "measured"
    assert float(np.max(np.asarray(term["known_angles+known_field"].value))) > 0.0


def test_constrained_basis(known_angles_problem):
    basis, data, pm, a = known_angles_problem
    CP, Cc, labels = basis.constrained(pm)
    P, c = pm.affine_pieces(reference=basis.reference)
    C = np.asarray(basis.response_matrix())
    scale = np.max(np.abs(C))
    assert_allclose(CP, C @ np.asarray(P), rtol=0, atol=1e-14 * scale)
    assert_allclose(Cc, C @ np.asarray(c), rtol=0, atol=1e-14 * scale)
    assert labels == pm.labels() and CP.shape == (4 * basis.n_ch, 25)
    CP2, Cc2, labels2 = basis.constrained(assumptions=Assume("known_angles", a_lk=a))
    assert_array_equal(np.asarray(CP2), np.asarray(CP)) and labels2 == labels


def test_log_density_keyword_and_bfgs_on_the_bilinear_map(known_angles_problem):
    """``angles_independent`` (non-affine) through ``LogDensity(assumptions=)``."""
    basis, _, _, _ = known_angles_problem
    pm = angles_independent(basis.index)
    rng = np.random.default_rng(13)
    theta = pm.unflatten(jnp.asarray(0.2 * rng.standard_normal(pm.n_free())))
    C = np.asarray(basis.response_matrix())
    clean = 2.0 * C @ np.asarray(pm(theta, basis.reference).to_vector())
    clean = clean.reshape(-1, 4)
    sigma = 1e-3 * np.abs(clean).max()
    noisy = clean + sigma * rng.standard_normal(clean.shape)
    data = StokesData(noisy, np.full(clean.size, sigma**2))
    by_name = LogDensity(basis, data, assumptions=("angles_independent",))
    by_map = LogDensity(basis, data, pm)
    assert by_name.n_params == by_map.n_params == 34
    assert by_name.parameter_map.record() == pm.record()
    z0 = by_name.transform.forward(
        Parameters(log_amplitude=jnp.log(1.5), tables=theta.tables)
    )
    assert float(by_name(z0)) == float(by_map(z0))
    fit = fit_bfgs(by_name, z0, maxiter=5, restarts=0, diagnostics=False)
    assert 1 <= fit.n_iter <= 5
    assert np.isfinite(float(fit.fun)) and float(fit.fun) < float(-by_name(z0))
    assert [r.name for r in fit.provenance.assumptions] == ["angles_independent"]


# -- refusals ---------------------------------------------------------------------------------


def test_keyword_and_parameter_map_are_exclusive(known_angles_problem, index_222):
    basis, data, pm, a = known_angles_problem
    with pytest.raises(ValueError, match="not both"):
        fit_linear(basis, data, pm, assumptions="known_angles")
    with pytest.raises(ValueError, match="not both"):
        fit_combinations(
            basis, data, max_sigma=1.0, parameter_map=pm, assumptions="known_angles"
        )
    with pytest.raises(ValueError, match="not both"):
        basis.constrained(pm, assumptions="known_angles")
    with pytest.raises(ValueError, match="parameter_map or assumptions"):
        fit_linear(basis, data)
    with pytest.raises(ValueError, match="not both"):
        LogDensity(basis, data, pm, assumptions="known_angles")
    with pytest.raises(ValueError, match="or pass assumptions="):
        LogDensity(basis, data)
    samples = _samples(_population("known_angles"))
    with pytest.raises(ValueError, match="not both"):
        JointMoments.from_samples(
            samples,
            index_222,
            REF,
            parameter_map=isotropic_pitch(index_222),
            assumptions="known_angles",
        )


def test_bilinear_map_is_refused_by_linear_routes(known_angles_problem):
    basis, data, pm, a = known_angles_problem
    calls = (
        lambda: fit_linear(basis, data, assumptions="angles_independent"),
        lambda: fit_combinations(
            basis, data, max_sigma=1.0, assumptions="angles_independent"
        ),
        lambda: basis.constrained(assumptions="angles_independent"),
        lambda: fit_linear(basis, data, angles_independent(basis.index)),
    )
    for call in calls:
        with pytest.raises(ValueError, match="bilinear") as error:
            call()
        assert "known_angles" in str(error.value) and "fit_bfgs" in str(error.value)
        assert "LogDensity(basis, data, assumptions=...)" in str(error.value)
    with pytest.raises(ValueError, match="fit_bfgs") as error:
        fit_linear(basis, data, assumptions="independent_screen")
    assert "bilinear" not in str(error.value)


FITTED_SCREEN = Assume("gaussian_screen", mean=0.4, sigma=0.6, fit_hyper=True)
# (assumptions, hint): the hint appears iff known_angles makes the map affine
HINT_CASES = [
    (("angles_independent",), True),
    (("angles_independent", Assume("known_field", B=2.4)), True),
    (("angles_independent", "uniform_azimuth"), True),
    (("angles_independent", "pitch_symmetric"), True),
    (("angles_independent", "field_reversal_symmetric"), True),
    (("angles_independent", "independent_screen"), False),  # three free groups
    (("angles_independent", FITTED_SCREEN), False),  # fitted hyper-parameters
    (("fully_independent",), False),  # mu and eta apart: known_angles is split
    (("isotropic_pitch", "independent_screen"), False),
]


@pytest.mark.parametrize(
    "names,hint", HINT_CASES, ids=[str(i) for i in range(len(HINT_CASES))]
)
def test_known_angles_hint_only_when_it_makes_the_map_affine(
    known_angles_problem, names, hint
):
    basis = known_angles_problem[0]
    with pytest.raises(ValueError, match="not affine") as error:
        basis.constrained(assumptions=names)
    message = str(error.value)
    assert "fit_bfgs" in message
    assert ("bilinear" in message) is hint and ("known_angles" in message) is hint
    if "angles_independent" in names:
        swapped = tuple(
            "known_angles" if n == "angles_independent" else n for n in names
        )
        assert declare(basis.index, swapped).is_affine() is hint
    else:
        with pytest.raises(ValueError, match="split by the refinement"):
            declare(basis.index, (*names, "known_angles"))


@pytest.mark.parametrize(
    "assumptions,match",
    [
        ("isotropic", "unknown assumption"),
        ((), "empty"),
        (("known_angles", "known_angles"), "more than once"),
        ((isotropic_pitch,), "must be a name"),
        (3, "must be a name"),
        ("known_field", "missing"),
        ("gaussian_screen", "missing"),
        (Assume("isotropic_pitch", foo=1), "unexpected keyword"),
        (("known_angles", "isotropic_pitch"), "split by the refinement"),
        (("known_angles", "uniform_field_direction"), "split by the refinement"),
        (
            (
                "pitch_symmetric",
                Assume("known_angles", a_lk=coefficients({(1, 0): 0.3})),
            ),
            r"<P_1\(mu\)> = 0.1, which pitch_symmetric sets to zero",
        ),
        (
            (
                "field_reversal_symmetric",
                Assume("known_angles", a_lk=coefficients({(0, 1): 0.3})),
            ),
            r"<P_1\(eta\)> = 0.1, which field_reversal_symmetric sets to zero",
        ),
        (
            (
                "field_reversal_symmetric",
                Assume("known_angles", a_lk=coefficients({(1, 2): -0.6})),
            ),
            r"<P_1\(mu\) P_2\(eta\)> = -0.04, which field_reversal_symmetric",
        ),
    ],
)
def test_declare_refusals(index_222, assumptions, match):
    with pytest.raises(ValueError, match=match):
        declare(index_222, assumptions)


def test_symmetry_check_spares_kept_entries_traced_tables_and_assume(index_222):
    """``declare`` refuses only a concrete entry the symmetry sets to zero;
    ``assume`` keeps the symmetrised semantics (the entry is ignored)."""
    even = coefficients({(1, 1): 0.2, (2, 0): 0.4, (0, 2): -0.3, (2, 2): 0.1})
    reversal = declare(
        index_222, ("field_reversal_symmetric", Assume("known_angles", a_lk=even))
    )
    pitch = declare(
        index_222,
        ("pitch_symmetric", Assume("known_angles", a_lk=coefficients({(0, 1): 0.3}))),
    )
    assert reversal.n_free() == pitch.n_free() == 25
    odd = coefficients({(1, 0): 0.3})
    reference = declare(index_222, ("pitch_symmetric", "known_angles"))
    rng = np.random.default_rng(6)
    theta = reference.unflatten(jnp.asarray(rng.normal(size=reference.n_free())))

    def moments(a):
        names = ("pitch_symmetric", Assume("known_angles", a_lk=a))
        return declare(index_222, names)(theta, REF).to_vector()

    traced = np.asarray(jax.jit(moments)(jnp.asarray(odd)))
    eager = np.asarray(reference(theta, REF).to_vector())
    assert_allclose(traced, eager, rtol=1e-14, atol=1e-15)
    assumed = known_angles(index_222, a_lk=odd).assume(pitch_symmetric(index_222))
    assert_array_equal(np.asarray(assumed(theta, REF).to_vector()), eager)


def test_constructor_refusals(index_222):
    with pytest.raises(ValueError, match="unknown assumption"):
        Assume("known_angle")
    with pytest.raises(AttributeError):
        Assume("known_angles").name = "x"
    with pytest.raises(ValueError, match="not both"):
        known_angles(index_222, angular_table(index_222), a_lk=np.eye(3))
    with pytest.raises(ValueError, match="shape"):
        angular_table(index_222, np.eye(2))
    bad = np.eye(3)
    bad[0, 0] = 0.9
    with pytest.raises(ValueError, match=r"a_lk\[0, 0\]"):
        angular_table(index_222, bad)
    with pytest.raises(ValueError, match="finite"):
        angular_table(index_222, np.full((3, 3), np.nan))
    with pytest.raises(ValueError, match="needs shape"):
        known_angles(index_222, jnp.zeros(3))
    for B in (0.0, -1.0, np.inf):
        with pytest.raises(ValueError, match="B > 0"):
            known_field(index_222, B)
    with pytest.raises(ValueError, match="scalar"):
        known_field(index_222, np.ones(2))
    for closure in (
        Closure(("mu",), "uniform_eta"),
        Closure(("eta",), "uniform_eta", (0.1,)),
        Closure(("eta",), "uniform_phi"),
    ):
        groups = Factorisation(((closure.group[0],), rest(closure.group[0])))
        with pytest.raises(ValueError, match="closes the group"):
            ParameterMap.build(index_222, groups, (closure,))


def test_repr_and_records(index_222):
    assert repr(Assume("known_field", B=2.0)) == "Assume('known_field', B=2.0)"
    record = uniform_azimuth(index_222).record()
    assert record.closure_kind == "uniform_phi" and record.label == "[extension]"
    assert known_angles(index_222).record().label == "eq: uncoupled moments"
    assert uniform_field_direction(index_222).record().closure_kind == "uniform_eta"
