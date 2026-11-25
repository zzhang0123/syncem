"""The physical energy partition and the critical-frequency measure at cell faces."""

import numpy as np
import pytest

import syncem as se
from syncem.base.window import A1_HZ_PER_G
from tests.v090.test_regime_review_regressions import _population, _regime


def _cell_masses(basis, moments):
    lattice = basis.lattice
    return np.array(
        [
            lattice.weights[cell] * moments.values[lattice.index.cell_rows(cell).start]
            for cell in range(lattice.n_cells)
        ]
    )


@pytest.mark.parametrize("conditional_field", [False, True])
def test_conditional_energy_splits_parent_at_switch_and_harmonic_faces(
    conditional_field,
):
    def energy(B, eta):
        return se.Delta(300.0 + (B.value - eta.value - 5.99998))

    def field(eta):
        return se.Uniform(4.0 + eta.value, 6.0 + eta.value, unit="uG")

    eta_max = 0.125 if conditional_field else 0.0
    eta = se.Uniform(-eta_max, eta_max) if conditional_field else 0.0
    pop = _population(se.Conditional(energy, units={"B": "uG"}), eta=eta).replace(
        B=se.Conditional(field) if conditional_field else se.Uniform(4, 6, unit="uG")
    )
    rb = _regime(
        pop,
        se.Channels.at([1.0], unit="MHz", half_width=0.12, n_nu=6),
        se.Cells.from_edges([298, 299.75, 300], order=0),
        se.Cells.log(1e5, 1e7, 1, order=0),
        B=se.Cells.linear(3, 7, 1, unit="uG", order=0),
        eta=se.Legendre(0),
        moments=12,
    )
    low, high = rb.moments(pop)
    # The switch selects the last 0.00002 uG of a two-microgauss uniform parent.
    expected_fraction = 1e-5
    np.testing.assert_allclose(
        rb.predict(pop, check=True).plan.source_columns,
        [1 - expected_fraction, expected_fraction],
        atol=2e-14,
        rtol=0,
    )
    np.testing.assert_allclose(
        _cell_masses(rb.harmonic, low),
        [0.875 - expected_fraction, 0.125],
        atol=2e-14,
        rtol=0,
    )
    # The continuum zeroth moment carries B_perp/B_ref evaluated on that sliver.
    mean_field = 5.99999e-6
    mean_sine = (
        (np.sqrt(1 - eta_max**2) + np.arcsin(eta_max) / eta_max) / 2 if eta_max else 1
    )
    np.testing.assert_allclose(
        _cell_masses(rb.continuum, high),
        [expected_fraction * mean_field * mean_sine / rb.field],
        atol=2e-14,
        rtol=0,
    )


@pytest.mark.parametrize("steps", [1, 16])
def test_critical_frequency_face_owns_atoms_and_separates_continuous_mass(steps):
    gamma, field = 1024.0, 2.0**-17
    face = A1_HZ_PER_G * field * gamma**2
    pop = _population(se.Delta(gamma), field=field)
    lower = field - steps * np.spacing(field) / 2
    upper = field + steps * np.spacing(field)
    above = pop.replace(B=se.Uniform(field, upper, unit="G"))
    below = pop.replace(B=se.Uniform(lower, field, unit="G"))
    channel = se.Channels.at([face * 0.55], unit="Hz", half_width=0.12, n_nu=8)

    def basis(edges):
        return _regime(
            pop,
            channel,
            se.Cells.linear(299, 300, 1, order=0),
            se.Cells.from_edges(edges, chart="log", order=0, unit="Hz"),
            B=se.Cells.linear(2, 16, 1, unit="uG", order=0),
            moments=20,
        )

    both = basis([face / 2, face, 2 * face])
    # The atom belongs to the first containing cell. The continuous mass on
    # either side is integrated in that cell even when its nodes round to the face.
    for source, expected in (
        (pop, [field, 0.0]),
        (below, [(lower + field) / 2, 0.0]),
        (above, [0.0, (field + upper) / 2]),
    ):
        _, moment = both.moments(source)
        np.testing.assert_allclose(
            _cell_masses(both.continuum, moment),
            np.array(expected) / both.field,
            rtol=2e-14,
            atol=0,
        )

    only_lower = basis([face / 2, face])
    prediction = only_lower.predict(above, check=True)
    np.testing.assert_array_equal(prediction.stokes, 0.0)
    assert prediction.ledger["outside_continuum"].value == pytest.approx(1.0)


def test_finite_panel_of_unbounded_law_retains_its_side_of_the_switch():
    from syncem.front._regime_measure import source_columns

    switch = 300.0
    low = np.nextafter(switch, -np.inf)
    source = _population(se.PowerLaw(2, lo=low, amplitude=1, pivot=low))
    before, after = source_columns(source.parts, switch, nodes=24)
    # The integrand is (low/gamma)**2 on [low, infinity).
    np.testing.assert_allclose(
        before, (switch - low) * low / switch, rtol=3e-15, atol=0
    )
    np.testing.assert_allclose(after, low**2 / switch, rtol=2e-12)


def test_critical_frequency_carries_all_coordinate_displacements_through_the_pole():
    from decimal import Decimal, localcontext

    from syncem.front._regime_frequency import projected_frequency

    eta = np.array([0.37, -0.85, 1.0, -1.0])
    point = {
        "gamma": np.arange(4.0) + 1024,
        "eta": eta,
        "B": np.linspace(1e-5, 2e-5, 4),
    }
    shifts = {key: np.spacing(value) / 4 for key, value in point.items()}
    shifts["eta"][2:] = [-(2.0**-55), 2.0**-55]
    _, transverse, frequency, residual = projected_frequency(point, shifts)
    with localcontext() as context:
        context.prec = 75
        for i in range(len(eta)):
            energy, cosine, field = (
                Decimal(float(value[i])) + Decimal(float(shifts[name][i]))
                for name, value in point.items()
            )
            true_field = field * (1 - cosine**2).sqrt()
            expected = Decimal(A1_HZ_PER_G) * true_field * energy**2
            actual = Decimal(float(frequency[i])) + Decimal(float(residual[i]))
            assert abs((actual - expected) / expected) < Decimal("1e-30")
            assert transverse[i] == float(true_field)


@pytest.mark.parametrize("host", [False, True])
@pytest.mark.parametrize("slope", [64.0, 128.0, 256.0])
@pytest.mark.parametrize("side", [-1, 1])
def test_continuous_parent_transports_conditional_atoms_across_switch(
    host, slope, side
):
    base = 2.0**-17

    def angle(B):
        field = float(B.value) if host else B.value
        return se.Delta(0.125 + 0.125 * (field - base) / base)

    def energy(eta):
        return se.Delta(300 - 8 * slope * (eta.value - 0.125))

    pop = _population(
        se.Conditional(energy), eta=se.Conditional(angle, units={"B": "G"})
    ).replace(
        B=se.Uniform(
            *sorted([base, np.nextafter(base, np.copysign(np.inf, side))]), unit="G"
        )
    )
    rb = _regime(
        pop,
        se.Channels.at([1.0], unit="MHz", half_width=0.12, n_nu=6),
        se.Cells.from_edges([299, 300], order=0),
        se.Cells.log(1e5, 1e7, 1, order=0),
        B=se.Cells.linear(2, 16, 1, unit="uG", order=0),
        eta=se.Legendre(0),
        moments=12,
    )
    low, high = rb.moments(pop)
    np.testing.assert_allclose(
        _cell_masses(rb.harmonic, low), float(side > 0), rtol=3e-15, atol=0
    )
    if side > 0:
        np.testing.assert_array_equal(high.values, 0.0)
    else:
        assert _cell_masses(rb.continuum, high)[0] > 0
    np.testing.assert_allclose(
        rb.predict(pop, check=True).plan.source_columns,
        [side > 0, side < 0],
        rtol=3e-15,
        atol=0,
    )
    # A point parent at the excluded endpoint instead leaves a real switch atom.
    atom = pop.replace(B=se.Delta(base, unit="G"))
    np.testing.assert_allclose(rb.predict(atom, check=True).plan.source_columns, [0, 1])


def test_channel_width_uses_the_same_transverse_geometry_at_a_rounded_pole():
    gamma = 1e9
    pop = _population(se.Delta(gamma), eta=se.Uniform(np.nextafter(1.0, 0.0), 1.0))

    def predict(width):
        basis = _regime(
            pop,
            se.Channels.at([2e11], unit="Hz", half_width=width, n_nu=8),
            se.Cells.from_edges([299, 300], order=0),
            se.Cells.log(2.5e11, 1e12, 1, order=0),
            eta=se.Legendre(0),
            moments=2,
        )
        return basis.predict(pop, check=True)

    # Even on the doubled four-node rule, 32 true spacings fit below 5e10 Hz.
    # Substituting the rounded pole for its interior node would require 9e11 Hz.
    assert predict(0.4).I[0] > 0
    with pytest.raises(se.RegimeError, match="32 harmonic spacings"):
        predict(0.0005)


@pytest.mark.parametrize("nodes", [8, 48])
def test_fractional_conditional_face_inside_two_adjacent_parent_floats(nodes):
    field = 2.0**-17
    step = np.spacing(field)

    def energy(B):
        return se.Delta(300 + ((B.value - field) / step - 0.3))

    pop = _population(se.Conditional(energy, units={"B": "G"})).replace(
        B=se.Uniform(field, field + step, unit="G")
    )
    rb = _regime(
        pop,
        se.Channels.at([1], unit="MHz", half_width=0.12, n_nu=6),
        se.Cells.from_edges([299, 300], order=0),
        se.Cells.log(1e5, 1e7, 1, order=0),
        B=se.Cells.linear(2, 16, 1, unit="uG", order=0),
        moments=nodes,
    )
    low, high = rb.moments(pop)
    np.testing.assert_allclose(
        _cell_masses(rb.harmonic, low), [0.3], atol=2e-14, rtol=0
    )
    np.testing.assert_allclose(
        _cell_masses(rb.continuum, high),
        [0.7 * (field + 0.65 * step) / rb.field],
        atol=4e-14,
        rtol=0,
    )
    np.testing.assert_allclose(
        rb.predict(pop, check=True).plan.source_columns, [0.3, 0.7], atol=2e-14, rtol=0
    )


@pytest.mark.parametrize("side", [-1, 1])
def test_switch_at_power_of_two_uses_unequal_half_ulps(side):
    from syncem.front._regime_measure import source_columns

    base, switch = 2.0**-17, 256.0

    def energy(B):
        return se.Delta(switch - 32 * (B.value - base) / base)

    edges = sorted([base, np.nextafter(base, np.copysign(np.inf, side))])
    pop = _population(se.Conditional(energy, units={"B": "G"})).replace(
        B=se.Uniform(*edges, unit="G")
    )
    np.testing.assert_allclose(
        source_columns(pop.parts, switch, nodes=12),
        [side > 0, side < 0],
        atol=0,
        rtol=3e-15,
    )


def test_deterministic_energy_touches_switch_without_a_continuum_atom():
    from syncem.front._regime_measure import source_columns

    def energy(eta):
        return se.Delta(300 - eta.value**2)

    source = _population(se.Conditional(energy), eta=se.Uniform(-1, 1))
    # An unsplit odd Gauss rule would put a finite weight at the isolated maximum.
    for nodes in (3, 5):
        np.testing.assert_allclose(
            source_columns(source.parts, 300, nodes=nodes), [1, 0], atol=0, rtol=2e-15
        )
