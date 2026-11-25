"""Pullbacks of critical-frequency cells in the continuum physical measure."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

import syncem as se
from syncem.base.window import A1_HZ_PER_G
from tests.v090.test_regime_review_regressions import _population, _regime


@pytest.mark.parametrize("axis", ["B", "gamma", "eta"])
@pytest.mark.parametrize("conditional", [False, True])
def test_face_whose_physical_preimage_is_between_adjacent_floats(axis, conditional):
    from decimal import Decimal, localcontext

    field, gamma, eta = 5e-6, 1100.0, 0.8 if axis == "eta" else 0.0
    base = {"B": field, "gamma": gamma, "eta": eta}[axis]
    upper = float(np.nextafter(base, np.inf))
    with localcontext() as context:
        context.prec = 80
        a, b = Decimal(base), Decimal(upper)
        scale = Decimal(A1_HZ_PER_G)
        magnetic, energy, angle = map(Decimal, (field, gamma, eta))

        def frequency(x):
            B = x if axis == "B" else magnetic
            g = x if axis == "gamma" else energy
            q = x if axis == "eta" else angle
            return scale * B * g**2 * (1 - q**2).sqrt()

        low, high = sorted((frequency(a), frequency(b)))
        face = float(low)
        if Decimal(face) <= low:
            face = float(np.nextafter(face, np.inf))
        assert low < Decimal(face) < high
        target = Decimal(face) / scale
        if axis == "B":
            root = target / energy**2
        elif axis == "gamma":
            root = (target / magnetic).sqrt()
        else:
            root = (1 - (target / (energy**2 * magnetic)) ** 2).sqrt()
        fraction = float((b - root if axis == "eta" else root - a) / (b - a))

    source = _population(se.Delta(gamma), field=field, eta=eta).replace(
        **{axis: se.Uniform(base, upper, unit="G" if axis == "B" else None)}
    )
    if conditional:
        # A constant conditional is the same joint measure as the independent atom.
        if axis == "B":
            source = source.replace(
                gamma=se.Conditional(lambda B: se.Delta(gamma + 0 * B.value))
            )
        elif axis == "gamma":
            source = source.replace(
                B=se.Conditional(
                    lambda gamma: se.Delta(field + 0 * gamma.value, unit="G")
                )
            )
        else:
            source = source.replace(
                gamma=se.Conditional(lambda eta: se.Delta(gamma + 0 * eta.value))
            )
    regime = _regime(
        source,
        se.Channels.at([0.55 * face], unit="Hz", half_width=0.12, n_nu=6),
        se.Cells.from_edges([299, 300], order=0),
        se.Cells.from_edges([face / 2, face, face * 2], unit="Hz", order=0),
        B=se.Cells.linear(2, 16, 1, unit="uG", order=0),
        eta=se.Legendre(0),
        moments=8,
    )
    moment = regime.moments(source)[1]
    lattice = regime.continuum.lattice
    mass = np.array([moment.values[lattice.index.cell_rows(i).start] for i in (0, 1)])
    mass *= lattice.weights
    # B_perp changes by at most a few machine epsilons over the interval.
    np.testing.assert_allclose(
        mass / mass.sum(), [fraction, 1 - fraction], atol=3e-14, rtol=0
    )


_G = 3e4
_B = 5e-6
_NU = A1_HZ_PER_G * _B * _G**2
_CHANNEL = se.Channels.at([0.55 * _NU], unit="Hz", half_width=0.12, n_nu=8)


def _basis(pop, relative, channel=_CHANNEL, **physical):
    return _regime(
        pop,
        channel,
        se.Cells.linear(299, 300, 1, order=0),
        se.Cells.log(*(_NU * np.array(relative)), 1, order=2),
        mu=se.Legendre(1),
        eta=se.Legendre(1),
        moments=12,
        **physical,
    )


def _mass(basis, pop):
    return basis.continuum.lattice.column(basis.moments(pop)[1])


def _observe(basis, pop):
    prediction = basis.predict(pop, check=True)
    return _mass(basis, pop), float(prediction.I[0])


@pytest.mark.parametrize("reverse", [False, True])
def test_angular_face_has_positive_and_negative_preimages(reverse):
    low, high = -0.8, 0.8
    qa, qb = 0.373, 0.375
    axis = "mu" if reverse else "eta"

    def density(gamma):
        return se.Uniform(low + gamma.value * 0, high)

    lo, hi = np.sqrt(1 - np.array([qb, qa]) ** 2)

    if reverse:
        # Make inclination a child so that pitch drives the diagonal integral.
        def inclination(mu):
            return se.Uniform(-1.0, 1.0 + mu.value * 0)

        pop = _population(
            se.Delta(_G), eta=se.Conditional(inclination), mu=se.Uniform(low, high)
        )
    else:
        pop = _population(se.Delta(_G), eta=se.Conditional(density))
    rb = _basis(pop, (lo, hi))
    expected = 2 * quad(lambda q: np.sqrt(1.0 - q * q), qa, qb)[0] / (high - low)
    mass, intensity = _observe(rb, pop)
    np.testing.assert_allclose(mass, expected, rtol=8e-13)
    edges = [low, -qb, -qa, qa, qb, high]
    pieces = [
        (b - a) / (high - low) * pop.replace(**{axis: se.Uniform(a, b)})
        for a, b in zip(edges, edges[1:])
    ]
    independent = pieces[0]
    for piece in pieces[1:]:
        independent = independent + piece
    _, paired = _observe(rb, independent)
    np.testing.assert_allclose(intensity, paired, atol=0, rtol=3e-12)


@pytest.mark.parametrize("conditional", [False, True])
def test_field_face_for_atomic_energy_and_sightline(conditional):
    low, high, fa, fb, q = 4.0e-6, 6.0e-6, 4.819e-6, 4.822e-6, 0.35

    def magnetic(eta):
        return se.Uniform(low + eta.value * 0, high, unit="G")

    law = se.Conditional(magnetic) if conditional else se.Uniform(low, high, unit="G")
    pop = _population(se.Delta(_G), eta=q).replace(B=law)
    sine = np.sqrt(1 - q * q)
    rb = _basis(
        pop,
        np.array([fa, fb]) * sine / _B,
        channel=se.Channels.at([0.55 * _NU * sine], unit="Hz", half_width=0.12, n_nu=8),
        B=se.Cells.linear("3.9 uG", "6.1 uG", 1, order=0),
    )
    expected = (fb - fa) * (fa + fb) / 2 * sine / (_B * (high - low))
    mass, intensity = _observe(rb, pop)
    np.testing.assert_allclose(mass, expected, rtol=3e-12)
    np.testing.assert_allclose(intensity, 1.06612791e-31, rtol=8e-9, atol=0)


def test_faces_on_a_deterministic_field_trajectory_and_folded_pitch_panels():
    def field(eta):
        return se.Delta(_B * (1 + 0.25 * eta.value), unit="G")

    def pitch(B):
        shift = B.value / _B - 1
        return se.Uniform(0.239 + shift, 0.241 + shift)

    def frequency(q):
        return (1 + 0.25 * q) * np.sqrt(1 - q * q)

    qa, qb = 0.31993, 0.32004
    # The frequency has a turning point. Its other preimage has zero pitch density.
    lo, hi = frequency(qb), frequency(qa)
    assert brentq(lambda q: frequency(q) - lo, -0.3, 0.22) < 0.239 / 0.75
    pop = _population(se.Delta(_G), eta=se.Isotropic()).replace(
        B=se.Conditional(field), mu=se.Conditional(pitch, units={"B": "G"})
    )
    rb = _basis(pop, (lo, hi), B=se.Cells.linear(2, 8, 1, unit="uG"))
    expected = quad(frequency, qa, qb)[0] / 0.002
    np.testing.assert_allclose(_mass(rb, pop), expected, rtol=2e-10)


@pytest.mark.parametrize("conditional", [False, True])
def test_energy_face_with_fixed_field_and_view(conditional):
    q = 0.35
    low, high, ga, gb = 2.9e4, 3.1e4, 3.0447e4, 3.0449e4

    def energy(B):
        return se.Uniform(low + B.value * 0, high)

    law = (
        se.Conditional(energy, units={"B": "G"})
        if conditional
        else se.Uniform(low, high)
    )
    pop = _population(law, eta=q)
    sine = np.sqrt(1 - q * q)
    rb = _basis(
        pop,
        (np.array([ga, gb]) / _G) ** 2 * sine,
        B=se.Cells.linear(4.8, 5.2, 1, unit="uG"),
    )
    np.testing.assert_allclose(
        _mass(rb, pop), sine * (gb - ga) / (high - low), rtol=1.0e-11
    )


def test_conditional_atoms_change_which_continuous_coordinate_has_the_faces():
    # At each value of eta one physical variable has a continuous law and the
    # other is a point. A common quadrature order must cover both branches.
    def energy(eta):
        return se.Uniform(2.9e4, 3.1e4) if eta.value < 0 else se.Delta(_G)

    def magnetic(eta):
        return (
            se.Delta(_B, unit="G") if eta.value < 0 else se.Uniform(4.8, 5.2, unit="uG")
        )

    eta = 0.3
    # A joint sample of the external angle leaves just these two branches.
    pop = se.Population(
        se.Samples(eta=[-eta, eta], weights=[0.5, 0.5]),
        gamma=se.Conditional(energy),
        B=se.Conditional(magnetic),
        mu=se.Isotropic(),
        phi=se.AzimuthFactor(0.2),
        depth=se.NoRotation(),
    )
    sine = np.sqrt(1 - eta**2)
    lo, hi = 0.9923, 0.99241
    rb = _basis(pop, (lo * sine, hi * sine), B=se.Cells.linear(4.7, 5.3, 1, unit="uG"))
    gamma_mass = sine * _G * (np.sqrt(hi) - np.sqrt(lo)) / (3.1e4 - 2.9e4)
    field_mass = sine * _B * (hi - lo) * (hi + lo) / (2 * (5.2e-6 - 4.8e-6))
    np.testing.assert_allclose(
        _mass(rb, pop), 0.5 * (gamma_mass + field_mass), rtol=2e-11
    )
