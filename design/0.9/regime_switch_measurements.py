"""Measure the harmonic and continuum responses near the Lorentz-factor switch.

Run from the repository root with
PYTHONPATH=. python3 design/0.9/regime_switch_measurements.py

The switch cases use isotropic pitch, a fixed field and viewing angle, and
unit-area bump channels with no depth phase. The dense reference evaluates
the real-order line integral on pitch-angle panels. It shares the SciPy
Bessel backend with the package. The continuum reference evaluates the
leading isotropic spectrum using SciPy modified Bessel functions. The
few-line reference uses all active integer orders and adaptive pitch
integration. Changes under refinement are numerical estimates. Zero
reference intensity gives NaN in the fractional columns.
"""

import argparse
import os
import platform
from time import perf_counter

import numpy as np
from numpy.polynomial.legendre import leggauss
import scipy
from scipy.integrate import quad, quad_vec
from scipy.special import jv, jvp, kve

import syncem as se
from syncem.base.window import A1_HZ_PER_G
from syncem.constants import C_CGS, E_ESU, M_E
from syncem.engine._regime_harmonic import harmonic_channel
from syncem.engine.assemble import predict_values

SWITCH = 300.0
FIELD_G = 5e-6
INCLINATION = 0.3
HALF_WIDTH = 0.2
SWITCH_X = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
RADIO_HZ = np.geomspace(0.1e9, 10e9, 41)[[0, 10, 20, 40]]
DEFAULT_RESOLUTIONS = [(2, 8, 2), (8, 8, 2), (32, 8, 2), (16, 32, 4)]


def isotropic_pitch(mu):
    return np.full_like(mu, 0.5)


def gyro_frequency(gamma, field):
    return E_ESU * field / (2 * np.pi * gamma * M_E * C_CGS)


def critical_frequency(gamma, field, eta):
    return A1_HZ_PER_G * field * np.sqrt(1 - eta**2) * gamma**2


def channel_grid(frequencies, nodes=64):
    return se.Channels.at(frequencies, unit="Hz", half_width=HALF_WIDTH, n_nu=nodes)


def line_powers(gamma, gyro, eta, pitch, orders):
    """Evaluate the helical-orbit natural I and Q line powers in cgs."""
    beta = np.sqrt(1 - gamma**-2)
    sin_theta = np.sqrt(1 - eta**2)
    transverse = beta * np.sqrt(1 - pitch**2)
    doppler = 1 - beta * pitch * eta
    argument = orders * transverse * sin_theta / doppler
    along = ((eta - beta * pitch) / sin_theta * jv(orders, argument)) ** 2
    across = (transverse * jvp(orders, argument)) ** 2
    scale = 2 * np.pi * E_ESU**2 * gyro**2 * orders**2 / (C_CGS * doppler**3)
    return np.stack([scale * (along + across), scale * (along - across)], axis=-1)


def dense_reference(channels, gamma, field, eta, cone_width, pitch_nodes):
    """Integrate the dense line approximation in pitch angle and frequency."""
    marks = np.array([0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6])
    edges = np.r_[marks[marks < cone_width], cone_width]
    edges = np.r_[-edges[:0:-1], edges]
    left = edges[:-1, None]
    half = np.diff(edges)[:, None] / 2
    nodes, weights = leggauss(pitch_nodes)
    angle = np.arccos(eta) + (left + half * (1 + nodes)).ravel() / gamma
    pitch = np.cos(angle)
    pitch_weights = (half * weights).ravel() * np.sin(angle) / (2 * gamma)
    frequency = np.asarray(channels.nodes)
    channel_weights = np.asarray(channels.response_weights())
    gyro = gyro_frequency(gamma, field)
    spacing = gyro / (1 - np.sqrt(1 - gamma**-2) * pitch * eta)
    weights = pitch_weights / spacing
    output = []
    for nu, w_nu in zip(frequency, channel_weights, strict=True):
        orders = nu[:, None] / spacing
        values = line_powers(gamma, gyro, eta, pitch, orders)
        output.append(np.einsum("f,p,fps->s", w_nu, weights, values))
    return np.asarray(output)


def continuum_reference(channels, gamma, field, eta):
    """Evaluate channel averages of the leading isotropic F and G spectra."""
    transverse = field * np.sqrt(1 - eta**2)
    x = np.asarray(channels.nodes) / critical_frequency(gamma, field, eta)
    scale = np.sqrt(3) * E_ESU**3 * transverse / (4 * np.pi * M_E * C_CGS**2)
    f = np.array(
        [
            quad(
                lambda u: np.exp(-u) * kve(5 / 3, t + u),
                0,
                np.inf,
                epsabs=0,
                epsrel=1e-11,
            )[0]
            for t in x.ravel()
        ]
    ).reshape(x.shape)
    g = kve(2 / 3, x)
    weights = np.asarray(channels.response_weights())
    common = scale * weights * x * np.exp(-x)
    return np.column_stack([np.sum(common * f, axis=1), -np.sum(common * g, axis=1)])


def report(label, delta, reference, centres):
    """Report signed channel differences scaled by local or peak reference I."""
    intensity = reference[:, 0]
    # A zero reference can occur when the radio-tail spectrum underflows.
    local = np.divide(
        delta,
        intensity[:, None],
        where=intensity[:, None] > 0,
        out=np.full_like(delta, np.nan),
    )
    print(f"\n{label}. Delta is in cgs spectral units.")
    print(
        "  centre_Hz      reference_I     delta_I         delta_Q      delta_I/I       delta_Q/I"
    )
    for nu, ref, raw, frac in zip(centres, intensity, delta, local, strict=True):
        print(
            f"  {nu:11.4e}   {ref:13.6e}   {raw[0]:13.6e}   {raw[1]:13.6e}"
            f"   {frac[0]:13.6e}   {frac[1]:13.6e}"
        )
    if intensity.max() > 0:
        peak = np.max(np.abs(delta), axis=0) / intensity.max()
        print(f"  max |delta| / peak I is {peak[0]:.6e} (I), {peak[1]:.6e} (Q)")


def compare_kernels(centres, eta):
    coarse = channel_grid(centres, 128)
    fine = channel_grid(centres, 256)
    lower = dense_reference(coarse, SWITCH, FIELD_G, eta, 40, 32)
    upper = dense_reference(fine, SWITCH, FIELD_G, eta, 60, 64)
    continuum = continuum_reference(fine, SWITCH, FIELD_G, eta)
    report("Dense reference refinement", upper - lower, upper, centres)
    report(
        "Dense reference minus leading continuum at the same gamma",
        upper - continuum,
        continuum,
        centres,
    )
    for angular, frequency in [(8, 64), (16, 128), (32, 256), (64, 256)]:
        values = np.asarray(
            harmonic_channel(
                fine,
                SWITCH,
                FIELD_G,
                eta,
                isotropic_pitch,
                angular_nodes=angular,
                frequency_nodes=frequency,
            )
        ).real
        report(
            f"Harmonic kernel minus dense reference with {angular} pitch nodes "
            f"and {frequency} frequency nodes",
            values - upper,
            upper,
            centres,
        )


def compare_monochromatic():
    """Evaluate the published gamma, angle and x combinations."""
    points = (
        (100.0, 0.0, 1.0),
        (SWITCH, 0.0, 1.0),
        (SWITCH, 0.5, 1.0),
        (SWITCH, 0.9, 0.01),
        (SWITCH, 0.0, 35.0),
        (SWITCH, 0.0, 266.0),
    )
    print("\nMonochromatic dense-line limit at B=5 microgauss and isotropic pitch.")
    print(
        "  gamma      eta      x        I_h/I_c-1      Q_h/Q_c-1      max rule change / I_h"
    )
    for gamma, eta, x in points:
        nu = x * critical_frequency(gamma, FIELD_G, eta)
        monochromatic = se.Channels.at([nu], unit="Hz")
        coarse = dense_reference(monochromatic, gamma, FIELD_G, eta, 40, 32)[0]
        fine = dense_reference(monochromatic, gamma, FIELD_G, eta, 60, 64)[0]
        continuum = continuum_reference(monochromatic, gamma, FIELD_G, eta)[0]
        residual = (fine - continuum) / continuum
        change = np.max(np.abs(fine - coarse)) / fine[0]
        print(
            f"  {gamma:5.0f}   {eta:6.2f}   {x:7.2f}   "
            f"{residual[0]:13.6e}  {residual[1]:13.6e}   {change:13.6e}"
        )


def point_population(gamma, eta):
    return se.Population(
        gamma=se.Delta(gamma),
        B=se.Delta(FIELD_G, unit="G"),
        eta=se.Delta(eta),
        mu=se.Isotropic(),
        phi=se.AzimuthFactor(1.0),
        depth=se.NoRotation(),
    )


def compare_cells(centres, eta, resolutions):
    channels = channel_grid(centres)
    reference_channels = channel_grid(centres, 256)
    critical = critical_frequency(SWITCH, FIELD_G, eta)
    gamma_below = SWITCH * (1 - 1e-6)
    gamma_above = SWITCH * (1 + 1e-6)
    harmonic_ref = dense_reference(
        reference_channels, gamma_below, FIELD_G, eta, 60, 64
    )
    continuum_ref = continuum_reference(reference_channels, gamma_above, FIELD_G, eta)
    report(
        "Dense reference below switch minus continuum reference above",
        harmonic_ref - continuum_ref,
        continuum_ref,
        centres,
    )
    # Hold the upper lattice fixed while changing the lower representation.
    # The separate upper residual shows its contribution in the faint tail.
    high = se.Lattice(
        nu_c=se.Cells.log(0.9 * critical, 1.5 * critical, 16, order=8, unit="Hz"),
        phi=se.AzimuthHarmonic(),
    )
    below = point_population(gamma_below, eta)
    above = point_population(gamma_above, eta)
    for cells, nodes, order in resolutions:
        start = perf_counter()
        low = se.Lattice(
            gamma=se.Cells.log(100.0, SWITCH, cells, order=order),
            phi=se.AzimuthHarmonic(),
            order=order,
        )
        basis = se.RegimeBasis.build(
            point_population(250.0, eta),
            channels,
            harmonic=low,
            continuum=high,
            gamma_switch=SWITCH,
            angular_nodes=nodes,
            frequency_nodes=64,
            field_nodes=1,
            inclination_nodes=1,
            check=True,
        )
        prediction = basis.predict(below)
        lower = np.asarray(prediction.stokes)[:, :2]
        upper = np.asarray(basis.predict(above).stokes)[:, :2]
        one_minute_load = os.getloadavg()[0]
        print(
            f"\n{cells} energy cells of order {order} and {nodes} base pitch nodes. "
            f"check=True. Elapsed {perf_counter() - start:.1f} s "
            f"at one-minute load {one_minute_load:.2f}."
        )
        column_estimate = prediction.ledger["harmonic_column_refinement"]
        print(
            "  Max unweighted column change under build-rule refinement "
            f"is {np.max(np.abs(column_estimate.value)):.6e}."
        )
        report("Below minus above from the cell basis", lower - upper, upper, centres)
        report(
            "Below from the cell basis minus dense direct response",
            lower - harmonic_ref,
            harmonic_ref,
            centres,
        )
        report(
            "Above from the cell basis minus direct continuum",
            upper - continuum_ref,
            continuum_ref,
            centres,
        )


def few_line_reference(gamma, field, eta, centre, fraction):
    """Sum every integer order admitted for any pitch cosine in the bump."""
    gyro = gyro_frequency(gamma, field)
    beta = np.sqrt(1 - gamma**-2)
    width = centre * fraction
    low, high = centre - width, centre + width
    max_order = int(high * (1 + beta * abs(eta)) / gyro)
    orders = np.arange(1, max_order + 1)
    area = width * quad(lambda t: np.exp(1 - 1 / (1 - t**2)), -1, 1, epsabs=1e-13)[0]

    def integrand(mu):
        frequency = orders * gyro / (1 - beta * mu * eta)
        t = (frequency - centre) / width
        inside = np.abs(t) < 1
        active = orders[inside]
        power = line_powers(gamma, gyro, eta, mu, active)
        weights = 0.5 * np.exp(1 - 1 / (1 - t[inside] ** 2)) / area
        return weights @ power

    edges = [
        (1 - order * gyro / nu) / (beta * eta) for order in orders for nu in (low, high)
    ]
    reference, estimate = quad_vec(
        integrand,
        -1,
        1,
        epsabs=0,
        epsrel=1e-12,
        points=[t for t in edges if -1 < t < 1],
    )
    return reference, estimate, max_order


def compare_few_lines():
    gamma, field, eta, fraction = 10.0, 10.0, 0.45, 0.3
    gyro = gyro_frequency(gamma, field)
    centre = gyro / (1 - np.sqrt(1 - gamma**-2) * 0.5 * eta)
    channels = se.Channels.at([centre], unit="Hz", half_width=fraction)
    reference, estimate, max_order = few_line_reference(
        gamma, field, eta, centre, fraction
    )
    print(
        f"\nFew-line case at gamma={gamma:g}, B={field:g} G, eta={eta:g}, "
        f"half-width={fraction:g}, positive orders through {max_order}."
    )
    print(
        f"  Adaptive pitch integration estimate is {estimate / reference[0]:.6e} of I."
    )
    for nodes in (8, 16, 32, 64, 128):
        values = np.asarray(
            harmonic_channel(
                channels, gamma, field, eta, isotropic_pitch, angular_nodes=nodes
            )
        ).real
        report(
            f"Harmonic kernel with {nodes} pitch nodes minus integer sum",
            values - reference,
            reference[None, :],
            [centre],
        )


def compare_paper():
    """Measure changes with the paper's field, inclination and screen laws."""
    channels = se.paper.channels()[:1]
    for name in ("E1", "E1M"):
        start = perf_counter()
        source = se.paper.population(name)
        default = se.paper.regime_basis(name, channels, check=False)
        checked = se.paper.regime_basis(name, channels, check=True)
        narrower = se.paper.regime_basis(
            name,
            channels,
            harmonic_cells=32,
            angular_nodes=64,
            frequency_nodes=128,
            field_nodes=8,
            inclination_nodes=8,
            check=False,
        )
        spectra = [
            basis.predict(source, check=True) for basis in (default, checked, narrower)
        ]
        harmonic = [
            np.asarray(predict_values(basis.harmonic, spectrum.moments[0]))[0]
            for basis, spectrum in zip(
                (default, checked, narrower), spectra, strict=True
            )
        ]
        total = float(spectra[-1].I[0])
        estimate = spectra[1].ledger["harmonic_column_refinement"]
        print(
            f"\n{name} at 0.1 GHz. One-minute load {os.getloadavg()[0]:.2f}. "
            f"Elapsed {perf_counter() - start:.1f} s."
        )
        print(
            f"  Default and refined harmonic I / total I: {[row[0] / total for row in harmonic]}"
        )
        print(
            f"  Build-rule Stokes change / total I: {np.max(np.abs(harmonic[1] - harmonic[0])) / total:.6e}"
        )
        print(
            f"  Combined cell-width and interpolation-rule Stokes change / total I: "
            f"{np.max(np.abs(harmonic[2] - harmonic[1])) / total:.6e}"
        )
        print(f"  Max unweighted column change: {np.max(np.abs(estimate.value)):.6e}")


def resolution(text):
    try:
        count, nodes, *explicit_order = map(int, text.split(","))
        (order,) = explicit_order or [2]
        if count < 1 or nodes < 1 or order < 0:
            raise ValueError
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected CELL_COUNT,PITCH_NODES[,ORDER] with positive counts and nonnegative order"
        ) from error
    return count, nodes, order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section",
        choices=["all", "monochromatic", "kernel", "cells", "few-line", "paper"],
        default="all",
        help="Use 'paper' separately for the full E1 and E1M population checks.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=INCLINATION,
        help="Viewing cosine for the finite-channel switch cases.",
    )
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Use the two original 41-channel grids in the kernel comparison.",
    )
    parser.add_argument(
        "--resolution",
        action="append",
        type=resolution,
        metavar="CELL_COUNT,PITCH_NODES[,ORDER]",
        help="Cell build configuration. Order defaults to two. Repeat to compare several.",
    )
    args = parser.parse_args()
    if not -0.95 < args.eta < 0.95:
        parser.error(
            "--eta must lie between -0.95 and 0.95 for the pitch-cone reference"
        )

    print(
        f"Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}"
    )
    print(f"SyncEM source {se.__file__}")
    if args.section in {"all", "kernel", "cells"}:
        critical = critical_frequency(SWITCH, FIELD_G, args.eta)
        switch_centres = SWITCH_X * critical
        radio_centres = RADIO_HZ
        if args.full_grid:
            switch_centres = np.geomspace(0.1, 3.0, 41) * critical
            radio_centres = np.geomspace(0.1e9, 10e9, 41)
        print(
            f"Switch at gamma={SWITCH:g}, B={FIELD_G:g} G, eta={args.eta:g}, "
            f"and nu_c={critical:.8e} Hz."
        )
        print(
            f"Isotropic pitch, zero depth, unit-area bump with half-width {HALF_WIDTH:g}. "
            "Paper channel centres are evaluated at fixed B and eta."
        )
    if args.section in {"all", "monochromatic"}:
        compare_monochromatic()
    if args.section in {"all", "kernel"}:
        print("\nKernel comparison near the critical frequency")
        compare_kernels(switch_centres, args.eta)
        print("\nKernel comparison at the paper radio channel centres")
        compare_kernels(radio_centres, args.eta)
    if args.section in {"all", "cells"}:
        centres = np.r_[SWITCH_X * critical, RADIO_HZ]
        compare_cells(centres, args.eta, args.resolution or DEFAULT_RESOLUTIONS)
    if args.section in {"all", "few-line"}:
        compare_few_lines()
    if args.section == "paper":
        compare_paper()


if __name__ == "__main__":
    main()
