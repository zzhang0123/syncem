"""Figure 1: one orthographic construction of the three angle planes.

Regenerate only this figure with ``python -m validation.angle_geometry``.
All vectors and angle arcs are constructed in three dimensions before projection;
the camera changes their appearance, never their definitions or handedness.
"""

from pathlib import Path
import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from .figure_style import BLUE, GREY, INK, VERMILION, WIDTH, configure, save

ROOT = Path(__file__).resolve().parents[1]


def great_arc(first, second, radius, samples=101):
    """Minor circular arc in the plane of two nonparallel unit directions."""
    angle = np.arccos(np.clip(np.dot(first, second), -1.0, 1.0))
    tangent = second - np.dot(first, second) * first
    tangent /= np.linalg.norm(tangent)
    phase = np.linspace(0.0, angle, samples)
    return radius * (np.cos(phase)[:, None] * first + np.sin(phase)[:, None] * tangent)


def geometry(ax=None):
    """Draw a schematic, checking its 3D construction against the conventions."""
    configure()
    theta, phi, alpha = np.deg2rad([42.0, 35.0, 28.0])
    x, y, n = np.eye(3)
    epar = np.cos(phi) * x + np.sin(phi) * y
    eperp = np.cross(n, epar)
    field = np.sin(theta) * epar + np.cos(theta) * n
    toward_n = (n - np.cos(theta) * field) / np.sin(theta)
    # A generic gyrophase: v deliberately lies outside the (n, B) plane.
    tangent = -np.sqrt(1.0 - 0.65**2) * toward_n - 0.65 * eperp
    velocity = np.cos(alpha) * field + np.sin(alpha) * tangent
    field_tip = 2.9 * field
    sky_tip = field_tip - np.dot(field_tip, n) * n

    # Orthographic camera above the sky plane. Positive z points towards the
    # physical observer; this schematic camera is a separate viewpoint.
    azimuth, elevation = np.deg2rad([20.0, 30.0])
    right = np.array([-np.cos(azimuth), np.sin(azimuth), 0.0])
    up = np.array(
        [
            -np.sin(elevation) * np.sin(azimuth),
            -np.sin(elevation) * np.cos(azimuth),
            np.cos(elevation),
        ]
    )
    camera = np.cross(right, up)
    projection = np.stack([right, up])
    assert camera[2] > 0.0, "View the sky plane from the observer side"

    checks = {
        "pitch_angle": float(np.dot(velocity, field) - np.cos(alpha)),
        "field_inclination": float(np.dot(field, n) - np.cos(theta)),
        "sky_azimuth": float(np.arctan2(epar[1], epar[0]) - phi),
        "sky_normal_component": float(np.dot(sky_tip, n)),
        "velocity_norm": float(np.linalg.norm(velocity) - 1.0),
    }
    np.testing.assert_allclose(list(checks.values()), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.cross(x, y), n, atol=1e-14)
    np.testing.assert_allclose(np.cross(epar, eperp), n, atol=1e-14)
    np.testing.assert_allclose(sky_tip / np.linalg.norm(sky_tip), epar, atol=1e-14)
    np.testing.assert_allclose(projection @ projection.T, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(projection @ camera, 0.0, atol=1e-14)
    triple_product = float(np.dot(np.cross(n, field), velocity))
    assert (
        abs(triple_product) > 0.1
    ), "The three spatial directions must not be coplanar"

    standalone = ax is None
    figure_width = 0.72 * WIDTH
    if standalone:
        fig, ax = plt.subplots(figsize=(figure_width, 3.10))
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    else:
        fig = ax.figure
    ax.set(xlim=(-2.65, 1.85), ylim=(-1.65, 2.23), aspect="equal")
    ax.axis("off")

    def project(points):
        return np.asarray(points) @ projection.T

    def line(points, color=GREY, lw=0.65, ls="-", zorder=2):
        projected = project(points)
        ax.plot(
            projected[:, 0],
            projected[:, 1],
            color=color,
            lw=lw,
            ls=ls,
            solid_capstyle="round",
            zorder=zorder,
        )

    def arrow(end, color=INK, lw=1.4, zorder=5):
        ax.annotate(
            "",
            xy=project(end),
            xytext=(0.0, 0.0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": lw,
                "shrinkA": 0,
                "shrinkB": 0,
                "mutation_scale": 10,
            },
            zorder=zorder,
        )

    def label(point, text, offset=(0, 0), color=INK, **kwargs):
        return ax.annotate(
            text,
            xy=project(point),
            xytext=offset,
            textcoords="offset points",
            color=color,
            ha="center",
            va="center",
            zorder=10,
            **kwargs,
        )

    plane = np.array(
        [[-0.90, -0.60, 0.0], [2.50, -0.60, 0.0], [2.50, 2.30, 0.0], [-0.90, 2.30, 0.0]]
    )
    ax.add_patch(
        Polygon(
            project(plane), facecolor="#F4F5F6", edgecolor="#CDD1D4", lw=0.65, zorder=0
        )
    )
    # Short dashed negative axes establish the plane without a plotting grid.
    line([-0.90 * x, np.zeros(3)], color="#B8BDC1", ls=(0, (3, 3)), zorder=1)
    line([-0.60 * y, np.zeros(3)], color="#B8BDC1", ls=(0, (3, 3)), zorder=1)
    arrow(2.35 * x, color=GREY, lw=0.8, zorder=2)
    arrow(2.20 * y, color=GREY, lw=0.8, zorder=2)
    label(2.35 * x, r"$\hat{\boldsymbol{x}}$", offset=(-8, -1), color=GREY)
    label(2.20 * y, r"$\hat{\boldsymbol{y}}$", offset=(6, -7), color=GREY)
    label([0.60, 1.95, 0.0], "sky plane", offset=(0, -4), color=GREY)

    # This drop is parallel to n: it is an orthogonal physical projection even
    # though right angles generally do not look square in the schematic view.
    line([field_tip, sky_tip], color="#8B969D", lw=0.8, ls=(0, (4, 3)))
    corner = 0.15
    line(
        [
            sky_tip - corner * epar,
            sky_tip - corner * epar + corner * n,
            sky_tip + corner * n,
        ],
        color="#8B969D",
        lw=0.65,
    )
    arrow(sky_tip, color=BLUE, lw=1.35)
    arrow(1.40 * eperp, color=INK, lw=1.05)
    label(1.40 * eperp, r"$\hat{\boldsymbol{e}}_\perp$", offset=(5, 10))
    label(
        sky_tip,
        r"$\boldsymbol{B}_{\rm sky}\parallel\hat{\boldsymbol{e}}_\parallel$",
        offset=(-16, -13),
        color=BLUE,
    )

    arrow(2.10 * n, color=INK, lw=1.3)
    label(2.10 * n, r"$\boldsymbol{n}=\hat{\boldsymbol{z}}$", offset=(0, 12))
    label(2.10 * n, "towards\nobserver", offset=(30, -2))
    arrow(field_tip, color=BLUE, lw=1.8)
    label(field_tip, r"$\boldsymbol{B}$", offset=(-9, 6), color=BLUE)
    arrow(1.95 * velocity, color=VERMILION, lw=1.5)
    label(1.95 * velocity, r"$\boldsymbol{v}$", offset=(-10, 3), color=VERMILION)

    # Separate radii make the three genuinely different angle planes legible.
    for first, second, radius, symbol, offset in [
        (n, field, 0.77, r"$\theta$", (-3, 5)),
        (field, velocity, 1.34, r"$\alpha$", (-7, 8)),
        (x, epar, 1.13, r"$\phi$", (-4, -7)),
    ]:
        points = great_arc(first, second, radius)
        line(points, color=INK, lw=0.85, zorder=6)
        if symbol == r"$\phi$":
            ax.annotate(
                "",
                xy=project(points[-1]),
                xytext=project(points[-7]),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": INK,
                    "lw": 0.85,
                    "shrinkA": 0,
                    "shrinkB": 0,
                    "mutation_scale": 7,
                },
                zorder=7,
            )
        label(points[len(points) // 2], symbol, offset=offset)
        # Arc endpoints lie on their declared rays; no screen-space angle is
        # substituted for the physical 3D angle.
        np.testing.assert_allclose(
            points[[0, -1]], radius * np.array([first, second]), atol=1e-14
        )
    ax.plot(0, 0, "o", ms=2.4, color=INK, zorder=9)
    if standalone:
        save(fig, "angle_geometry", width=figure_width)
    return {
        "view": "Single orthographic construction, generic noncoplanar velocity",
        "illustrative_angles_deg": {"theta": 42.0, "phi": 35.0, "alpha": 28.0},
        "schematic_camera_deg": {"azimuth": 20.0, "elevation": 30.0},
        "geometric_residuals": checks,
        "noncoplanar_scalar_triple_product": triple_product,
        "sky_handedness": "e_parallel cross e_perp = n; x cross y = n",
        "scope": "Direction schematic; arrow lengths do not compare physical magnitudes",
    }


def main():
    result = geometry()
    result["source_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("validation/angle_geometry.py", "validation/figure_style.py")
    }
    result["output_sha256"] = {
        f"figures/angle_geometry.{extension}": hashlib.sha256(
            (ROOT / "figures" / f"angle_geometry.{extension}").read_bytes()
        ).hexdigest()
        for extension in ("pdf", "png")
    }
    (ROOT / "validation/angle_geometry_results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
