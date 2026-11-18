"""Rule R5 (Spec Sec. 5.2): the azimuth factor c = <exp(2i phi) | parents> follows its parents.

Routes under test.
(i)   constant c, applied to the spin-2 block at assembly (``assemble_constant``).
(ii)  parents on Cells: per-cell Cauchy product of the bare spin-2 columns with the
      Taylor coefficients of c in the cell (``factor_taylor`` and ``cauchy_product``).
(iii) an averaged parent: c is a weight inside the averaged spin-2 kernel at the parents'
      quadrature nodes (``FactorKernel`` inside ``AveragedKernel``).
(iv)  a cosine parent on Legendre(L): c multiplies the kernel inside the projection
      (``FactorKernel`` with a Legendre axis).

``FactorKernel`` puts c inside the kernel. Every averaged, Legendre or celled parent is then
handled by the core (quadrature, projection, jet), so it is the reference implementation of
"averaging the factor into the kernel". The wrong routes apply a factorised <c> to <K_P>.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import core as C
from core.lattice import ProductTable, from_u


class FactorKernel:
    """Point kernel with P multiplied by c(point). I is unchanged.

    ``c(point) -> complex scalar`` is written in jnp, so the jet differentiates it in celled
    parents and quadrature evaluates it at averaged nodes. ``frozen`` lists parents whose
    derivatives are cut by stop_gradient (a wrong route used as a negative control).
    """

    def __init__(self, kernel, c, frozen=()):
        self.kernel, self.c, self.frozen = kernel, c, tuple(frozen)
        self.n_ch = kernel.n_ch

    def __call__(self, p):
        I, P = self.kernel(p)
        q = {k: (jax.lax.stop_gradient(v) if k in self.frozen else v) for k, v in p.items()}
        return I, P * self.c(q)


def factor_taylor(table: ProductTable, c, fixed=None) -> np.ndarray:
    """(n_rows,) complex: (1/k!) d^k c / du^k at each cell's reference point, on its lower set."""
    fixed = dict(fixed or {})
    names, d = table.variables, len(table.variables)
    out = np.zeros(table.n_rows, complex)
    o = table.offsets
    for cell in range(table.n_cells):
        charts, ref, hw = table.chart[cell], table.reference[cell], table.half_width[cell]

        def f(z, charts=charts, ref=ref, hw=hw):
            pt = {v: from_u(charts[i], z[i], ref[i], hw[i]) for i, v in enumerate(names)}
            val = c({**fixed, **pt})
            return jnp.stack([jnp.real(val), jnp.imag(val)])

        t = np.asarray(C.taylor_coefficients(f, d, table.exponents(cell)))
        out[o[cell]:o[cell + 1]] = t[:, 0] + 1j * t[:, 1]
    return out


def cauchy_product(cols_P: np.ndarray, table: ProductTable, ctay: np.ndarray) -> np.ndarray:
    """Route (ii): P'[b, n] = sum over r + k = n of P[b, r] c[b, k] on each cell's lower set."""
    out = np.zeros_like(cols_P, dtype=complex)
    o = table.offsets
    for cell in range(table.n_cells):
        exps = table.exponents(cell)
        pos = {e: i for i, e in enumerate(exps)}
        for i, n in enumerate(exps):
            acc = 0.0
            for j, r in enumerate(exps):
                k = tuple(a - b for a, b in zip(n, r))
                if min(k) < 0:
                    continue
                acc = acc + cols_P[o[cell] + j] * ctay[o[cell] + pos[k]]  # k is in the lower set
            out[o[cell] + i] = acc
    return out


def zeroth_order_factor(cols_P: np.ndarray, table: ProductTable, ctay: np.ndarray) -> np.ndarray:
    """Wrong route: c at each cell's reference point only (the k = 0 term of the product)."""
    c0 = ctay[table.offsets[:-1]][table.row_cell]
    return cols_P * c0.reshape((-1,) + (1,) * (cols_P.ndim - 1))


def predict_with_P(cols: dict, cols_P: np.ndarray, table, moments, w_cells) -> np.ndarray:
    """predict with the spin-2 columns replaced by ``cols_P`` and no further factor."""
    return C.predict({"I": cols["I"], "P": cols_P}, table, moments, w_cells, azimuth=1.0)


def assemble_constant(cols: dict, table, moments, w_cells, cbar) -> np.ndarray:
    """Route (i), and the factorised wrong route when ``cbar`` is the population mean of c."""
    return C.predict(cols, table, moments, w_cells, azimuth=cbar)


def rel_err_P(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per channel |dP| / |P_ref| with P = Q + iU, shape (n_ch,)."""
    dP = (pred[:, 1] - ref[:, 1]) + 1j * (pred[:, 2] - ref[:, 2])
    return np.abs(dP) / np.abs(ref[:, 1] + 1j * ref[:, 2])


def rel_err_I(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per channel max over I, Q, U of |dX| / I_ref, shape (n_ch,)."""
    return np.max(np.abs(pred - ref) / np.abs(ref[:, :1]), axis=1)


def verdict(e: float) -> str:
    """Boundary-validation thresholds of the user's rule."""
    return "OK" if e < 1e-3 else ("WARN" if e < 1e-1 else "BAD")
