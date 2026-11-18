"""Rule R3a: the fixed linear node from free parent moments to joint (parent, child) moments.

Spec Sec. 5.2 R3a with the r3-01 correction. For parent product cell b and child cell c

    a[(b, c), (e, s)] = (w_b / w_{b,c}) sum_k h[b, c, s, k] a_parent[b, e + k]

over the multi-indices k of the conditioning parents with e + k in the parent block's lower set.
h[b, c, s, k] = (1/k!) d^k g[c, s] / du^k at the reference point of b (u in the parent chart),
taken by nested jax.jacfwd (core.jet.taylor_coefficients).

Also here: nested-quadrature joint moments of a specified parent (the reference), a joint
measure for direct integration, the per-unit-mass bound of the dropped terms, and the
averaged-child route that the plan names as the remedy.
"""
from __future__ import annotations

import math


import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss

import core as C
from core.lattice import ProductTable, from_u, to_u


# ------------------------------------------------------------------ Taylor coefficients h
def k_set(n_cond: int, kmax: int) -> tuple[tuple[int, ...], ...]:
    """Multi-indices k over the conditioning parents with |k| <= kmax."""
    return C.lower_set((kmax,) * n_cond, kmax)


def taylor_h(cond, parent: ProductTable, cond_parents, child_chart, child_cells, S: int, kmax: int,
             method="exact", fixed_point=None, deriv="jacfwd", **kw) -> tuple[np.ndarray, tuple]:
    """h (n_parent_cells, n_child_cells, S+1, n_k) and the k list.

    ``cond_parents``: the parent variables the conditional depends on (a subset of
    parent.variables). Parent variables outside it are held at their reference point.
    ``deriv``: 'jacfwd' (nested forward mode of core.jet, the Spec's route, compile time grows
    about 3x per order), 'jet' (jax.experimental.jet Taylor mode, one parent), or 'analytic'
    (cond.taylor_u in NumPy, one gamma parent). ``method`` picks the child-cell quadrature of g.
    """
    ks = k_set(len(cond_parents), kmax)
    if deriv == "analytic":
        if tuple(cond_parents) != ("gamma",) or parent.variables != ("gamma",):
            raise ValueError("analytic h is written for one gamma parent")
        h = np.stack([cond.taylor_u(parent.reference[b, 0], parent.half_width[b, 0], child_chart,
                                    child_cells, S, kmax) for b in range(parent.n_cells)])
        return h, ks
    if deriv == "jet":
        return _taylor_h_jet(cond, parent, cond_parents, child_chart, child_cells, S, kmax, method,
                             **kw), ks
    pos = [parent.variables.index(v) for v in cond_parents]
    ch = parent.chart[0]
    if any(parent.chart[b] != ch for b in range(parent.n_cells)):
        raise ValueError("one chart per parent variable is assumed")

    def one(args):
        ref, hw = args

        def f(z):
            pt = {v: ref[i] for i, v in enumerate(parent.variables)}
            for j, i in enumerate(pos):
                pt[parent.variables[i]] = from_u(ch[i], z[j], ref[i], hw[i])
            if fixed_point:
                pt.update(fixed_point)
            return cond.g(pt, child_chart, child_cells, S, method=method, **kw)

        return C.taylor_coefficients(f, len(pos), ks)  # (n_k, n_c, S+1)

    run = jax.jit(lambda r, w: jax.lax.map(one, (r, w)))
    out = np.asarray(run(jnp.asarray(parent.reference), jnp.asarray(parent.half_width)))
    return np.moveaxis(out, 1, -1), ks


def _taylor_h_jet(cond, parent, cond_parents, child_chart, child_cells, S, kmax, method, **kw):
    from jax.experimental import jet

    if len(cond_parents) != 1 or len(parent.variables) != 1:
        raise ValueError("jet h is written for one parent variable")
    ch = parent.chart[0][0]
    v = parent.variables[0]
    fact = [math.factorial(k) for k in range(1, kmax + 1)]

    def coeffs(ref, hw):
        def f(u):
            return cond.g({v: from_u(ch, u, ref, hw)}, child_chart, child_cells, S, method=method, **kw)

        if kmax == 0:
            return f(0.0)[..., None]
        y0, ys = jet.jet(f, (0.0,), ((1.0,) + (0.0,) * (kmax - 1),))
        return jnp.stack([y0] + [ys[k] / fact[k] for k in range(kmax)], axis=-1)

    run = jax.jit(jax.vmap(coeffs))
    return np.asarray(run(jnp.asarray(parent.reference[:, 0]), jnp.asarray(parent.half_width[:, 0])))


# ------------------------------------------------------------------ the node
def _parent_cell_lookup(parent: ProductTable):
    return {tuple(parent.index[b]): b for b in range(parent.n_cells)}


def node_matrix(parent: ProductTable, joint: ProductTable, cond_parents, h, ks, w_parent, w_joint,
                truncate=True) -> np.ndarray:
    """L (joint.n_rows, parent.n_rows) with a_joint = L a_parent.

    joint.variables = parent.variables + (child,), joint cells from ProductTable.tensor.
    ``truncate=False`` raises when a needed parent moment is outside the parent lower set
    (used to check that a polynomial g needs no dropped term).
    """
    if joint.variables[:-1] != parent.variables:
        raise ValueError("joint variables must be the parent variables followed by the child")
    lookup = _parent_cell_lookup(parent)
    pos = [parent.variables.index(v) for v in cond_parents]
    L = np.zeros((joint.n_rows, parent.n_rows))
    jo, po = joint.offsets, parent.offsets
    for cj in range(joint.n_cells):
        b = lookup[tuple(joint.index[cj, :-1])]
        c = int(joint.index[cj, -1])
        pexp = {e: m for m, e in enumerate(parent.exponents(b))}
        ratio = w_parent[b] / w_joint[cj]
        for m, ex in enumerate(joint.exponents(cj)):
            e, s = ex[:-1], ex[-1]
            for ik, k in enumerate(ks):
                shifted = list(e)
                for j, i in enumerate(pos):
                    shifted[i] += k[j]
                shifted = tuple(shifted)
                if shifted in pexp:
                    L[jo[cj] + m, po[b] + pexp[shifted]] += ratio * h[b, c, s, ik]
    return L


def dropped_terms(parent: ProductTable, joint: ProductTable, cond_parents, h, ks) -> int:
    """Number of (row, k) pairs with h != 0 whose shifted parent moment is outside the lower set."""
    lookup = _parent_cell_lookup(parent)
    pos = [parent.variables.index(v) for v in cond_parents]
    n = 0
    for cj in range(joint.n_cells):
        b = lookup[tuple(joint.index[cj, :-1])]
        c = int(joint.index[cj, -1])
        pexp = set(parent.exponents(b))
        for ex in joint.exponents(cj):
            e, s = ex[:-1], ex[-1]
            for ik, k in enumerate(ks):
                shifted = list(e)
                for j, i in enumerate(pos):
                    shifted[i] += k[j]
                if tuple(shifted) not in pexp and abs(h[b, c, s, ik]) > 1e-13:
                    n += 1
    return n


# ------------------------------------------------------------------ specified parent: references
def joint_moments_nested(cond, parent_measure: C.Measure, parent: ProductTable, joint: ProductTable,
                         child_chart, child_cells, w_joint, fixed_point=None) -> np.ndarray:
    """a_joint by nested quadrature: parent nodes times the closed-form child-cell moments g."""
    S = max(e[-1] for c in range(joint.n_cells) for e in joint.exponents(c))
    cell = parent.locate(parent_measure.nodes)
    keep = cell >= 0
    nodes = {v: np.asarray(x)[keep] for v, x in parent_measure.nodes.items()}
    wts, cell = parent_measure.weights[keep], cell[keep]
    pts = {v: jnp.asarray(nodes[v]) for v in parent.variables}

    def one(pt):
        if fixed_point:
            pt = {**pt, **fixed_point}
        return cond.g(pt, child_chart, child_cells, S, method="exact")

    G = np.asarray(jax.vmap(one)(pts))  # (M, n_c, S+1)
    lookup = _parent_cell_lookup(parent)
    a = np.zeros(joint.n_rows)
    jo = joint.offsets
    for cj in range(joint.n_cells):
        b = lookup[tuple(joint.index[cj, :-1])]
        c = int(joint.index[cj, -1])
        sel = cell == b
        U = [np.asarray(to_u(parent.chart[b][i], nodes[v][sel], parent.reference[b, i],
                             parent.half_width[b, i])) for i, v in enumerate(parent.variables)]
        for m, ex in enumerate(joint.exponents(cj)):
            mono = wts[sel] * np.prod([U[i] ** ex[i] for i in range(len(U))], axis=0)
            a[jo[cj] + m] = mono @ G[sel, c, ex[-1]] / w_joint[cj]
    return a


def joint_measure(cond, parent_measure: C.Measure, child_var, child_chart, child_cells, n=8,
                  fixed_point=None) -> C.Measure:
    """Nodes (parent nodes x child nodes) with weights w_i p(x | parents_i), child panels split at
    the child-cell edges and at the breaks that resolve the conditional density."""
    lo = np.maximum(np.where(np.isfinite(child_cells.lo), child_cells.lo, 0.0), 0.0)
    hi = np.asarray(child_cells.hi, float)
    t, w = leggauss(n)
    out_nodes = {v: [] for v in list(parent_measure.nodes) + [child_var]}
    out_w = []
    M = parent_measure.weights.size
    for i in range(M):
        pt = {v: float(parent_measure.nodes[v][i]) for v in parent_measure.nodes}
        if fixed_point:
            pt.update(fixed_point)
        if hasattr(cond, "breaks_lnB"):
            br = np.exp(cond.breaks_lnB(pt))
        else:
            br = cond.breaks_B(pt)
        edges = np.unique(np.concatenate([lo, hi[np.isfinite(hi)]]))
        top = hi.max() if np.all(np.isfinite(hi)) else br.max()
        bot = lo.min() if lo.min() > 0 else max(br.min(), 1e-300)
        allb = np.unique(np.concatenate([edges, br]))
        allb = allb[(allb >= bot) & (allb <= top)]
        allb = np.unique(np.concatenate([[bot, top], allb]))
        if child_chart == "log" or hasattr(cond, "breaks_lnB"):
            s = np.log(allb)
            xs = np.concatenate([np.exp(0.5 * (c - a) * t + 0.5 * (c + a)) for a, c in zip(s[:-1], s[1:])])
            js = np.concatenate([0.5 * (c - a) * w for a, c in zip(s[:-1], s[1:])]) * xs
        else:
            xs = np.concatenate([0.5 * (c - a) * t + 0.5 * (c + a) for a, c in zip(allb[:-1], allb[1:])])
            js = np.concatenate([0.5 * (c - a) * w for a, c in zip(allb[:-1], allb[1:])])
        dens = np.asarray(cond.density(jnp.asarray(xs), pt))
        for v in parent_measure.nodes:
            out_nodes[v].append(np.full(xs.size, parent_measure.nodes[v][i]))
        out_nodes[child_var].append(xs)
        out_w.append(parent_measure.weights[i] * dens * js)
    return C.Measure({v: np.concatenate(x) for v, x in out_nodes.items()}, np.concatenate(out_w))


# ------------------------------------------------------------------ bound for a free parent
def truncation_sup(cond, parent: ProductTable, joint: ProductTable, cond_parents, h, ks, child_chart,
                   child_cells, n_grid=2001) -> np.ndarray:
    """R[row] = max over |u| <= 1 of |u^e (T g_s - g_s)(u)| for one conditioning parent.

    For a positive parent measure, |a_node - a_true| <= (w_b / w_{b,c}) a_parent[b, 0] R[row]
    on every joint row, since |u| <= 1 in a bounded parent cell. The truncation acts on a
    known function, so the bound needs only the parent mass of the cell.
    """
    if len(cond_parents) != 1 or len(parent.variables) != 1:
        raise ValueError("sup bound written for one parent variable")
    S = h.shape[2] - 1
    u = np.linspace(-1.0, 1.0, n_grid)
    lookup = _parent_cell_lookup(parent)
    R = np.zeros(joint.n_rows)
    jo = joint.offsets
    cache = {}
    for cj in range(joint.n_cells):
        b = lookup[tuple(joint.index[cj, :-1])]
        c = int(joint.index[cj, -1])
        if b not in cache:
            x = from_u(parent.chart[b][0], jnp.asarray(u), parent.reference[b, 0], parent.half_width[b, 0])
            cache[b] = np.asarray(jax.vmap(lambda g: cond.g({parent.variables[0]: g}, child_chart,
                                                             child_cells, S))(x))  # (n_grid, n_c, S+1)
        G = cache[b]
        Nb = max(e[0] for e in parent.exponents(b))
        for m, (r, s) in enumerate(joint.exponents(cj)):
            T = sum(h[b, c, s, k] * u**k for k in range(0, Nb - r + 1))
            R[jo[cj] + m] = np.max(np.abs(u**r * (T - G[:, c, s])))
    return R


def prediction_bound(cond, parent: ProductTable, joint: ProductTable, h, columns, w_parent, a_parent,
                     child_chart, child_cells, n_grid=1001) -> np.ndarray:
    """(n_ch, 2) bound on |predict(node) - predict(exact joint moments)| for I and |P|.

    sum over parent cells b of w_b a_parent[b, 0] max_{|u| <= 1} |sum_{c, (r, s)} col[(b, c), (r, s)]
    u^r (T_{N_b - r} g_s - g_s)(u)|. Valid for every positive parent measure with these cell masses.
    """
    S = h.shape[2] - 1
    u = np.linspace(-1.0, 1.0, n_grid)
    lookup = _parent_cell_lookup(parent)
    n_ch = columns["I"].shape[1]
    acc = {b: (np.zeros((n_grid, n_ch)), np.zeros((n_grid, n_ch), complex)) for b in range(parent.n_cells)}
    G = {}
    jo = joint.offsets
    for cj in range(joint.n_cells):
        b = lookup[tuple(joint.index[cj, :-1])]
        c = int(joint.index[cj, -1])
        if b not in G:
            x = from_u(parent.chart[b][0], jnp.asarray(u), parent.reference[b, 0], parent.half_width[b, 0])
            G[b] = np.asarray(jax.vmap(lambda g: cond.g({parent.variables[0]: g}, child_chart, child_cells,
                                                        S))(x))
        Nb = max(e[0] for e in parent.exponents(b))
        for m, (r, s) in enumerate(joint.exponents(cj)):
            T = sum(h[b, c, s, k] * u**k for k in range(0, Nb - r + 1))
            d = u**r * (T - G[b][:, c, s])
            acc[b][0][:] += np.outer(d, columns["I"][jo[cj] + m])
            acc[b][1][:] += np.outer(d, columns["P"][jo[cj] + m])
    out = np.zeros((n_ch, 2))
    for b, (aI, aP) in acc.items():
        mass = w_parent[b] * a_parent[parent.offsets[b]]
        out[:, 0] += mass * np.abs(aI).max(axis=0)
        out[:, 1] += mass * np.abs(aP).max(axis=0)
    return out


# ------------------------------------------------------------------ the averaged-child route
def averaged_child_kernel(kernel, child_var, rule, extra=(), **kw):
    """The plan's remedy: the child averaged with its conditional inside the kernel average."""
    return C.AveragedKernel(kernel, [(child_var, rule)] + list(extra), **kw)
