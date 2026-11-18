"""Prediction-level checks of R3a: node prediction, exact-joint prediction, direct integration,
and the averaged-child route (the plan's remedy), for one gamma parent and a resolved B child.

    pred_node  = predict(columns, joint, L a_parent)         the R3a route
    pred_joint = predict(columns, joint, a_joint_nested)     exact joint moments, same columns
    ref        = direct(kernel, joint measure, joint table)   no expansion, within the cells
    pred_avg   = predict(columns of the B-averaged kernel on the gamma table, a_parent)

conditional_truncation = pred_node - pred_joint, representation = pred_joint - ref.
"""
from __future__ import annotations

import time

import numpy as np

import core as C

from . import node as R
from . import setups as T


def run(kernel_avg, cond, case: T.Case, pop="pl2.6", *, n_pop=24, n_child=8, azimuth=1.0,
        averaged_rule=None, with_bound=True, direct_batch=64):
    t0 = time.time()
    S = max(e[-1] for c in range(case.joint.n_cells) for e in case.joint.exponents(c))
    N = max(max(o) for o in case.parent.order)
    h, ks = R.taylor_h(cond, case.parent, ("gamma",), case.child_chart, case.child, S, N, deriv="analytic")
    L = R.node_matrix(case.parent, case.joint, ("gamma",), h, ks, case.w_parent, case.w_joint)
    m = T.parent_measure(case.parent, pop, n=n_pop)
    ap = C.pair(case.parent, m, case.w_parent)["M0"]
    a_true = R.joint_moments_nested(cond, m, case.parent, case.joint, case.child_chart, case.child,
                                    case.w_joint)
    t_mom = time.time() - t0
    t0 = time.time()
    cols = C.build_columns(kernel_avg, case.joint)
    t_cols = time.time() - t0
    pred_node = C.predict(cols, case.joint, {"M0": L @ ap}, case.w_joint, azimuth=azimuth)
    pred_joint = C.predict(cols, case.joint, {"M0": a_true}, case.w_joint, azimuth=azimuth)
    t0 = time.time()
    jm = R.joint_measure(cond, m, "B", case.child_chart, case.child, n=n_child)
    ref = C.direct(kernel_avg, jm, case.joint, azimuth=azimuth, batch_size=direct_batch)
    t_dir = time.time() - t0
    out = dict(
        cond_trunc=_err(pred_node, pred_joint, ref),
        representation=_err(pred_joint, ref, ref),
        node_vs_direct=_err(pred_node, ref, ref),
        n_direct_nodes=int(jm.weights.size),
        seconds=dict(moments=t_mom, columns=t_cols, direct=t_dir),
    )
    if with_bound:
        bnd = R.prediction_bound(cond, case.parent, case.joint, h, cols, case.w_parent, ap, case.child_chart,
                                 case.child)
        dI = np.abs(pred_node[:, 0] - pred_joint[:, 0])
        dP = np.abs((pred_node[:, 1] - pred_joint[:, 1]) + 1j * (pred_node[:, 2] - pred_joint[:, 2]))
        b_I, b_P = bnd[:, 0], bnd[:, 1] * abs(azimuth)
        out["bound_rel_I"] = float(np.max(b_I / np.abs(ref[:, 0])))
        floor = 1e-13 * np.abs(ref[:, 0])  # roundoff of the two predictions
        out["bound_holds"] = bool(np.all(dI <= b_I * (1 + 1e-9) + floor) and np.all(dP <= b_P * (1 + 1e-9) + floor))
        out["bound_over_actual"] = float(np.max(b_I / np.abs(ref[:, 0])) / max(out["cond_trunc"], 1e-300))
    if averaged_rule is not None:
        avgB = R.averaged_child_kernel(kernel_avg.kernel, "B", averaged_rule,
                                       extra=list(kernel_avg.averaged), fixed=kernel_avg.fixed)
        colsB = C.build_columns(avgB, case.parent)
        pred_avg = C.predict(colsB, case.parent, {"M0": ap}, case.w_parent, azimuth=azimuth)
        out["avg_vs_node"] = _err(pred_avg, pred_node, ref)
        out["avg_vs_direct"] = _err(pred_avg, ref, ref)
    return out


def _err(a, b, ref):
    """max over channels and I, Q, U of |a - b| / I_ref."""
    return float(np.max(np.abs(a - b) / np.abs(ref[:, :1])))
