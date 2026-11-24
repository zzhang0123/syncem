"""Write a versioned numeric snapshot of a fit result."""

import dataclasses

from ..engine._basis_archive import write_archive
from ..engine._basis_io_codec import pack


def _moments(moment):
    return {
        "values": moment.values,
        "key": moment.key,
        "masked": moment.masked,
        "origin": moment.origin,
        "component": moment.component,
    }


def save_fit(fit, path):
    """Save arrays and records without storing Python functions or initial data."""
    moments = fit.moments if isinstance(fit.moments, tuple) else (fit.moments,)
    params = getattr(fit, "params", None)
    retained = getattr(fit, "retained", None)
    withheld = getattr(fit, "withheld", None)
    body = {
        "kind": fit.kind,
        "estimator": fit.estimator,
        "rank": fit.rank,
        "chi2": fit.chi2,
        "dof": fit.dof,
        "p_value": fit.p_value,
        "effective_parameters": getattr(fit, "effective_parameters", None),
        "weights": getattr(fit, "weights", None),
        "parameter_covariance": getattr(fit, "parameter_covariance", None),
        "moments": tuple(map(_moments, moments)),
        "params": (
            None
            if params is None
            else {
                name: (
                    dataclasses.asdict(param)
                    if dataclasses.is_dataclass(param)
                    else param
                )
                for name, param in params.items()
            }
        ),
        "retained": (
            None
            if retained is None
            else {
                name: getattr(retained, name)
                for name in ("beta", "sigma", "directions", "rank", "key")
            }
        ),
        "withheld": None if withheld is None else dict(withheld._asdict()),
        "ledger": fit.ledger,
        "plan": fit.plan.to_dict(),
        "defaults": fit.defaults,
    }
    arrays = {}
    write_archive(path, pack(body, arrays), arrays, kind="SyncEM Fit")
