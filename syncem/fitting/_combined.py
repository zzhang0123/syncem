"""Column assembly for independent mechanisms in the same observation space."""

import equinox as eqx
import jax.numpy as jnp

from ..engine.assemble import select_stokes
from ..engine.basis_sum import BasisSum
from ..values.errors import DataError
from .free_fit import _design, _fit_basis


class IndependentBases(eqx.Module):
    """Disjoint moment coordinates that share one observational likelihood."""

    parts: tuple
    components: tuple[str, ...] = eqx.field(static=True)


def parts_of(basis):
    return basis.parts if isinstance(basis, (BasisSum, IndependentBases)) else (basis,)


def combined_columns(basis, channels, axes):
    outputs = []
    for part in parts_of(basis):
        fitted = _fit_basis(part, channels)
        missing = set(axes) - set(fitted.record.stokes)
        if missing:
            raise DataError(f"the component basis lacks Stokes axes {sorted(missing)}")
        outputs.append(select_stokes(_design(fitted), fitted.record.stokes, axes))
    return outputs[0] if len(outputs) == 1 else jnp.concatenate(outputs, axis=-1)
