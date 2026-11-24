"""Signed direct comparisons of the assumed statistics of an existing basis."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from syncem.base.channels import Channels
from syncem.base.user import kernel
from syncem.base.variables import Variable
from syncem.distributions import Uniform
from syncem.front._predictions import _direct_values, _factor_record, build_one
from syncem.front._setup import placed
from syncem.front._specified import comparison_plan, replace_terms, specified_terms
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.population import Population
from syncem.population.declaration import ordered
from syncem.records.ledger import ErrorTerm, Ledger


@kernel(
    variables=(
        Variable("x"),
        Variable("B", kind="positive", unit="G"),
        Variable("tilt"),
    ),
    stokes=("I", "P"),
)
def template(nu, x, B, tilt):
    intensity = (1 + x * x) * (1e6 * B + tilt) * (nu / 1e9)
    return {"I": intensity, "P": (0.1 + 0.2j) * intensity}


@pytest.fixture(scope="module")
def built():
    channels = Channels.at([1.0, 2.0], unit="GHz")
    lattice = Lattice(x=Cells.linear(-0.4, 1.0, 3, order=1))
    source = Population(x=Uniform(-2.0, 1.1), B="3 uG", tilt=0.4)
    return build_one(
        source, channels, lattice, kernel=template, check=False, cache=False
    )


def _plan(basis, target):
    return comparison_plan(basis, target, place=placed, record=_factor_record)


def test_two_replacements_close_the_signed_direct_error(built):
    target = 1.7 * Population(x=Uniform(-1.0, 2.0), B="7 uG", tilt=-0.3)
    plan = _plan(built, target)
    assert plan.changed == ("B", "tilt")
    assert plan.evaluation_count() == 4
    assert plan.evaluation_count(projected=True) == 5
    before = plan.baseline_parts[0]
    assert before.column_factor == target.column_factor
    assert before.factors["B"].unit == "G"
    covered = _direct_values(built, before, "cells")[0]
    outside = _direct_values(built, before, "outside")[0]
    after = _direct_values(built, target, "support")[0]
    prediction = built.predict(target, check=False, allow_mismatch=True).stokes
    measured = plan.measure(
        (covered + outside,), lambda member: _direct_values(built, member, "support")[0]
    )
    assert all(term.kind == "measured" and term.signed for term in measured)
    assert measured[0].value[0, 0] < 0
    assert measured[1].value[0, 0] > 0
    represented = prediction - covered
    total = represented - outside + sum(term.value for term in measured)
    np.testing.assert_allclose(total, prediction - after, atol=1e-13)
    term_names = [term.name for term in measured]
    assert term_names == ["specified:B", "specified:tilt"]


def test_dependency_order_avoids_a_cycle_in_the_intermediate_population(built):
    def tilt(B):
        return 1e6 * B.value

    def B(tilt):
        return (4 + tilt, "uG")

    source, _ = placed(Population(x=0.4, B="3 uG", tilt=tilt), built.averaged.kernel)
    basis = replace(built, forward_population=source)
    plan = _plan(basis, Population(x=0.6, B=B, tilt=0.1))
    assert plan.changed == ("tilt", "B")
    for _, candidate in plan.parts[0].replacements:
        ordered(candidate.factors)


def test_numerical_check_disabled_or_traced_keeps_assumptions_unbounded(built):
    original = specified_terms(built)
    ledger = Ledger((ErrorTerm.unbounded(name="physical_kernel"), *original))
    plan = _plan(built, Population(x=0.3, B="4 uG", tilt=0.4))
    result = replace_terms(ledger, plan.unbounded("cost limit"))
    assert result["specified:B"].kind == result["specified:tilt"].kind == "unbounded"
    assert result["specified:B"].note == "cost limit"
    assert result["specified:tilt"].note != "cost limit"
    assert len(result.terms) == 3

    @jax.jit
    def traced(x):
        declaration = Population(x=x, B="4 uG", tilt=0.4)
        terms = _plan(built, declaration).unbounded()
        return jnp.asarray(all(t.note == "unmeasured (traced)" for t in terms))

    assert bool(traced(0.5))
