"""Step 5b-1 of design/0.8/SPEC.md: the canonical encoder and the channel row keys (decision 153).

``syncem.base._fingerprint.encode`` writes a value as canonical bytes and ``digest`` hashes them with
sha256. ``Channels.row_keys()`` gives one digest per channel. Every assertion here is an exact identity of
bytes or strings, so no tolerance enters.
"""

import dataclasses
import functools
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from syncem.base._fingerprint import NotCacheable, digest, encode
from syncem.base.channels import Channels
from syncem.base.continuum import ContinuumKernel

FINGERPRINT = (
    Path(__file__).resolve().parents[2] / "syncem" / "base" / "_fingerprint.py"
)


@dataclasses.dataclass(frozen=True)
class _Pair:
    a: object
    b: object


@dataclasses.dataclass(frozen=True)
class _Other:
    a: object
    b: object


# ------------------------------------------------------------------ encoder
def test_type_tags_make_equal_python_values_distinct():
    values = (1, 1.0, True, "1", (1,), [1], b"1", 1 + 0j, None)
    encodings = [encode(v) for v in values]
    assert len(set(encodings)) == len(values)


def test_negative_zero_encodes_as_zero():
    assert encode(-0.0) == encode(0.0)
    assert encode(np.array([-0.0, 1.0])) == encode(np.array([0.0, 1.0]))
    assert encode(complex(-0.0, -0.0)) == encode(0j)


def test_nan_payloads_encode_alike():
    other_nan = np.frombuffer(np.uint64(0x7FF8000000000001).tobytes(), "<f8")[0]
    assert encode(float(other_nan)) == encode(float("nan"))
    assert encode(np.array([other_nan])) == encode(np.array([np.nan]))


def test_dict_and_set_order_does_not_matter():
    assert encode({"a": 1, "b": (2, 3)}) == encode({"b": (2, 3), "a": 1})
    assert encode({3, "x", 1.5}) == encode({1.5, 3, "x"})
    assert encode(frozenset({"p", "q"})) == encode(frozenset({"q", "p"}))
    assert encode({"a": 1}) != encode({"a": 2})


def test_containers_with_equal_items_differ_by_type_and_nesting():
    assert encode((1, 2)) != encode([1, 2])
    assert encode(((1, 2), 3)) != encode((1, (2, 3)))
    assert encode(("ab", "c")) != encode(("a", "bc"))


def test_arrays_encode_dtype_and_shape():
    a = np.arange(6, dtype=np.float64)
    assert encode(a) != encode(a.astype(np.float32))
    assert encode(a) != encode(a.astype(np.int64))
    assert encode(a.reshape(2, 3)) != encode(a.reshape(3, 2))
    assert encode(a.reshape(2, 3)) != encode(a)
    assert encode(a.reshape(2, 3).T) == encode(np.ascontiguousarray(a.reshape(2, 3).T))
    assert encode(a.astype(">f8")) == encode(a)
    assert encode(jnp.asarray(a)) == encode(a)


def test_numpy_scalars_encode_as_python_scalars():
    assert encode(np.float64(0.5)) == encode(0.5)
    assert encode(np.int32(3)) == encode(3)
    assert encode(np.bool_(True)) == encode(True)


def test_dataclasses_encode_by_qualname_and_fields():
    assert encode(_Pair(1, 2)) == encode(_Pair(1, 2))
    assert encode(_Pair(1, 2)) != encode(_Pair(2, 1))
    assert encode(_Pair(1, 2)) != encode(_Other(1, 2))
    assert encode(_Pair(1, 2)) != encode((1, 2))


def test_functions_raise_not_cacheable():
    for fn in (lambda x: x, np.sin, functools.partial(max, 1), _Pair, encode):
        with pytest.raises(NotCacheable, match="not cacheable"):
            encode(fn)
    with pytest.raises(NotCacheable):
        encode(_Pair(1, lambda x: x))


def test_traced_arrays_raise_not_cacheable():
    seen = []

    def body(x):
        with pytest.raises(NotCacheable, match="traced"):
            encode(x)
        seen.append(True)
        return x

    jax.jit(body)(jnp.ones(2))
    assert seen == [True]


def test_object_arrays_raise_not_cacheable():
    with pytest.raises(NotCacheable):
        encode(np.array([1, "a"], dtype=object))


def test_digest_is_sha256_hex():
    key = digest(("x", 1.0))
    assert len(key) == 64 and int(key, 16) >= 0
    assert digest(("x", 1.0)) == key != digest(("x", 2.0))


def test_digests_are_equal_across_hash_seeds():
    """One subprocess per seed loads the encoder from its file and prints a digest of hash-ordered values."""
    script = (
        "import importlib.util, sys\n"
        "spec = importlib.util.spec_from_file_location('fp', sys.argv[1])\n"
        "fp = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(fp)\n"
        "value = ({'s%d' % i for i in range(40)}, frozenset('abcdefgh'),"
        " {'k%d' % i: i for i in range(40)})\n"
        "print(fp.digest(value))\n"
    )
    outputs = set()
    for seed in ("1", "2", "3", "4"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        done = subprocess.run(
            [sys.executable, "-c", script, str(FINGERPRINT)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.add(done.stdout.strip())
    assert len(outputs) == 1 and len(outputs.pop()) == 64


def test_encoder_module_loads_without_the_package():
    spec = importlib.util.spec_from_file_location("fp_alone", FINGERPRINT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.digest((1, "a")) == digest((1, "a"))


# ----------------------------------------------------------------- row keys
@pytest.fixture(scope="module")
def channels():
    return Channels.log(1, 10, 5, unit="GHz", half_width=0.2, n_nu=16)


def test_row_keys_are_one_digest_per_channel(channels):
    keys = channels.row_keys()
    assert isinstance(keys, tuple) and len(keys) == channels.n_ch
    assert len(set(keys)) == channels.n_ch
    assert channels.row_keys() == keys


def _changed(channels, field, j, k, factor):
    values = getattr(channels, field)
    return dataclasses.replace(
        channels, **{field: values.at[j, k].set(values[j, k] * factor)}
    )


@pytest.mark.parametrize("field", ["nodes", "weights"])
def test_row_keys_change_only_for_the_edited_channel(channels, field):
    keys = channels.row_keys()
    edited = _changed(channels, field, 2, 7, 1 + 1e-12).row_keys()
    assert edited[2] != keys[2]
    assert edited[:2] + edited[3:] == keys[:2] + keys[3:]


def test_row_keys_carry_the_labels(channels):
    keys = channels.row_keys()
    relabelled = dataclasses.replace(channels, smoothness=0).row_keys()
    assert all(a != b for a, b in zip(keys, relabelled))


def test_row_keys_ignore_the_unit_label(channels):
    assert dataclasses.replace(channels, unit="MHz").row_keys() == channels.row_keys()
    in_mhz = Channels.log(1000, 10000, 5, unit="MHz", half_width=0.2, n_nu=16)
    np.testing.assert_array_equal(np.asarray(in_mhz.nodes), np.asarray(channels.nodes))
    assert in_mhz.row_keys() == channels.row_keys()


@pytest.mark.parametrize("rows", [[3], [4, 0, 2], [2, 2, 1], slice(1, 4)])
def test_selected_row_keys_equal_the_selected_keys(channels, rows):
    keys = channels.row_keys()
    picked = np.atleast_1d(np.arange(channels.n_ch)[rows])
    assert channels[rows].row_keys() == tuple(keys[i] for i in picked)


def test_monochromatic_row_keys():
    mono = Channels.at([1, 2, 3], unit="GHz")
    keys = mono.row_keys()
    assert len(set(keys)) == 3
    assert mono[[2, 0]].row_keys() == (keys[2], keys[0])


def test_row_keys_raise_not_cacheable_under_trace(channels):
    def body(nodes):
        with pytest.raises(NotCacheable):
            dataclasses.replace(channels, nodes=nodes).row_keys()
        return nodes

    jax.jit(body)(channels.nodes)


# ------------------------------------------------------------------- guards
def test_continuum_kernel_names_its_guards():
    assert ContinuumKernel.guards == ("x_min",)
    names = [name for name, _ in ContinuumKernel().describe()]
    assert set(ContinuumKernel.guards) <= set(names)
