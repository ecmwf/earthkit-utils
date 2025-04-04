#!/usr/bin/env python3

# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import pytest

from earthkit.utils.array import _CUPY
from earthkit.utils.array import _JAX
from earthkit.utils.array import _NUMPY
from earthkit.utils.array import _TORCH
from earthkit.utils.array import get_backend
from earthkit.utils.testing import NO_CUPY
from earthkit.utils.testing import NO_JAX
from earthkit.utils.testing import NO_TORCH


def test_utils_array_backend_numpy():
    b = get_backend("numpy")
    assert b.name == "numpy"
    assert b is _NUMPY

    import numpy as np

    v = np.ones(10)
    v_lst = [1.0] * 10

    assert id(b.to_numpy(v)) == id(v)
    assert id(b.from_numpy(v)) == id(v)
    assert id(b.from_other(v)) == id(v)

    assert np.allclose(b.from_other(v_lst, dtype=np.float64), v)
    assert get_backend(v) is b
    assert get_backend(np) is b

    assert np.isclose(b.namespace.mean(v), 1.0)
    assert b.namespace.isclose(b.namespace.mean(v_lst), 1.0)
    assert b.compat_namespace.isclose(b.compat_namespace.mean(v_lst), 1.0)
    assert b.raw_namespace.isclose(b.raw_namespace.mean(v_lst), 1.0)

    if not NO_TORCH:
        import torch

        v_pt = torch.ones(10, dtype=torch.float64)
        pt_b = get_backend("pytorch")
        r = pt_b.from_other(v)
        assert torch.is_tensor(r)
        assert torch.allclose(r, v_pt)


@pytest.mark.skipif(NO_TORCH, reason="No pytorch installed")
def test_utils_array_backend_torch():
    b = get_backend("pytorch")
    assert b.name == "torch"
    assert b is _TORCH

    b = get_backend("torch")
    assert b.name == "torch"
    assert b is _TORCH

    import numpy as np
    import torch

    v = torch.ones(10, dtype=torch.float64)
    v_np = np.ones(10, dtype=np.float64)
    v_lst = [1.0] * 10

    assert torch.allclose(b.from_numpy(v_np), v)
    assert torch.allclose(b.from_other(v_lst, dtype=torch.float64), v)
    assert get_backend(v) is b

    x = b.asarray(v_lst, dtype=b.float64)
    ref = b.asarray(1.0, dtype=b.float64)
    assert torch.isclose(b.namespace.mean(x), ref)
    assert b.namespace.isclose(b.namespace.mean(x), ref)
    assert b.compat_namespace.isclose(b.compat_namespace.mean(x), ref)
    assert b.raw_namespace.isclose(b.raw_namespace.mean(x), ref)

    r = b.to_numpy(v)
    assert isinstance(r, np.ndarray)
    assert np.allclose(r, v_np)

    assert np.isclose(b.namespace.mean(v), 1.0)


@pytest.mark.skipif(NO_CUPY, reason="No cupy installed")
def test_utils_array_backend_cupy():
    b = get_backend("cupy")
    assert b.name == "cupy"
    assert b is _CUPY

    import cupy as cp
    import numpy as np

    v = cp.ones(10, dtype=cp.float64)
    v_np = np.ones(10, dtype=np.float64)
    v_lst = [1.0] * 10

    # assert b.is_native_array(v)
    # assert id(b.from_backend(v, b)) == id(v)
    # assert id(b.from_backend(v, None)) == id(v)
    assert cp.allclose(b.from_numpy(v_np), v)
    assert cp.allclose(b.from_other(v_lst, dtype=cp.float64), v)
    assert get_backend(v) is b

    r = b.to_numpy(v)
    assert isinstance(r, np.ndarray)
    assert np.allclose(r, v_np)

    assert np.isclose(b.namespace.mean(v), 1.0)


@pytest.mark.skipif(NO_JAX, reason="No jax installed")
def test_utils_array_backend_jax():
    b = get_backend("jax")
    assert b.name == "jax"
    assert b is _JAX

    import jax.numpy as ja
    import numpy as np

    v = ja.ones(10, dtype=ja.float64)
    v_np = np.ones(10, dtype=np.float64)
    v_lst = [1.0] * 10

    # assert b.is_native_array(v)
    # assert id(b.from_backend(v, b)) == id(v)
    # assert id(b.from_backend(v, None)) == id(v)
    assert ja.allclose(b.from_numpy(v_np), v)
    assert ja.allclose(b.from_other(v_lst, dtype=ja.float64), v)
    assert get_backend(v) is b

    r = b.to_numpy(v)
    assert isinstance(r, np.ndarray)
    assert np.allclose(r, v_np)

    assert np.isclose(b.namespace.mean(v), 1.0)


def test_patched_namespace_numpy():
    from earthkit.utils.array import get_backend

    b = get_backend("numpy")
    ns = b.namespace

    c = ns.asarray([1.0, 2.0, 3.0])
    x = ns.asarray([1.0, 2.0, 3.0])
    assert ns.allclose(ns.polyval(c, x), ns.asarray([6.0, 17.0, 34.0]))

    assert ns.allclose(ns.pow(c, 2), ns.asarray([1.0, 4.0, 9.0]))


@pytest.mark.skipif(NO_TORCH, reason="No pytorch installed")
def test_patched_namespace_torch():
    from earthkit.utils.array import get_backend

    b = get_backend("torch")
    ns = b.namespace

    # polyval
    c = ns.asarray([1.0, 2.0, 3.0])
    x = ns.asarray([1.0, 2.0, 3.0])
    assert ns.allclose(ns.polyval(c, x), ns.asarray([6.0, 17.0, 34.0]))

    # pow
    assert ns.allclose(ns.pow(c, 2), ns.asarray([1.0, 4.0, 9.0]))

    # percentile
    x = ns.asarray([[10, 7, 4], [3, 2, 1]], dtype=ns.float64)
    assert ns.allclose(ns.percentile(x, 50), ns.asarray(3.5, dtype=ns.float64))

    # sign
    x = ns.asarray([1.0, -2.4, ns.nan], dtype=ns.float64)
    assert ns.allclose(ns.sign(x), ns.asarray([1.0, -1.0, ns.nan], dtype=ns.float64), equal_nan=True)


if __name__ == "__main__":
    from earthkit.utils.testing import main

    main(__file__)
