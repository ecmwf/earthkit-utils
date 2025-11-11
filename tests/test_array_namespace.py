#!/usr/bin/env python3

# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import array_api_compat
import pytest

from earthkit.utils.array import array_namespace
from earthkit.utils.array.namespace import _CUPY_NAMESPACE
from earthkit.utils.array.namespace import _JAX_NAMESPACE
from earthkit.utils.array.namespace import _NUMPY_NAMESPACE
from earthkit.utils.array.namespace import _TORCH_NAMESPACE
from earthkit.utils.array.namespace import UnknownPatchedNamespace
from earthkit.utils.array.testing.testing import NO_CUPY
from earthkit.utils.array.testing.testing import NO_JAX
from earthkit.utils.array.testing.testing import NO_TORCH


def test_array_namespace_numpy():
    xp = array_namespace("numpy")
    assert xp._earthkit_array_namespace_name == "numpy"
    assert xp is _NUMPY_NAMESPACE

    import numpy as np

    assert array_namespace(np) is _NUMPY_NAMESPACE

    v = np.ones(10)
    v_lst = [1.0] * 10
    v_hat = xp.asarray(v_lst)

    assert array_namespace(v) is _NUMPY_NAMESPACE
    assert array_namespace(v_lst) is _NUMPY_NAMESPACE
    assert array_namespace(v_hat) is _NUMPY_NAMESPACE

    assert xp.isclose(xp.mean(v), 1.0)
    assert xp.allclose(v_hat, v)


@pytest.mark.skipif(NO_TORCH, reason="No torch installed")
def test_array_namespace_torch():
    xp = array_namespace("torch")
    assert xp._earthkit_array_namespace_name == "torch"
    assert xp is _TORCH_NAMESPACE

    import torch

    assert array_namespace(torch) is _TORCH_NAMESPACE

    v = torch.ones(10)
    v_lst = [1.0] * 10
    v_hat = xp.asarray(v_lst)

    assert array_namespace(v) is _TORCH_NAMESPACE
    assert array_namespace(v_hat) is _TORCH_NAMESPACE

    assert xp.isclose(xp.mean(v), xp.asarray(1.0))
    assert xp.allclose(v_hat, v)


@pytest.mark.skipif(NO_CUPY, reason="No cupy installed")
def test_array_namespace_cupy():
    xp = array_namespace("cupy")
    assert xp._earthkit_array_namespace_name == "cupy"
    assert xp is _CUPY_NAMESPACE

    import cupy as cp

    assert array_namespace(cp) is _CUPY_NAMESPACE

    v = cp.ones(10)
    v_lst = [1.0] * 10
    v_hat = xp.asarray(v_lst)

    assert array_namespace(v) is _CUPY_NAMESPACE
    assert array_namespace(v_hat) is _CUPY_NAMESPACE

    assert xp.isclose(xp.mean(v), 1.0)
    assert xp.allclose(v_hat, v)


@pytest.mark.skipif(NO_JAX, reason="No jax installed")
def test_array_namespace_jax():
    xp = array_namespace("jax")
    assert xp._earthkit_array_namespace_name == "jax"
    assert xp is _JAX_NAMESPACE

    import jax.numpy as jnp

    assert array_namespace(jnp) is _JAX_NAMESPACE

    v = jnp.ones(10)
    v_lst = [1.0] * 10
    v_hat = xp.asarray(v_lst)

    assert array_namespace(v) is _JAX_NAMESPACE
    assert array_namespace(v_hat) is _JAX_NAMESPACE

    assert xp.isclose(xp.mean(v), 1.0)
    assert xp.allclose(v_hat, v)


def test_patched_namespace_numpy():
    xp = array_namespace("numpy")
    generic_xp = UnknownPatchedNamespace(array_api_compat.numpy)

    test_input = [1.0, 2.0, 3.0]
    arr = xp.asarray(test_input)

    # test polyval
    res = xp.asarray([6.0, 17.0, 34.0])
    assert xp.allclose(xp.polyval(arr, arr), res)
    assert generic_xp.allclose(generic_xp.polyval(arr, arr), res)

    # test percentile and quantile
    res = xp.asarray([2])
    q = xp.asarray([50])
    assert xp.allclose(xp.percentile(arr, q), res)
    assert generic_xp.allclose(generic_xp.percentile(arr, q), res)
    assert xp.allclose(xp.quantile(arr, q / 100), res)
    assert generic_xp.allclose(generic_xp.quantile(arr, q / 100), res)

    # test dtype, shape, size and device
    assert xp.dtype(arr) is not None
    assert xp.shape(arr) == (3,)
    assert xp.size(arr) == 3
    assert xp.device(arr) in xp.devices()

    # TODO: test histogramdd and histogram2d


@pytest.mark.skipif(NO_TORCH, reason="No torch installed")
def test_patched_namespace_torch():
    xp = array_namespace("torch")
    generic_xp = UnknownPatchedNamespace(array_api_compat.torch)

    test_input = [1.0, 2.0, 3.0]
    arr = xp.asarray(test_input)

    # test polyval
    res = xp.asarray([6.0, 17.0, 34.0])
    assert xp.allclose(xp.polyval(arr, arr), res)
    assert generic_xp.allclose(generic_xp.polyval(arr, arr), res)

    # test percentile and quantile
    res = xp.asarray([2])
    q = xp.asarray([50])
    assert xp.allclose(xp.percentile(arr, q), res)
    # assert generic_xp.allclose(generic_xp.percentile(arr, q), res) # .take issue for torch
    assert xp.allclose(xp.quantile(arr, q / 100), res)
    # assert generic_xp.allclose(generic_xp.quantile(arr, q/100), res) # .take issue for torch

    # test dtype, shape, size and device
    assert xp.dtype(arr) is not None
    assert xp.shape(arr) == (3,)
    assert xp.size(arr) == 3
    # assert xp.device(arr) in xp.devices() # string vs torch.device

    # TODO: test histogramdd and histogram2d


@pytest.mark.skipif(NO_CUPY, reason="No cupy installed")
def test_patched_namespace_cupy():
    xp = array_namespace("cupy")
    generic_xp = UnknownPatchedNamespace(array_api_compat.cupy)

    test_input = [1.0, 2.0, 3.0]
    arr = xp.asarray(test_input)

    # test polyval
    res = xp.asarray([6.0, 17.0, 34.0])
    assert xp.allclose(xp.polyval(arr, arr), res)
    assert generic_xp.allclose(generic_xp.polyval(arr, arr), res)

    # test percentile and quantile
    res = xp.asarray([2])
    q = xp.asarray([50])
    assert xp.allclose(xp.percentile(arr, q), res)
    assert generic_xp.allclose(generic_xp.percentile(arr, q), res)
    assert xp.allclose(xp.quantile(arr, q / 100), res)
    assert generic_xp.allclose(generic_xp.quantile(arr, q / 100), res)

    # test dtype, shape, size and device
    assert xp.dtype(arr) is not None
    assert xp.shape(arr) == (3,)
    assert xp.size(arr) == 3
    assert xp.device(arr) in xp.devices()

    # TODO: test histogramdd and histogram2d


@pytest.mark.skipif(NO_JAX, reason="No jax installed")
def test_patched_namespace_jax():
    xp = array_namespace("jax")
    import jax.numpy as jnp

    generic_xp = UnknownPatchedNamespace(jnp)

    test_input = [1.0, 2.0, 3.0]
    arr = xp.asarray(test_input)

    # test polyval
    res = xp.asarray([6.0, 17.0, 34.0])
    assert xp.allclose(xp.polyval(arr, arr), res)
    assert generic_xp.allclose(generic_xp.polyval(arr, arr), res)

    # test percentile and quantile
    res = xp.asarray([2])
    q = xp.asarray([50])
    assert xp.allclose(xp.percentile(arr, q), res)
    # assert generic_xp.allclose(generic_xp.percentile(arr, q), res)
    assert xp.allclose(xp.quantile(arr, q / 100), res)
    # assert generic_xp.allclose(generic_xp.quantile(arr, q/100), res)

    # test dtype, shape, size and device
    assert xp.dtype(arr) is not None
    assert xp.shape(arr) == (3,)
    assert xp.size(arr) == 3
    assert xp.device(arr) in xp.devices()

    # TODO: test histogramdd and histogram2d


if __name__ == "__main__":
    from earthkit.utils.testing import main

    main(__file__)
