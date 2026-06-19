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
from earthkit.utils.array.namespace import (
    _CUPY_NAMESPACE,
    _JAX_NAMESPACE,
    _NUMPY_NAMESPACE,
    _TORCH_NAMESPACE,
    UnknownPatchedNamespace,
)
from earthkit.utils.array.testing.testing import NO_CUPY, NO_JAX, NO_TORCH


def _random_choice(xp, rng1, rng2):
    random_samples = xp.choice(xp.arange(10, dtype=float), size=5, replace=False)
    assert random_samples.shape == (5,)
    assert len(xp.unique(random_samples)) == len(random_samples)
    random_samples = xp.choice(xp.arange(2, dtype=float), size=5, replace=True)
    assert random_samples.shape == (5,)
    assert len(xp.unique(random_samples)) != len(random_samples)

    random_samples1 = xp.choice(xp.arange(10, dtype=float), size=5, replace=False, generator=rng1)
    random_samples2 = xp.choice(xp.arange(10, dtype=float), size=5, replace=False, generator=rng2)
    assert xp.allclose(random_samples1, random_samples2)


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
    q = xp.asarray(50)
    assert xp.allclose(xp.percentile(arr, q), res)
    assert generic_xp.allclose(generic_xp.percentile(arr, q), res)
    assert xp.allclose(xp.quantile(arr, q / 100), res)
    assert generic_xp.allclose(generic_xp.quantile(arr, q / 100), res)

    # test dtype, shape, size and device
    assert xp.dtype(arr) is not None
    assert xp.shape(arr) == (3,)
    assert xp.size(arr) == 3
    assert xp.device(arr) in xp.__array_namespace_info__().devices()
    xp.to_device(arr, "cpu")  # should not raise

    # TODO: test histogramdd and histogram2d

    _random_choice(xp, rng1=xp.random.default_rng(0), rng2=xp.random.default_rng(0))


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
    assert xp.device(arr) in xp.__array_namespace_info__().devices()

    assert xp.allclose(xp.rad2deg(arr), generic_xp.rad2deg(arr))
    assert xp.allclose(xp.deg2rad(arr), generic_xp.deg2rad(arr))

    # TODO: test histogramdd and histogram2d

    g1 = xp.Generator()
    g1.manual_seed(0)
    g2 = xp.Generator()
    g2.manual_seed(0)

    _random_choice(xp, rng1=g1, rng2=g2)


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
    assert xp.device(arr) in xp.__array_namespace_info__().devices()

    assert xp.allclose(xp.rad2deg(arr), generic_xp.rad2deg(arr))
    assert xp.allclose(xp.deg2rad(arr), generic_xp.deg2rad(arr))

    # TODO: test histogramdd and histogram2d

    _random_choice(xp, rng1=xp.random.default_rng(0), rng2=xp.random.default_rng(0))


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
    assert xp.device(arr) in xp.__array_namespace_info__().devices()

    assert xp.allclose(xp.rad2deg(arr), generic_xp.rad2deg(arr))
    assert xp.allclose(xp.deg2rad(arr), generic_xp.deg2rad(arr))

    # TODO: test histogramdd and histogram2d

    import jax.random

    _random_choice(xp, rng1=jax.random.PRNGKey(0), rng2=jax.random.PRNGKey(0))


if __name__ == "__main__":
    from earthkit.utils.testing import main

    main(__file__)
