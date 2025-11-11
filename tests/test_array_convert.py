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

from earthkit.utils.array import array_namespace
from earthkit.utils.array import convert
from earthkit.utils.array.namespace import _CUPY_NAMESPACE
from earthkit.utils.array.namespace import _JAX_NAMESPACE
from earthkit.utils.array.namespace import _NUMPY_NAMESPACE
from earthkit.utils.array.namespace import _TORCH_NAMESPACE
from earthkit.utils.array.testing.testing import NO_CUPY
from earthkit.utils.array.testing.testing import NO_JAX
from earthkit.utils.array.testing.testing import NO_TORCH


def test_array_convert_numpy_to_numpy():
    xp = _NUMPY_NAMESPACE
    x = xp.asarray([1.0, 2.0, 3.0])
    res = convert(x, array_namespace="numpy")
    assert array_namespace(res) is _NUMPY_NAMESPACE

    import numpy as np

    res = convert(x, array_namespace=np)
    assert array_namespace(res) is _NUMPY_NAMESPACE

    res = convert(x, array_namespace="numpy", device="cpu")
    assert array_namespace(res) is _NUMPY_NAMESPACE
    assert xp.device(res) == "cpu"


@pytest.mark.skipif(NO_TORCH, reason="No torch installed")
def test_array_convert_torch_to_torch():
    xp = _TORCH_NAMESPACE
    x = xp.asarray([1.0, 2.0, 3.0])
    res = convert(x, array_namespace="torch")
    assert array_namespace(res) is _TORCH_NAMESPACE

    import torch

    res = convert(x, array_namespace=torch)
    assert array_namespace(res) is _TORCH_NAMESPACE

    res = convert(x, array_namespace="torch", device="cpu")
    assert array_namespace(res) is _TORCH_NAMESPACE
    assert xp.device(res) == torch.device("cpu")

    if torch.backends.mps.is_available():
        res = convert(x, array_namespace="torch", device="mps:0")
        assert array_namespace(res) is _TORCH_NAMESPACE
        assert xp.device(res) == torch.device("mps:0")

    if torch.cuda.is_available():
        res = convert(x, array_namespace="torch", device="cuda:0")
        assert array_namespace(res) is _TORCH_NAMESPACE
        assert xp.device(res) == torch.device("cuda:0")


@pytest.mark.skipif(NO_CUPY, reason="No cupy installed")
def test_array_convert_cupy_to_cupy():
    xp = _CUPY_NAMESPACE
    x = xp.asarray([1.0, 2.0, 3.0])
    res = convert(x, array_namespace="cupy")
    assert array_namespace(res) is _CUPY_NAMESPACE

    import cupy as cp

    res = convert(x, array_namespace=cp)
    assert array_namespace(res) is _CUPY_NAMESPACE

    res = convert(x, array_namespace="cupy", device="cuda:0")
    assert array_namespace(res) is _CUPY_NAMESPACE
    # assert xp.device(res) == "cuda:0" # Not implemented yet


@pytest.mark.skipif(NO_JAX, reason="No jax installed")
def test_array_convert_jax_to_jax():
    xp = _JAX_NAMESPACE
    x = xp.asarray([1.0, 2.0, 3.0])
    res = convert(x, array_namespace="jax")
    assert array_namespace(res) is _JAX_NAMESPACE

    import jax.numpy as jnp

    res = convert(x, array_namespace=jnp)
    assert array_namespace(res) is _JAX_NAMESPACE

    # TODO: check other devices


@pytest.mark.skipif(NO_TORCH, reason="No torch installed")
def test_array_convert_numpy_to_torch():
    import torch

    xp = _NUMPY_NAMESPACE
    x = xp.asarray([1.0, 2.0, 3.0], dtype="float32")
    res = convert(x, array_namespace="torch")
    assert array_namespace(res) is _TORCH_NAMESPACE
    assert xp.device(res) == torch.device("cpu")

    if torch.backends.mps.is_available():
        res = convert(x, array_namespace="torch", device="mps:0")
        assert array_namespace(res) is _TORCH_NAMESPACE
        assert xp.device(res) == torch.device("mps:0")

    if torch.cuda.is_available():
        res = convert(x, array_namespace="torch", device="cuda:0")
        assert array_namespace(res) is _TORCH_NAMESPACE
        assert xp.device(res) == torch.device("cuda:0")


@pytest.mark.skipif(NO_TORCH, reason="No torch installed")
def test_array_convert_torch_to_numpy():
    import torch

    xp = _TORCH_NAMESPACE

    x = xp.asarray([1.0, 2.0, 3.0])
    res = convert(x, array_namespace="numpy")
    assert array_namespace(res) is _NUMPY_NAMESPACE

    if torch.backends.mps.is_available():
        x = xp.asarray([1.0, 2.0, 3.0], dtype=torch.float32, device="mps:0")
        res = convert(x, array_namespace="numpy")
        assert array_namespace(res) is _NUMPY_NAMESPACE

    if torch.cuda.is_available():
        x = xp.asarray([1.0, 2.0, 3.0], device="cuda:0")
        res = convert(x, array_namespace="numpy")
        assert array_namespace(res) is _NUMPY_NAMESPACE


@pytest.mark.skipif(NO_CUPY, reason="No cupy installed")
def test_array_convert_numpy_to_cupy():
    xp = _NUMPY_NAMESPACE
    x = xp.asarray([1.0, 2.0, 3.0])
    x_np = convert(x, array_backend="cupy")
    assert array_namespace(x_np) is _CUPY_NAMESPACE


@pytest.mark.skipif(NO_CUPY, reason="No cupy installed")
def test_array_convert_cupy_to_numpy():
    xp = _CUPY_NAMESPACE
    x = xp.asarray([1.0, 2.0, 3.0])
    x_np = convert(x, array_backend="numpy")
    assert array_namespace(x_np) is _NUMPY_NAMESPACE


@pytest.mark.skipif(NO_TORCH, reason="No torch installed")
@pytest.mark.skipif(NO_CUPY, reason="No cupy installed")
def test_array_convert_torch_to_cupy():
    xp = _TORCH_NAMESPACE
    import torch

    x = xp.asarray([1.0, 2.0, 3.0])
    x_cp = convert(x, array_backend="cupy")
    assert array_namespace(x_cp) is _CUPY_NAMESPACE

    if torch.backends.mps.is_available():
        x = xp.asarray([1.0, 2.0, 3.0], dtype=torch.float32, device="mps:0")
        res = convert(x, array_namespace="cupy")
        assert array_namespace(res) is _CUPY_NAMESPACE

    if torch.cuda.is_available():
        x = xp.asarray([1.0, 2.0, 3.0], device="cuda:0")
        res = convert(x, array_namespace="cupy")
        assert array_namespace(res) is _CUPY_NAMESPACE


@pytest.mark.skipif(NO_TORCH, reason="No torch installed")
@pytest.mark.skipif(NO_CUPY, reason="No cupy installed")
def test_array_convert_cupy_to_torch():
    import torch

    xp = _CUPY_NAMESPACE
    x = xp.asarray([1.0, 2.0, 3.0], dtype="float32")
    res = convert(x, array_namespace="torch")
    assert array_namespace(res) is _TORCH_NAMESPACE
    assert xp.device(res) == torch.device("cpu")

    if torch.backends.mps.is_available():
        res = convert(x, array_namespace="torch", device="mps:0")
        assert array_namespace(res) is _TORCH_NAMESPACE
        assert xp.device(res) == torch.device("mps:0")

    if torch.cuda.is_available():
        res = convert(x, array_namespace="torch", device="cuda:0")
        assert array_namespace(res) is _TORCH_NAMESPACE
        assert xp.device(res) == torch.device("cuda:0")
