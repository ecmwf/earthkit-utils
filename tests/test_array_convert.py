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

from earthkit.utils.array import convert_array
from earthkit.utils.array import get_backend
from earthkit.utils.array import to_device
from earthkit.utils.array.backend import _CUPY
from earthkit.utils.array.backend import _NUMPY
from earthkit.utils.array.backend import _TORCH
from earthkit.utils.testing import NO_CUPY
from earthkit.utils.testing import NO_TORCH


def test_array_convert_numpy_to_numpy():
    x = _NUMPY.asarray([1.0, 2.0, 3.0], dtype="float32")
    x_np = convert_array(x, target_backend="numpy")
    assert get_backend(x_np) is _NUMPY
    assert _NUMPY.allclose(x, x_np)


@pytest.mark.skipif(NO_TORCH, reason="No pytorch installed")
def test_array_convert_torch_to_torch():
    x = _TORCH.asarray([1.0, 2.0, 3.0], dtype="float32")
    x_torch = convert_array(x, target_backend="torch")
    assert get_backend(x_torch) is _TORCH
    assert _TORCH.allclose(x, x_torch)


@pytest.mark.skipif(NO_CUPY, reason="No cupy installed")
def test_array_convert_cupy_to_cupy():
    x = _CUPY.asarray([1.0, 2.0, 3.0], dtype="float32")
    x_cp = convert_array(x, target_backend="cupy")
    assert get_backend(x_cp) is _CUPY
    assert _CUPY.allclose(x, x_cp)


@pytest.mark.skipif(NO_TORCH, reason="No pytorch installed")
def test_array_convert_numpy_to_torch():
    x = _NUMPY.asarray([1.0, 2.0, 3.0], dtype="float32")
    x_torch = convert_array(x, target_backend="torch")
    assert get_backend(x_torch) is _TORCH
    assert _TORCH.allclose(x_torch, _TORCH.asarray([1.0, 2.0, 3.0], dtype="float32"))


@pytest.mark.skipif(NO_TORCH, reason="No pytorch installed")
def test_array_convert_torch_to_numpy():
    x = _TORCH.asarray([1.0, 2.0, 3.0], dtype="float32")
    assert get_backend(x) is _TORCH
    x_np = convert_array(x, target_backend="numpy")
    assert get_backend(x_np) is _NUMPY
    assert _NUMPY.allclose(x_np, _NUMPY.asarray([1.0, 2.0, 3.0], dtype="float32"))


@pytest.mark.skipif(NO_CUPY, reason="No cupy installed")
def test_array_convert_numpy_to_cupy():
    x = _NUMPY.asarray([1.0, 2.0, 3.0], dtype="float32")
    x_cp = convert_array(x, target_backend="cupy")
    assert get_backend(x_cp) is _CUPY
    assert _CUPY.allclose(x_cp, _CUPY.asarray([1.0, 2.0, 3.0], dtype="float32"))


@pytest.mark.skipif(NO_CUPY, reason="No cupy installed")
def test_array_convert_cupy_to_numpy():
    x = _CUPY.asarray([1.0, 2.0, 3.0], dtype="float32")
    assert get_backend(x) is _CUPY
    x_np = convert_array(x, target_backend="numpy")
    assert get_backend(x_np) is _NUMPY
    assert _NUMPY.allclose(x_np, _NUMPY.asarray([1.0, 2.0, 3.0], dtype="float32"))


@pytest.mark.skipif(NO_TORCH, reason="No pytorch installed")
@pytest.mark.skipif(NO_CUPY, reason="No cupy installed")
def test_array_convert_torch_to_cupy():
    x = _TORCH.asarray([1.0, 2.0, 3.0], dtype="float32")
    assert get_backend(x) is _TORCH
    x_cp = convert_array(x, target_backend="cupy")
    assert get_backend(x_cp) is _CUPY
    assert _CUPY.allclose(x_cp, _CUPY.asarray([1.0, 2.0, 3.0], dtype="float32"))


@pytest.mark.skipif(NO_TORCH, reason="No pytorch installed")
@pytest.mark.skipif(NO_CUPY, reason="No cupy installed")
def test_array_convert_cupy_to_torch():
    x = _CUPY.asarray([1.0, 2.0, 3.0], dtype="float32")
    assert get_backend(x) is _CUPY
    x_torch = convert_array(x, target_backend="torch")
    assert get_backend(x_torch) is _TORCH
    x_torch_cpu = to_device(x_torch, "cpu", array_backend="torch")
    assert _TORCH.allclose(x_torch_cpu, _TORCH.asarray([1.0, 2.0, 3.0], dtype="float32"))
