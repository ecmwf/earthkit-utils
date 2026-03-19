# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import builtins
import sys
from typing import Tuple
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from earthkit.utils.decorators.xarray_ufunc import _infer_output_count
from earthkit.utils.decorators.xarray_ufunc import xarray_ufunc

# Test data
TEST_NUMPY_ARRAY = np.array([1, 2, 3, 4, 5])
TEST_XARRAY_DATAARRAY = xr.DataArray(TEST_NUMPY_ARRAY, name="test", dims=["x"], coords={"x": [0, 1, 2, 3, 4]})


class TestInferOutputCount:
    """Test the _infer_output_count helper function."""

    def test_no_annotation(self):
        """Test function with no return annotation."""

        def func():
            return 1

        assert _infer_output_count(func) == 1

    def test_single_output(self):
        """Test function with single output annotation."""

        def func() -> float:
            return 1.0

        assert _infer_output_count(func) == 1

    def test_tuple_output(self):
        """Test function with tuple output annotation."""

        def func() -> Tuple[float, float]:
            return 1.0, 2.0

        assert _infer_output_count(func) == 2

    def test_tuple_output_three_elements(self):
        """Test function with tuple of three elements."""

        def func() -> Tuple[float, float, float]:
            return 1.0, 2.0, 3.0

        assert _infer_output_count(func) == 3

    def test_tuple_with_ellipsis(self):
        """Test function with tuple and ellipsis annotation."""

        def func() -> Tuple[float, ...]:
            return 1.0, 2.0, 3.0

        # Should return 1 because ellipsis means variable length
        assert _infer_output_count(func) == 1

    def test_empty_tuple(self):
        """Test function with empty tuple annotation."""

        def func() -> Tuple[()]:
            return ()

        assert _infer_output_count(func) == 1

    def test_non_tuple_annotation(self):
        """Test function with non-tuple annotation."""

        def func() -> int:
            return 42

        assert _infer_output_count(func) == 1


class TestXarrayUfunc:
    """Test the xarray_ufunc function."""

    def test_basic_ufunc(self):
        """Test basic xarray ufunc application."""

        def add_one(x):
            return x + 1

        result = xarray_ufunc(add_one, TEST_XARRAY_DATAARRAY)

        assert isinstance(result, xr.DataArray)
        np.testing.assert_array_equal(result.values, TEST_NUMPY_ARRAY + 1)

    def test_ufunc_with_kwargs(self):
        """Test xarray ufunc with keyword arguments."""

        def multiply(x, factor):
            return x * factor

        result = xarray_ufunc(multiply, TEST_XARRAY_DATAARRAY, factor=3)

        assert isinstance(result, xr.DataArray)
        np.testing.assert_array_equal(result.values, TEST_NUMPY_ARRAY * 3)

    def test_ufunc_preserves_attributes(self):
        """Test that xarray ufunc preserves attributes."""
        data = TEST_XARRAY_DATAARRAY.copy()
        data.attrs["unit"] = "meters"

        def identity(x):
            return x

        result = xarray_ufunc(identity, data)

        assert result.attrs.get("unit") == "meters"

    def test_ufunc_with_custom_xarray_kwargs(self):
        """Test xarray ufunc with custom xarray_ufunc_kwargs."""

        def add_one(x):
            return x + 1

        result = xarray_ufunc(
            add_one,
            TEST_XARRAY_DATAARRAY,
            xarray_ufunc_kwargs={"keep_attrs": False},
        )

        assert isinstance(result, xr.DataArray)

    def test_ufunc_with_multiple_outputs(self):
        """Test xarray ufunc with function returning multiple outputs."""

        def split_data(x) -> Tuple[float, float]:
            return x / 2, x * 2

        result = xarray_ufunc(split_data, TEST_XARRAY_DATAARRAY)

        # Result should be a tuple of DataArrays
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_ufunc_with_multiple_inputs(self):
        """Test xarray ufunc with multiple input arrays."""
        data1 = TEST_XARRAY_DATAARRAY
        data2 = TEST_XARRAY_DATAARRAY * 2

        def add_arrays(x, y):
            return x + y

        result = xarray_ufunc(add_arrays, data1, data2)

        assert isinstance(result, xr.DataArray)
        np.testing.assert_array_equal(result.values, TEST_NUMPY_ARRAY * 3)

    def test_ufunc_without_xarray_raises_error(self):
        """Test that xarray_ufunc raises error when xarray is not available."""

        def add_one(x):
            return x + 1

        # Simulate ImportError when xarray is imported inside xarray_ufunc
        with patch.dict(sys.modules, {"xarray": None}, clear=False):
            original_import = builtins.__import__

            def _mock_import(name, *args, **kwargs):
                if name == "xarray":
                    raise ImportError("xarray not found")
                return original_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", side_effect=_mock_import):
                with pytest.raises(RuntimeError, match="xarray dependency is required"):
                    xarray_ufunc(add_one, TEST_NUMPY_ARRAY)
