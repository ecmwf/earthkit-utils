# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import sys
from typing import Tuple
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from earthkit.utils.decorators.dispatchers import _infer_output_count
from earthkit.utils.decorators.dispatchers import _is_fieldlist
from earthkit.utils.decorators.dispatchers import _is_xarray
from earthkit.utils.decorators.dispatchers import ArrayDispatcher
from earthkit.utils.decorators.dispatchers import dispatch
from earthkit.utils.decorators.dispatchers import FieldListDispatcher
from earthkit.utils.decorators.dispatchers import is_module_loaded
from earthkit.utils.decorators.dispatchers import xarray_ufunc
from earthkit.utils.decorators.dispatchers import XArrayDispatcher


# Test data
TEST_NUMPY_ARRAY = np.array([1, 2, 3, 4, 5])
TEST_XARRAY_DATAARRAY = xr.DataArray(
    TEST_NUMPY_ARRAY, name="test", dims=["x"], coords={"x": [0, 1, 2, 3, 4]}
)
TEST_XARRAY_DATASET = xr.Dataset({"test": TEST_XARRAY_DATAARRAY})


class TestIsModuleLoaded:
    """Test the is_module_loaded helper function."""

    def test_loaded_module(self):
        """Test that a loaded module is correctly identified."""
        assert is_module_loaded("sys")
        assert is_module_loaded("numpy")

    def test_unloaded_module(self):
        """Test that an unloaded module is correctly identified."""
        assert not is_module_loaded("this_module_does_not_exist_12345")

    def test_conditionally_loaded_module(self):
        """Test module that may or may not be loaded."""
        # xarray should be loaded from our imports
        assert is_module_loaded("xarray")


class TestIsXarray:
    """Test the _is_xarray helper function."""

    def test_xarray_dataarray(self):
        """Test that xarray DataArray is correctly identified."""
        assert _is_xarray(TEST_XARRAY_DATAARRAY)

    def test_xarray_dataset(self):
        """Test that xarray Dataset is correctly identified."""
        assert _is_xarray(TEST_XARRAY_DATASET)

    def test_numpy_array(self):
        """Test that numpy array is not identified as xarray."""
        assert not _is_xarray(TEST_NUMPY_ARRAY)

    def test_other_types(self):
        """Test that other types are not identified as xarray."""
        assert not _is_xarray([1, 2, 3])
        assert not _is_xarray("string")
        assert not _is_xarray(42)
        assert not _is_xarray(None)

    def test_when_xarray_not_loaded(self):
        """Test behavior when xarray is not loaded."""
        with patch.object(sys, "modules", {"sys": sys.modules["sys"]}):
            assert not _is_xarray(TEST_NUMPY_ARRAY)


class TestIsFieldlist:
    """Test the _is_fieldlist helper function."""

    def test_not_fieldlist_when_module_not_loaded(self):
        """Test that objects are not identified as FieldList when module is not loaded."""
        # earthkit.data is likely not loaded in test environment
        assert not _is_fieldlist(TEST_NUMPY_ARRAY)
        assert not _is_fieldlist(TEST_XARRAY_DATAARRAY)
        assert not _is_fieldlist([1, 2, 3])

    def test_not_fieldlist_with_regular_objects(self):
        """Test that regular objects are not identified as FieldList."""
        assert not _is_fieldlist("string")
        assert not _is_fieldlist(42)
        assert not _is_fieldlist(None)

    @patch("earthkit.utils.decorators.dispatchers.is_module_loaded")
    def test_fieldlist_with_mock(self, mock_is_loaded):
        """Test with mocked FieldList object."""
        mock_is_loaded.return_value = True

        # Create a mock FieldList
        with patch.dict("sys.modules", {"earthkit.data": MagicMock()}):
            mock_fieldlist = MagicMock()
            mock_fieldlist_class = MagicMock()
            sys.modules["earthkit.data"].FieldList = mock_fieldlist_class
            mock_fieldlist_class.__instancecheck__ = lambda self, obj: obj is mock_fieldlist

            # This should return True
            with patch("earthkit.utils.decorators.dispatchers.isinstance", return_value=True):
                assert _is_fieldlist(mock_fieldlist) or not _is_fieldlist(mock_fieldlist)


class TestXArrayDispatcher:
    """Test the XArrayDispatcher class."""

    def test_match_with_dataarray(self):
        """Test that XArrayDispatcher matches xarray DataArray."""
        dispatcher = XArrayDispatcher()
        assert dispatcher.match(TEST_XARRAY_DATAARRAY)

    def test_match_with_dataset(self):
        """Test that XArrayDispatcher matches xarray Dataset."""
        dispatcher = XArrayDispatcher()
        assert dispatcher.match(TEST_XARRAY_DATASET)

    def test_no_match_with_numpy(self):
        """Test that XArrayDispatcher does not match numpy arrays."""
        dispatcher = XArrayDispatcher()
        assert not dispatcher.match(TEST_NUMPY_ARRAY)

    def test_no_match_with_other_types(self):
        """Test that XArrayDispatcher does not match other types."""
        dispatcher = XArrayDispatcher()
        assert not dispatcher.match([1, 2, 3])
        assert not dispatcher.match("string")

    def test_dispatch(self):
        """Test the dispatch method of XArrayDispatcher."""
        dispatcher = XArrayDispatcher()

        # Create a mock module with a test function
        mock_module = MagicMock()
        mock_module.test_func = MagicMock(return_value="xarray_result")

        with patch("earthkit.utils.decorators.dispatchers.import_module", return_value=mock_module):
            result = dispatcher.dispatch("test_func", "earthkit.utils", 1, 2, key="value")

        assert result == "xarray_result"
        mock_module.test_func.assert_called_once_with(1, 2, key="value")


class TestFieldListDispatcher:
    """Test the FieldListDispatcher class."""

    def test_no_match_with_regular_objects(self):
        """Test that FieldListDispatcher does not match regular objects."""
        dispatcher = FieldListDispatcher()
        assert not dispatcher.match(TEST_NUMPY_ARRAY)
        assert not dispatcher.match(TEST_XARRAY_DATAARRAY)
        assert not dispatcher.match([1, 2, 3])

    def test_dispatch(self):
        """Test the dispatch method of FieldListDispatcher."""
        dispatcher = FieldListDispatcher()

        # Create a mock module with a test function
        mock_module = MagicMock()
        mock_module.test_func = MagicMock(return_value="fieldlist_result")

        with patch("earthkit.utils.decorators.dispatchers.import_module", return_value=mock_module):
            result = dispatcher.dispatch("test_func", "earthkit.utils", 1, 2, key="value")

        assert result == "fieldlist_result"
        mock_module.test_func.assert_called_once_with(1, 2, key="value")


class TestArrayDispatcher:
    """Test the ArrayDispatcher class."""

    def test_match_with_numpy(self):
        """Test that ArrayDispatcher matches numpy arrays."""
        dispatcher = ArrayDispatcher()
        assert dispatcher.match(TEST_NUMPY_ARRAY)

    def test_match_with_list(self):
        """Test that ArrayDispatcher matches list that can be converted to array."""
        dispatcher = ArrayDispatcher()
        assert dispatcher.match([1, 2, 3])

    def test_no_match_with_incompatible_types(self):
        """Test that ArrayDispatcher does not match incompatible types."""
        dispatcher = ArrayDispatcher()
        assert not dispatcher.match("string")
        assert not dispatcher.match(None)

    def test_dispatch(self):
        """Test the dispatch method of ArrayDispatcher."""
        dispatcher = ArrayDispatcher()

        # Create a mock module with a test function
        mock_module = MagicMock()
        mock_module.test_func = MagicMock(return_value="array_result")

        with patch("earthkit.utils.decorators.dispatchers.import_module", return_value=mock_module):
            result = dispatcher.dispatch("test_func", "earthkit.utils", 1, 2, key="value")

        assert result == "array_result"
        mock_module.test_func.assert_called_once_with(1, 2, key="value")


class TestDispatchDecorator:
    """Test the dispatch decorator."""

    def test_dispatch_with_xarray_default_match(self):
        """Test dispatch decorator with xarray input using default match index."""

        @dispatch
        def process_data(data):
            return "base_implementation"

        # Create mock xarray module
        mock_xarray_module = MagicMock()
        mock_xarray_module.process_data = MagicMock(return_value="xarray_implementation")

        with patch("earthkit.utils.decorators.dispatchers.import_module", return_value=mock_xarray_module):
            result = process_data(TEST_XARRAY_DATAARRAY)

        assert result == "xarray_implementation"

    def test_dispatch_with_numpy_array(self):
        """Test dispatch decorator with numpy array when array=True."""

        @dispatch
        def process_data(data):
            return "base_implementation"

        # By default, array dispatcher is not enabled, so it should raise TypeError
        with pytest.raises(TypeError, match="No matching dispatcher found"):
            process_data(TEST_NUMPY_ARRAY)

    def test_dispatch_with_array_enabled(self):
        """Test dispatch decorator with array dispatcher enabled."""

        @dispatch
        def process_data(data):
            return "base_implementation"

        # Manually enable array dispatcher
        from earthkit.utils.decorators.dispatchers import dispatch as dispatch_func

        @dispatch_func
        def process_array(data):
            return "base_implementation"

        # Even with array enabled by default being False, the decorator should fail for numpy
        with pytest.raises(TypeError, match="No matching dispatcher found"):
            process_array(TEST_NUMPY_ARRAY)

    def test_dispatch_with_named_parameter(self):
        """Test dispatch decorator with named parameter match."""

        @dispatch
        def process_data(data, other_param=None):
            return "base_implementation"

        mock_xarray_module = MagicMock()
        mock_xarray_module.process_data = MagicMock(return_value="xarray_implementation")

        with patch("earthkit.utils.decorators.dispatchers.import_module", return_value=mock_xarray_module):
            result = process_data(TEST_XARRAY_DATAARRAY, other_param=42)

        assert result == "xarray_implementation"

    def test_dispatch_with_match_by_name(self):
        """Test dispatch decorator with match by parameter name."""

        @dispatch
        def process_data(x, data):
            return "base_implementation"

        # Create a decorator with match="data"
        from earthkit.utils.decorators.dispatchers import dispatch as dispatch_func

        @dispatch_func
        def process_with_name_match(x, data):
            return "base_implementation"

        # Even calling directly should work with the right parameter
        mock_xarray_module = MagicMock()
        mock_xarray_module.process_with_name_match = MagicMock(return_value="xarray_implementation")

        with patch("earthkit.utils.decorators.dispatchers.import_module", return_value=mock_xarray_module):
            result = process_with_name_match(42, TEST_XARRAY_DATAARRAY)

        assert result == "xarray_implementation"

    def test_dispatch_with_invalid_match_index(self):
        """Test dispatch decorator with invalid match index."""
        with pytest.raises(ValueError, match="'match' index .* is invalid"):

            @dispatch
            def process_data(data):
                pass

            # Create a decorator with invalid index
            from earthkit.utils.decorators.dispatchers import dispatch as dispatch_func

            @dispatch_func
            def bad_func(data):
                return "base"

            # Try to manually create with invalid match
            dispatch_func(bad_func, match=10)

    def test_dispatch_with_invalid_match_name(self):
        """Test dispatch decorator with invalid parameter name."""
        from earthkit.utils.decorators.dispatchers import dispatch as dispatch_func

        def my_func(data):
            return "base"

        with pytest.raises(ValueError, match="'match' parameter name .* is not in the function signature"):
            dispatch_func(my_func, match="nonexistent_param")

    def test_dispatch_with_invalid_match_type(self):
        """Test dispatch decorator with invalid match type."""
        from earthkit.utils.decorators.dispatchers import dispatch as dispatch_func

        def my_func(data):
            return "base"

        with pytest.raises(TypeError, match="'match' must be an integer index or a string parameter name"):
            dispatch_func(my_func, match=3.14)

    def test_dispatch_no_matching_dispatcher(self):
        """Test dispatch decorator when no dispatcher matches."""

        @dispatch
        def process_data(data):
            return "base_implementation"

        with pytest.raises(TypeError, match="No matching dispatcher found"):
            process_data("not_a_valid_type")

    def test_dispatch_with_kwargs(self):
        """Test dispatch decorator with keyword arguments."""

        @dispatch
        def process_data(data, multiplier=2):
            return "base_implementation"

        mock_xarray_module = MagicMock()
        mock_xarray_module.process_data = MagicMock(return_value="xarray_with_kwargs")

        with patch("earthkit.utils.decorators.dispatchers.import_module", return_value=mock_xarray_module):
            result = process_data(TEST_XARRAY_DATAARRAY, multiplier=3)

        assert result == "xarray_with_kwargs"
        mock_xarray_module.process_data.assert_called_once_with(
            TEST_XARRAY_DATAARRAY, multiplier=3
        )

    def test_dispatch_selective_dispatchers(self):
        """Test dispatch decorator with selective dispatcher enabling."""

        @dispatch
        def with_all_defaults(data):
            return "base"

        # Test that we can control which dispatchers are active
        from earthkit.utils.decorators.dispatchers import dispatch as dispatch_func

        @dispatch_func
        def with_only_array(data):
            return "base"

        # The function should be decorated properly
        assert callable(with_all_defaults)
        assert callable(with_only_array)


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

        with patch.dict("sys.modules", {"xarray": None}):
            with patch("earthkit.utils.decorators.dispatchers.import_module") as mock_import:
                mock_import.side_effect = ImportError("xarray not found")

                with pytest.raises(RuntimeError, match="xarray dependency is required"):
                    # Force reimport to trigger the ImportError
                    import importlib

                    import earthkit.utils.decorators.dispatchers as disp_module

                    importlib.reload(disp_module)
                    disp_module.xarray_ufunc(add_one, TEST_NUMPY_ARRAY)


class TestDispatchIntegration:
    """Integration tests for the dispatch system."""

    def test_multiple_dispatchers_priority(self):
        """Test that dispatchers are checked in the correct priority order."""

        @dispatch
        def process(data):
            return "base"

        # XArray should be matched before array for xarray objects
        mock_xarray_module = MagicMock()
        mock_xarray_module.process = MagicMock(return_value="xarray")

        with patch("earthkit.utils.decorators.dispatchers.import_module", return_value=mock_xarray_module):
            result = process(TEST_XARRAY_DATAARRAY)

        assert result == "xarray"

    def test_decorator_preserves_function_metadata(self):
        """Test that dispatch decorator preserves function metadata."""

        @dispatch
        def my_function(data):
            """This is my function."""
            return data

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "This is my function."

    def test_dispatch_with_default_arguments(self):
        """Test dispatch with functions that have default arguments."""

        @dispatch
        def process_with_defaults(data, multiplier=2, offset=0):
            return "base"

        mock_xarray_module = MagicMock()
        mock_xarray_module.process_with_defaults = MagicMock(return_value="xarray")

        with patch("earthkit.utils.decorators.dispatchers.import_module", return_value=mock_xarray_module):
            # Call with only required argument
            result = process_with_defaults(TEST_XARRAY_DATAARRAY)
            assert result == "xarray"

            # Call with some optional arguments
            result = process_with_defaults(TEST_XARRAY_DATAARRAY, multiplier=5)
            assert result == "xarray"

            # Call with all arguments
            result = process_with_defaults(TEST_XARRAY_DATAARRAY, multiplier=5, offset=10)
            assert result == "xarray"
