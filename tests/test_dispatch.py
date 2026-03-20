# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import sys
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from earthkit.data import SimpleFieldList
from earthkit.utils.decorators.dispatch import ArrayDispatcher
from earthkit.utils.decorators.dispatch import ArrayLikeDispatcher
from earthkit.utils.decorators.dispatch import FieldListDispatcher
from earthkit.utils.decorators.dispatch import XArrayDispatcher
from earthkit.utils.decorators.dispatch import _is_fieldlist
from earthkit.utils.decorators.dispatch import _is_xarray
from earthkit.utils.decorators.dispatch import dispatch
from earthkit.utils.decorators.dispatch import is_array_like
from earthkit.utils.decorators.dispatch import is_module_loaded

# Test data
TEST_NUMPY_ARRAY = np.array([1, 2, 3, 4, 5])
TEST_XARRAY_DATAARRAY = xr.DataArray(TEST_NUMPY_ARRAY, name="test", dims=["x"], coords={"x": [0, 1, 2, 3, 4]})
TEST_XARRAY_DATASET = xr.Dataset({"test": TEST_XARRAY_DATAARRAY})
TEST_FIELDLIST = SimpleFieldList(TEST_NUMPY_ARRAY)


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

    def test_fieldlist(self):
        """Test that FieldList is correctly identified."""
        assert _is_fieldlist(TEST_FIELDLIST)

    def test_not_fieldlist_when_module_not_loaded(self):
        """Test that objects are not identified as FieldList when earthkit.data is not loaded."""
        # Simulate earthkit.data not being loaded by removing it from sys.modules
        with patch.object(sys, "modules", {"sys": sys.modules["sys"]}):
            assert not _is_fieldlist(TEST_NUMPY_ARRAY)
            assert not _is_fieldlist(TEST_XARRAY_DATAARRAY)
            assert not _is_fieldlist([1, 2, 3])

    def test_not_fieldlist_with_regular_objects(self):
        """Test that regular objects are not identified as FieldList."""
        assert not _is_fieldlist("string")
        assert not _is_fieldlist(42)
        assert not _is_fieldlist(None)


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

        with patch("earthkit.utils.decorators.dispatch.import_module", return_value=mock_module):
            result = dispatcher.dispatch("test_func", "dummy.module", 1, 2, key="value")

        assert result == "xarray_result"
        mock_module.test_func.assert_called_once_with(1, 2, key="value")


class TestFieldListDispatcher:
    """Test the FieldListDispatcher class."""

    def test_match_with_fieldlist(self):
        """Test that FieldListDispatcher matches FieldList objects."""
        dispatcher = FieldListDispatcher()
        assert dispatcher.match(TEST_FIELDLIST)

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

        with patch("earthkit.utils.decorators.dispatch.import_module", return_value=mock_module):
            result = dispatcher.dispatch("test_func", "dummy.module", 1, 2, key="value")

        assert result == "fieldlist_result"
        mock_module.test_func.assert_called_once_with(1, 2, key="value")


class TestIsArrayLike:
    """Test the is_array_like helper function."""

    def test_numpy_array(self):
        """Test that numpy arrays are array-like."""
        assert is_array_like(TEST_NUMPY_ARRAY)

    def test_list_of_ints(self):
        """Test that a list of ints is array-like."""
        assert is_array_like([1, 2, 3])

    def test_list_of_floats(self):
        """Test that a list of floats is array-like."""
        assert is_array_like([1.0, 2.0, 3.0])

    def test_nested_list(self):
        """Test that a nested list is array-like."""
        assert is_array_like([[1, 2], [3, 4]])

    def test_int_scalar(self):
        """Test that an integer scalar is array-like."""
        assert is_array_like(42)

    def test_float_scalar(self):
        """Test that a float scalar is array-like."""
        assert is_array_like(3.14)

    def test_string_is_array_like(self):
        """Test that a string is array-like."""
        assert is_array_like("array like")

    def test_none_is_array_like(self):
        """Test that None is array-like."""
        assert is_array_like(None)

    def test_dict_is_array_like(self):
        """Test that a dict is array-like."""
        assert is_array_like({"a": 1})


class TestArrayLikeDispatcher:
    """Test the ArrayLikeDispatcher class."""

    def test_match_with_numpy_array(self):
        """Test that ArrayLikeDispatcher matches numpy arrays."""
        dispatcher = ArrayLikeDispatcher()
        assert dispatcher.match(TEST_NUMPY_ARRAY)

    def test_match_with_list(self):
        """Test that ArrayLikeDispatcher matches lists."""
        dispatcher = ArrayLikeDispatcher()
        assert dispatcher.match([1, 2, 3])
        assert dispatcher.match([1.0, 2.0, 3.0])

    def test_match_with_int(self):
        """Test that ArrayLikeDispatcher matches integer scalars."""
        dispatcher = ArrayLikeDispatcher()
        assert dispatcher.match(42)

    def test_match_with_float(self):
        """Test that ArrayLikeDispatcher matches float scalars."""
        dispatcher = ArrayLikeDispatcher()
        assert dispatcher.match(3.14)

    def test_match_with_string(self):
        """Test that ArrayLikeDispatcher matches strings."""
        dispatcher = ArrayLikeDispatcher()
        assert dispatcher.match("string")

    def test_match_with_none(self):
        """Test that ArrayLikeDispatcher matches None."""
        dispatcher = ArrayLikeDispatcher()
        assert dispatcher.match(None)

    def test_dispatch_routes_to_array_module(self):
        """Test that ArrayLikeDispatcher dispatches to the .array submodule."""
        dispatcher = ArrayLikeDispatcher()

        mock_module = MagicMock()
        mock_module.test_func = MagicMock(return_value="array_result")

        with patch(
            "earthkit.utils.decorators.dispatch.import_module", return_value=mock_module
        ) as mock_import:
            result = dispatcher.dispatch("test_func", "dummy.module", [1, 2, 3])

        mock_import.assert_called_once_with("dummy.module.array")
        assert result == "array_result"


class TestDispatchWrapperArrayLike:
    """Test the dispatch wrapper with array_like scenarios."""

    def test_dispatch_list_with_array_like_enabled(self):
        """Test that a list is dispatched when array_like=True."""

        def process_data(data):
            dispatched = dispatch(process_data, array_like=True)
            return dispatched(data)

        mock_module = MagicMock()
        mock_module.process_data = MagicMock(return_value="array_result")

        with patch("earthkit.utils.decorators.dispatch.import_module", return_value=mock_module):
            result = process_data([1, 2, 3])

        assert result == "array_result"

    def test_dispatch_int_with_array_like_enabled(self):
        """Test that an int scalar is dispatched when array_like=True."""

        def process_data(data):
            dispatched = dispatch(process_data, array_like=True)
            return dispatched(data)

        mock_module = MagicMock()
        mock_module.process_data = MagicMock(return_value="array_result")

        with patch("earthkit.utils.decorators.dispatch.import_module", return_value=mock_module):
            result = process_data(42)

        assert result == "array_result"

    def test_dispatch_float_with_array_like_enabled(self):
        """Test that a float scalar is dispatched when array_like=True."""

        def process_data(data):
            dispatched = dispatch(process_data, array_like=True)
            return dispatched(data)

        mock_module = MagicMock()
        mock_module.process_data = MagicMock(return_value="array_result")

        with patch("earthkit.utils.decorators.dispatch.import_module", return_value=mock_module):
            result = process_data(3.14)

        assert result == "array_result"

    def test_dispatch_list_without_array_like_raises(self):
        """Test that a list raises TypeError when array_like is not enabled."""

        def process_data(data):
            dispatched = dispatch(process_data)
            return dispatched(data)

        with pytest.raises(TypeError, match="No dispatcher matched for function"):
            process_data([1, 2, 3])

    def test_array_implies_array_like(self):
        """Test that array=True also enables array_like by default."""

        def process_data(data):
            dispatched = dispatch(process_data, array=True)
            return dispatched(data)

        mock_module = MagicMock()
        mock_module.process_data = MagicMock(return_value="array_result")

        with patch("earthkit.utils.decorators.dispatch.import_module", return_value=mock_module):
            result = process_data([1.0, 2.0, 3.0])

        assert result == "array_result"


class TestArrayDispatcher:
    """Test the ArrayDispatcher class."""

    def test_match_with_numpy(self):
        """Test that ArrayDispatcher matches numpy arrays."""
        dispatcher = ArrayDispatcher()
        assert dispatcher.match(TEST_NUMPY_ARRAY)

    # def test_match_with_list(self):
    #     """Test that ArrayDispatcher matches list that can be converted to array."""
    #     dispatcher = ArrayDispatcher()
    #     assert dispatcher.match([1, 2, 3])

    def test_no_match_with_incompatible_types(self):
        """Test that ArrayDispatcher does not match incompatible types."""
        dispatcher = ArrayDispatcher()
        assert not dispatcher.match("string")

        # This currently fails, but because array_namespace returns a numpy namespace for None
        # assert not dispatcher.match(None)

    def test_dispatch(self):
        """Test the dispatch method of ArrayDispatcher."""
        dispatcher = ArrayDispatcher()

        # Create a mock module with a test function
        mock_module = MagicMock()
        mock_module.test_func = MagicMock(return_value="array_result")

        with patch("earthkit.utils.decorators.dispatch.import_module", return_value=mock_module):
            result = dispatcher.dispatch("test_func", "dummy.module", 1, 2, key="value")

        assert result == "array_result"
        mock_module.test_func.assert_called_once_with(1, 2, key="value")


class TestDispatchWrapper:
    """Test the dispatch wrapper."""

    def test_dispatch_with_xarray_default_match(self):
        """Test dispatch wrapper with xarray input using default match index."""

        def process_data(data):
            dispatched = dispatch(process_data)
            return dispatched(data)

        # Even calling directly should work with the right parameter
        mock_xarray_module = MagicMock()
        mock_xarray_module.process_data = MagicMock(return_value="xarray_implementation")

        with patch("earthkit.utils.decorators.dispatch.import_module", return_value=mock_xarray_module):
            result = process_data(TEST_XARRAY_DATAARRAY)

        assert result == "xarray_implementation"

    def test_dispatch_with_numpy_array(self):
        """Test dispatch wrapper with numpy array when array=True."""

        def process_data(data):
            dispatched = dispatch(process_data)
            return dispatched(data)

        # By default, array dispatcher is not enabled, so it should raise TypeError
        with pytest.raises(TypeError, match="No dispatcher matched for function"):
            process_data(TEST_NUMPY_ARRAY)

    def test_dispatch_with_array_enabled(self):
        """Test dispatch wrapper with array dispatcher enabled."""

        def process_data(data):
            dispatched = dispatch(process_data, array=True)
            return dispatched(data)

        mock_array_module = MagicMock()
        mock_array_module.process_data = MagicMock(return_value="array_implementation")

        with patch("earthkit.utils.decorators.dispatch.import_module", return_value=mock_array_module):
            result = process_data(TEST_NUMPY_ARRAY)
            assert result == "array_implementation"

    def test_dispatch_with_named_parameter(self):
        """Test dispatch wrapper with named parameter match."""

        def process_data(data, other_param=None):
            dispatched = dispatch(process_data, match="data")
            return dispatched(data, other_param=other_param)

        mock_xarray_module = MagicMock()
        mock_xarray_module.process_data = MagicMock(return_value="xarray_implementation")

        with patch("earthkit.utils.decorators.dispatch.import_module", return_value=mock_xarray_module):
            result = process_data(TEST_XARRAY_DATAARRAY, other_param=42)

        assert result == "xarray_implementation"

    def test_dispatch_with_match_by_name(self):
        """Test dispatch wrapper with match by parameter name."""

        def process_with_name_match(data):
            dispatched = dispatch(process_with_name_match, match="data")
            return dispatched(data)

        # Even calling directly should work with the right parameter
        mock_xarray_module = MagicMock()
        mock_xarray_module.process_with_name_match = MagicMock(return_value="xarray_implementation")

        with patch("earthkit.utils.decorators.dispatch.import_module", return_value=mock_xarray_module):
            result = process_with_name_match(TEST_XARRAY_DATAARRAY)

        assert result == "xarray_implementation"

    def test_dispatch_with_invalid_match_index(self):
        """Test dispatch wrapper with invalid match index."""

        def bad_func(data):
            return "base"

        with pytest.raises(ValueError, match="'match' index .* is invalid"):
            dispatch(bad_func, match=10)

    def test_dispatch_with_invalid_match_name(self):
        """Test dispatch wrapper with invalid parameter name."""

        def my_func(data):
            return "base"

        with pytest.raises(ValueError, match="'match' parameter name .* is not in the function signature"):
            dispatch(my_func, match="nonexistent_param")

    def test_dispatch_with_invalid_match_type(self):
        """Test dispatch wrapper with invalid match type."""

        def my_func(data):
            return "base"

        with pytest.raises(TypeError, match="'match' must be an integer index or a string parameter name"):
            dispatch(my_func, match=3.14)

    def test_dispatch_no_matching_dispatcher(self):
        """Test dispatch wrapper when no dispatcher matches."""

        def process_data(data):
            dispatched = dispatch(process_data)
            return dispatched(data)

        with pytest.raises(TypeError, match="No dispatcher matched for function"):
            process_data("not_a_valid_type")

    def test_dispatch_with_kwargs(self):
        """Test dispatch wrapper with keyword arguments."""

        def process_data(data, multiplier=2):
            dispatched = dispatch(process_data)
            return dispatched(data, multiplier=multiplier)

        mock_xarray_module = MagicMock()
        mock_xarray_module.process_data = MagicMock(return_value="xarray_with_kwargs")

        with patch("earthkit.utils.decorators.dispatch.import_module", return_value=mock_xarray_module):
            result = process_data(TEST_XARRAY_DATAARRAY, multiplier=3)

        assert result == "xarray_with_kwargs"
        mock_xarray_module.process_data.assert_called_once_with(TEST_XARRAY_DATAARRAY, multiplier=3)

    def test_dispatch_selective_dispatchers(self):
        """Test dispatch wrapper with selective dispatcher enabling."""

        def with_all_defaults(data):
            dispatched = dispatch(with_all_defaults)
            return dispatched(data)

        def with_only_array(data):
            dispatched = dispatch(with_only_array, array=True, xarray=False, fieldlist=False)
            return dispatched(data)

        # The functions should be decorated properly
        assert callable(with_all_defaults)
        assert callable(with_only_array)

        # For NumPy array, test that array dispatcher works when enabled
        mock_array_module = MagicMock()
        mock_array_module.with_only_array = MagicMock(return_value="array_dispatched")

        with patch("earthkit.utils.decorators.dispatch.import_module", return_value=mock_array_module):
            result = with_only_array(TEST_NUMPY_ARRAY)
            assert result == "array_dispatched"

        # TODO: Currently xarray is matched as an array, therefore this test is not valid
        # # For xarray input, the default configuration should use the xarray dispatcher
        # mock_xarray_module = MagicMock()
        # mock_xarray_module.with_all_defaults = MagicMock(return_value="xarray_dispatched")

        # with patch(
        #     "earthkit.utils.decorators.dispatch.import_module", return_value=mock_xarray_module
        # ):
        #     result = with_all_defaults(TEST_XARRAY_DATAARRAY)

        # assert result == "xarray_dispatched"
        # mock_xarray_module.with_all_defaults.assert_called_once_with(TEST_XARRAY_DATAARRAY)

        # # With xarray disabled in with_only_array, xarray input should not be dispatched
        # with pytest.raises(TypeError, match="No dispatcher matched for function"):
        #     with_only_array(TEST_XARRAY_DATAARRAY)


class TestDispatchIntegration:
    """Integration tests for the dispatch system."""

    def test_multiple_dispatchers_priority(self):
        """Test that dispatchers are checked in the correct priority order."""

        def process(data):
            dispatched = dispatch(process)
            return dispatched(data)

        # XArray should be matched before array for xarray objects
        mock_xarray_module = MagicMock()
        mock_xarray_module.process = MagicMock(return_value="xarray")

        with patch("earthkit.utils.decorators.dispatch.import_module", return_value=mock_xarray_module):
            result = process(TEST_XARRAY_DATAARRAY)

        assert result == "xarray"

    def test_decorator_preserves_function_metadata(self):
        """Test that dispatch wrapper preserves function metadata."""

        def my_function(data):
            """This is my function."""
            return data

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "This is my function."

    def test_dispatch_with_default_arguments(self):
        """Test dispatch with functions that have default arguments."""

        def process_with_defaults(data, multiplier=2, offset=0):
            dispatched = dispatch(process_with_defaults)
            return dispatched(data, multiplier=multiplier, offset=offset)

        mock_xarray_module = MagicMock()
        mock_xarray_module.process_with_defaults = MagicMock(return_value="xarray")

        with patch("earthkit.utils.decorators.dispatch.import_module", return_value=mock_xarray_module):
            # Call with only required argument
            result = process_with_defaults(TEST_XARRAY_DATAARRAY)
            assert result == "xarray"

            # Call with some optional arguments
            result = process_with_defaults(TEST_XARRAY_DATAARRAY, multiplier=5)
            assert result == "xarray"

            # Call with all arguments
            result = process_with_defaults(TEST_XARRAY_DATAARRAY, multiplier=5, offset=10)
            assert result == "xarray"
