# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import builtins
import copy
import sys
from typing import Tuple
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from earthkit.utils.decorators._xarray_ufunc import _infer_output_count, xarray_ufunc

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
        """Test xarray ufunc with keep_attrs passed in xarray_apply_ufunc_kwargs."""
        data = TEST_XARRAY_DATAARRAY.copy()
        data.attrs["unit"] = "meters"

        def add_one(x):
            return x + 1

        result = xarray_ufunc(
            add_one,
            data,
            xarray_ufunc_kwargs={"xarray_apply_ufunc_kwargs": {"keep_attrs": False}},
        )

        assert isinstance(result, xr.DataArray)
        assert "unit" not in result.attrs

    def test_ufunc_legacy_xarray_kwargs(self):
        """apply_ufunc kwargs given directly in xarray_ufunc_kwargs are still forwarded."""
        data = TEST_XARRAY_DATAARRAY.copy()
        data.attrs["unit"] = "meters"

        def add_one(x):
            return x + 1

        result = xarray_ufunc(add_one, data, xarray_ufunc_kwargs={"keep_attrs": False})

        assert "unit" not in result.attrs

    def test_ufunc_legacy_core_dims(self):
        """The kwargs used by earthkit-meteo (core dims, dtypes) work at the top level."""
        da = xr.DataArray(np.arange(6.0).reshape(2, 3), dims=["a", "b"])

        def total(x):
            return x.sum(axis=-1)

        result = xarray_ufunc(
            total,
            da,
            xarray_ufunc_kwargs={"input_core_dims": [["b"]], "output_core_dims": [[]], "output_dtypes": [float]},
        )

        assert result.dims == ("a",)
        np.testing.assert_array_equal(result.values, [3.0, 12.0])

    def test_ufunc_legacy_mixed_with_options(self):
        """Legacy keys can be combined with parameters; nested apply_ufunc kwargs take precedence."""
        data = TEST_XARRAY_DATAARRAY.copy()
        data.attrs["unit"] = "meters"

        def add_one(x):
            return x + 1

        result = xarray_ufunc(
            add_one,
            data,
            xarray_ufunc_kwargs={
                "parameters": "wind_speed",
                "keep_attrs": False,
                "xarray_apply_ufunc_kwargs": {"keep_attrs": True},
            },
        )

        assert result.name == "ws"
        assert result.attrs["unit"] == "meters"

    def test_ufunc_does_not_modify_xarray_ufunc_kwargs(self):
        opts = {"parameters": "wind_speed", "keep_attrs": False, "xarray_apply_ufunc_kwargs": {}}
        xarray_ufunc(lambda x: x + 1, TEST_XARRAY_DATAARRAY, xarray_ufunc_kwargs=opts)
        assert opts == {"parameters": "wind_speed", "keep_attrs": False, "xarray_apply_ufunc_kwargs": {}}

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
                with pytest.raises(ImportError):
                    xarray_ufunc(add_one, TEST_NUMPY_ARRAY)


class TestXarrayUfuncParametersAndAttrs:
    """Test the "parameters" and "attrs" options of xarray_ufunc_kwargs."""

    @staticmethod
    def _add_one(x):
        return x + 1

    @staticmethod
    def _split(x) -> Tuple[float, float]:
        return x / 2, x * 2

    def test_parameters_sets_name_and_metadata(self):
        r = xarray_ufunc(self._add_one, TEST_XARRAY_DATAARRAY, xarray_ufunc_kwargs={"parameters": "wind_speed"})
        assert r.name == "ws"
        assert r.attrs["long_name"] == "Wind speed"
        assert r.attrs["standard_name"] == "wind_speed"
        assert r.attrs["units"] == "m s-1"

    def test_attrs_adds_attributes(self):
        r = xarray_ufunc(
            self._add_one,
            TEST_XARRAY_DATAARRAY,
            xarray_ufunc_kwargs={"parameters": "wind_speed", "attrs": {"foo": "bar"}},
        )
        assert r.attrs["foo"] == "bar"
        assert r.attrs["standard_name"] == "wind_speed"

    def test_attrs_overrides_metadata(self):
        r = xarray_ufunc(
            self._add_one,
            TEST_XARRAY_DATAARRAY,
            xarray_ufunc_kwargs={"parameters": "wind_speed", "attrs": {"long_name": "My speed"}},
        )
        assert r.attrs["long_name"] == "My speed"

    def test_attrs_none_removes_attribute(self):
        r = xarray_ufunc(
            self._add_one,
            TEST_XARRAY_DATAARRAY,
            xarray_ufunc_kwargs={"parameters": "wind_speed", "attrs": {"standard_name": None}},
        )
        assert "standard_name" not in r.attrs

    def test_attrs_multi_output_list(self):
        r = xarray_ufunc(
            self._split,
            TEST_XARRAY_DATAARRAY,
            xarray_ufunc_kwargs={
                "parameters": ["wind_speed", "temperature"],
                "attrs": [{"foo": 1}, None],
            },
        )
        assert [da.name for da in r] == ["ws", "t"]
        assert r[0].attrs["foo"] == 1
        assert "foo" not in r[1].attrs

    def test_attrs_multi_output_single_dict(self):
        r = xarray_ufunc(
            self._split,
            TEST_XARRAY_DATAARRAY,
            xarray_ufunc_kwargs={"parameters": ("wind_speed", "temperature"), "attrs": {"foo": 1}},
        )
        assert all(da.attrs["foo"] == 1 for da in r)

    def test_attrs_multi_output_length_mismatch(self):
        with pytest.raises(ValueError, match="must match"):
            xarray_ufunc(
                self._split,
                TEST_XARRAY_DATAARRAY,
                xarray_ufunc_kwargs={"parameters": ["wind_speed", "temperature"], "attrs": [{"foo": 1}]},
            )

    @pytest.mark.parametrize("attrs", ["foo", 1, [1, 2]])
    def test_attrs_multi_output_invalid_type(self, attrs):
        with pytest.raises(ValueError, match="dict or a list/tuple of dicts"):
            xarray_ufunc(
                self._split,
                TEST_XARRAY_DATAARRAY,
                xarray_ufunc_kwargs={"parameters": ["wind_speed", "temperature"], "attrs": attrs},
            )

    @pytest.mark.parametrize("attrs", [[{"foo": 1}], "foo", 1])
    def test_attrs_single_output_invalid_type(self, attrs):
        with pytest.raises(ValueError, match="must be a dict"):
            xarray_ufunc(
                self._add_one,
                TEST_XARRAY_DATAARRAY,
                xarray_ufunc_kwargs={"parameters": "wind_speed", "attrs": attrs},
            )

    def test_attrs_without_parameters_raises(self):
        with pytest.raises(ValueError, match="requires 'parameters'"):
            xarray_ufunc(self._add_one, TEST_XARRAY_DATAARRAY, xarray_ufunc_kwargs={"attrs": {"foo": 1}})

    def test_old_variables_key_no_longer_sets_metadata(self):
        """The removed "variables" key is no longer an option.

        It is forwarded to :func:`xarray.apply_ufunc` as an unknown keyword, which raises
        a TypeError.
        """
        with pytest.raises(TypeError, match="unexpected keyword argument 'variables'"):
            xarray_ufunc(self._add_one, TEST_XARRAY_DATAARRAY, xarray_ufunc_kwargs={"variables": "wind_speed"})


class TestXarrayUfuncParameterDefinitions:
    """Test the "parameter_definitions" option of xarray_ufunc_kwargs."""

    @staticmethod
    def _add_one(x):
        return x + 1

    @staticmethod
    def _split(x) -> Tuple[float, float]:
        return x / 2, x * 2

    @staticmethod
    def _definition(**overrides):
        """A valid parameter definition, with ``overrides`` applied to it."""
        item = {
            "variable": "custom",
            "units": "K",
            "long_name": "Custom parameter",
            "standard_name": None,
            "cf_units": None,
        }
        item.update(overrides)
        return item

    def test_custom_name_single_output_cf_units(self):
        """A custom name sets the variable name and the metadata; cf_units is used as units."""
        definitions = {
            "my_parameter": self._definition(
                variable="my_var",
                long_name="My parameter",
                standard_name="my_standard_name",
                cf_units="m",
            )
        }
        r = xarray_ufunc(
            self._add_one,
            TEST_XARRAY_DATAARRAY,
            xarray_ufunc_kwargs={"parameters": "my_parameter", "parameter_definitions": definitions},
        )
        assert r.name == "my_var"
        assert r.attrs["long_name"] == "My parameter"
        assert r.attrs["standard_name"] == "my_standard_name"
        # cf_units is not None: it is used as "units" instead of "units" itself
        assert r.attrs["units"] == "m"

    def test_custom_name_single_output_units_fallback(self):
        """When cf_units is None, "units" is used; a None standard_name is not set."""
        definitions = {"my_parameter": self._definition(variable="my_var", units="kg/s")}
        r = xarray_ufunc(
            self._add_one,
            TEST_XARRAY_DATAARRAY,
            xarray_ufunc_kwargs={"parameters": "my_parameter", "parameter_definitions": definitions},
        )
        assert r.name == "my_var"
        assert r.attrs["long_name"] == "Custom parameter"
        assert r.attrs["units"] == "kg/s"
        assert "standard_name" not in r.attrs

    def test_custom_name_multi_output_mixed_with_builtin(self):
        """A multi-output call can mix a custom name with a built-in one."""
        definitions = {"my_parameter": self._definition(variable="my_var", units="km")}
        r = xarray_ufunc(
            self._split,
            TEST_XARRAY_DATAARRAY,
            xarray_ufunc_kwargs={
                "parameters": ["my_parameter", "wind_speed"],
                "parameter_definitions": definitions,
            },
        )
        assert [da.name for da in r] == ["my_var", "ws"]
        assert r[0].attrs["units"] == "km"
        assert r[1].attrs["standard_name"] == "wind_speed"
        assert r[1].attrs["units"] == "m s-1"

    def test_custom_name_overrides_builtin(self):
        """A caller definition overrides a built-in definition of the same name, for that call only."""
        definitions = {
            "wind_speed": self._definition(variable="ws_custom", long_name="My wind", units="km/h", cf_units="km/h")
        }
        r = xarray_ufunc(
            self._add_one,
            TEST_XARRAY_DATAARRAY,
            xarray_ufunc_kwargs={"parameters": "wind_speed", "parameter_definitions": definitions},
        )
        assert r.name == "ws_custom"
        assert r.attrs["long_name"] == "My wind"
        assert r.attrs["units"] == "km/h"
        assert "standard_name" not in r.attrs

        # the built-in database is untouched: a later call without definitions gets the built-in
        r = xarray_ufunc(self._add_one, TEST_XARRAY_DATAARRAY, xarray_ufunc_kwargs={"parameters": "wind_speed"})
        assert r.name == "ws"
        assert r.attrs["units"] == "m s-1"

    def test_attrs_with_custom_name(self):
        """The "attrs" option keeps working on top of a custom definition."""
        definitions = {"my_parameter": self._definition(variable="my_var")}
        r = xarray_ufunc(
            self._add_one,
            TEST_XARRAY_DATAARRAY,
            xarray_ufunc_kwargs={
                "parameters": "my_parameter",
                "parameter_definitions": definitions,
                "attrs": {"foo": "bar", "long_name": "Overridden"},
            },
        )
        assert r.name == "my_var"
        assert r.attrs["foo"] == "bar"
        assert r.attrs["long_name"] == "Overridden"

    def test_unknown_name_raises_key_error(self):
        """A name in neither "parameter_definitions" nor the database raises KeyError."""
        definitions = {"my_parameter": self._definition()}
        with pytest.raises(KeyError):
            xarray_ufunc(
                self._add_one,
                TEST_XARRAY_DATAARRAY,
                xarray_ufunc_kwargs={"parameters": "not_a_parameter", "parameter_definitions": definitions},
            )

    def test_invalid_definitions_not_a_dict(self):
        with pytest.raises(TypeError):
            xarray_ufunc(
                self._add_one,
                TEST_XARRAY_DATAARRAY,
                xarray_ufunc_kwargs={"parameters": "my_parameter", "parameter_definitions": "my_parameter"},
            )

    def test_invalid_definitions_entry_not_a_dict(self):
        with pytest.raises(TypeError):
            xarray_ufunc(
                self._add_one,
                TEST_XARRAY_DATAARRAY,
                xarray_ufunc_kwargs={"parameters": "my_parameter", "parameter_definitions": {"my_parameter": None}},
            )

    def test_invalid_definitions_missing_key(self):
        item = self._definition()
        del item["units"]
        with pytest.raises(ValueError, match="units"):
            xarray_ufunc(
                self._add_one,
                TEST_XARRAY_DATAARRAY,
                xarray_ufunc_kwargs={"parameters": "my_parameter", "parameter_definitions": {"my_parameter": item}},
            )

    def test_invalid_definitions_are_checked_even_when_unused(self):
        """Definitions are validated even when "parameters" does not use them."""
        with pytest.raises(ValueError):
            xarray_ufunc(
                self._add_one,
                TEST_XARRAY_DATAARRAY,
                xarray_ufunc_kwargs={"parameters": "wind_speed", "parameter_definitions": {"other": {}}},
            )

    def test_invalid_definitions_are_checked_without_parameters(self):
        """Definitions are validated even when "parameters" is not given."""
        with pytest.raises(TypeError):
            xarray_ufunc(self._add_one, TEST_XARRAY_DATAARRAY, xarray_ufunc_kwargs={"parameter_definitions": 1})

    def test_definitions_none_is_accepted(self):
        """None means no caller definitions and is accepted."""
        r = xarray_ufunc(
            self._add_one,
            TEST_XARRAY_DATAARRAY,
            xarray_ufunc_kwargs={"parameters": "wind_speed", "parameter_definitions": None},
        )
        assert r.name == "ws"

    def test_kwargs_and_definitions_not_modified(self):
        """Neither the caller's xarray_ufunc_kwargs nor the definitions dict is modified."""
        definitions = {"my_parameter": self._definition()}
        opts = {"parameters": "my_parameter", "parameter_definitions": definitions}
        expected_definitions = copy.deepcopy(definitions)
        expected_opts = copy.deepcopy(opts)

        xarray_ufunc(self._add_one, TEST_XARRAY_DATAARRAY, xarray_ufunc_kwargs=opts)

        assert opts == expected_opts
        assert definitions == expected_definitions
        assert definitions["my_parameter"] == expected_definitions["my_parameter"]
