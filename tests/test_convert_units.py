#!/usr/bin/env python3

# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import numpy as np
import pytest
import xarray as xr

from earthkit.utils.units.convert import (
    are_compatible,
    are_equal,
    convert_array,
    convert_dataarray,
    convert_dataset,
    convert_units,
)
from earthkit.utils.units.units import Units, ureg

# ---- are_equal ----


class TestAreEqual:
    def test_same_units(self):
        assert are_equal("m", "m")

    def test_equivalent_units(self):
        assert are_equal("m/s", "m s-1")

    def test_different_units(self):
        assert not are_equal("m", "s")

    def test_none_none(self):
        # Both None -> both become "dimensionless"
        assert are_equal(None, None)

    def test_none_vs_unit(self):
        assert not are_equal(None, "m")


# ---- convert_array ----


class TestConvertArray:
    def test_basic_conversion(self):
        data = np.array([1.0, 2.0, 3.0])
        result = convert_array(data, target_units="km", source_units="m")
        np.testing.assert_allclose(result, [0.001, 0.002, 0.003])

    def test_temperature_conversion(self):
        data = np.array([0.0, 100.0])
        result = convert_array(data, target_units="degF", source_units="degC")
        np.testing.assert_allclose(result, [32.0, 212.0])

    def test_same_units_no_op(self):
        data = np.array([1.0, 2.0, 3.0])
        result = convert_array(data, target_units="m", source_units="m")
        np.testing.assert_array_equal(result, data)

    def test_missing_source_units_returns_unchanged(self):
        data = np.array([1.0, 2.0])
        result = convert_array(data, target_units="m", source_units=None)
        np.testing.assert_array_equal(result, data)

    def test_missing_target_units_returns_unchanged(self):
        data = np.array([1.0, 2.0])
        result = convert_array(data, target_units=None, source_units="m")
        np.testing.assert_array_equal(result, data)

    def test_missing_both_units_returns_unchanged(self):
        data = np.array([1.0, 2.0])
        result = convert_array(data, target_units=None, source_units=None)
        np.testing.assert_array_equal(result, data)

    def test_incompatible_units_returns_unchanged(self):
        data = np.array([1.0, 2.0])
        result = convert_array(data, target_units="kelvin", source_units="m")
        np.testing.assert_array_equal(result, data)

    def test_unrecognised_source_units_returns_unchanged(self):
        data = np.array([1.0, 2.0])
        result = convert_array(data, target_units="m", source_units="foobar")
        np.testing.assert_array_equal(result, data)

    def test_unrecognised_target_units_returns_unchanged(self):
        data = np.array([1.0, 2.0])
        result = convert_array(data, target_units="foobar", source_units="m")
        np.testing.assert_array_equal(result, data)

    def test_compound_units(self):
        data = np.array([1.0])
        result = convert_array(data, target_units="km/h", source_units="m/s")
        np.testing.assert_allclose(result, [3.6])

    def test_pint_style_units(self):
        data = np.array([1000.0])
        result = convert_array(data, target_units="kg m**-2", source_units="g m**-2")
        np.testing.assert_allclose(result, [1.0])


# ---- convert_dataarray ----


class TestConvertDataArray:
    def test_basic_conversion(self):
        da = xr.DataArray([1.0, 2.0, 3.0], attrs={"units": "m"})
        result = convert_dataarray(da, target_units="km", source_units="m")
        np.testing.assert_allclose(result.values, [0.001, 0.002, 0.003])

    def test_source_units_from_attrs(self):
        da = xr.DataArray([1000.0, 2000.0], attrs={"units": "m"})
        result = convert_dataarray(da, target_units="km", source_units=None)
        np.testing.assert_allclose(result.values, [1.0, 2.0])

    def test_source_units_overrides_attrs(self):
        # attrs say "km" but source_units says "m" -> should use "m"
        da = xr.DataArray([1.0, 2.0], attrs={"units": "km"})
        result = convert_dataarray(da, target_units="km", source_units="m")
        np.testing.assert_allclose(result.values, [0.001, 0.002])

    def test_updates_units_attr_to_target(self):
        da = xr.DataArray([1000.0], attrs={"units": "m"})
        result = convert_dataarray(da, target_units="km", source_units="m")
        assert result.attrs["units"] == "km"

    def test_preserves_user_provided_units_string(self):
        da = xr.DataArray([1.0], attrs={"units": "m/s"})
        result = convert_dataarray(da, target_units="km/h", source_units=None)
        assert result.attrs["units"] == "km/h"

    def test_preserves_other_attrs(self):
        da = xr.DataArray([1.0], attrs={"units": "m", "long_name": "distance"})
        result = convert_dataarray(da, target_units="km", source_units="m")
        assert result.attrs["long_name"] == "distance"

    def test_no_source_no_attrs_returns_unchanged(self):
        da = xr.DataArray([1.0, 2.0])
        result = convert_dataarray(da, target_units="km", source_units=None)
        np.testing.assert_array_equal(result.values, da.values)

    def test_same_units_no_op(self):
        da = xr.DataArray([1.0, 2.0], attrs={"units": "m"})
        result = convert_dataarray(da, target_units="m", source_units="m")
        np.testing.assert_array_equal(result.values, da.values)
        assert result.attrs["units"] == "m"

    def test_incompatible_units_returns_unchanged(self):
        da = xr.DataArray([1.0], attrs={"units": "m"})
        result = convert_dataarray(da, target_units="kelvin", source_units="m")
        np.testing.assert_array_equal(result.values, da.values)

    def test_unrecognised_units_returns_unchanged(self):
        da = xr.DataArray([1.0], attrs={"units": "foobar"})
        result = convert_dataarray(da, target_units="m", source_units=None)
        np.testing.assert_array_equal(result.values, da.values)

    def test_does_not_mutate_input(self):
        da = xr.DataArray([1000.0, 2000.0], attrs={"units": "m"})
        original_data = da.values.copy()
        convert_dataarray(da, target_units="km", source_units="m")
        np.testing.assert_array_equal(da.values, original_data)
        assert da.attrs["units"] == "m"

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            convert_dataarray("not a dataarray", target_units="km", source_units="m")

    def test_dict_target_units_by_name(self):
        da = xr.DataArray([1000.0, 2000.0], attrs={"units": "m"}, name="dist")
        result = convert_dataarray(da, target_units={"dist": "km"}, source_units=None)
        np.testing.assert_allclose(result.values, [1.0, 2.0])
        assert result.attrs["units"] == "km"

    def test_dict_target_units_name_not_found_returns_unchanged(self):
        da = xr.DataArray([1000.0], attrs={"units": "m"}, name="dist")
        result = convert_dataarray(da, target_units={"other": "km"}, source_units=None)
        np.testing.assert_array_equal(result.values, da.values)
        assert result.attrs["units"] == "m"

    def test_dict_source_units_by_name(self):
        da = xr.DataArray([1.0, 2.0], attrs={"units": "km"}, name="dist")
        result = convert_dataarray(da, target_units="km", source_units={"dist": "m"})
        np.testing.assert_allclose(result.values, [0.001, 0.002])

    def test_dict_source_units_falls_back_to_attrs(self):
        da = xr.DataArray([1000.0], attrs={"units": "m"}, name="dist")
        # source dict doesn't have "dist", should fall back to attrs "m"
        result = convert_dataarray(da, target_units="km", source_units={"other": "ft"})
        np.testing.assert_allclose(result.values, [1.0])

    def test_dict_both_source_and_target(self):
        da = xr.DataArray([1000.0], attrs={"units": "cm"}, name="dist")
        result = convert_dataarray(
            da,
            target_units={"dist": "km"},
            source_units={"dist": "m"},
        )
        np.testing.assert_allclose(result.values, [1.0])


# ---- convert_dataset ----


class TestConvertDataset:
    def test_convert_matching_variable(self):
        ds = xr.Dataset({
            "temp": xr.DataArray([273.15, 300.0], attrs={"units": "K"}),
            "wind": xr.DataArray([10.0, 20.0], attrs={"units": "m/s"}),
        })
        result = convert_dataset(ds, target_units="degC", source_units="K")
        np.testing.assert_allclose(result["temp"].values, [0.0, 26.85])
        # wind should be unchanged
        np.testing.assert_array_equal(result["wind"].values, [10.0, 20.0])

    def test_source_none_converts_compatible_vars(self):
        ds = xr.Dataset({
            "dist_m": xr.DataArray([1000.0], attrs={"units": "m"}),
            "dist_km": xr.DataArray([5.0], attrs={"units": "km"}),
            "temp": xr.DataArray([300.0], attrs={"units": "K"}),
        })
        result = convert_dataset(ds, target_units="km", source_units=None)
        # Both distance vars should be converted
        np.testing.assert_allclose(result["dist_m"].values, [1.0])
        np.testing.assert_allclose(result["dist_km"].values, [5.0])
        # temp should be unchanged
        np.testing.assert_array_equal(result["temp"].values, [300.0])

    def test_skips_vars_without_units_attr(self):
        ds = xr.Dataset({
            "with_units": xr.DataArray([1000.0], attrs={"units": "m"}),
            "no_units": xr.DataArray([1.0]),
        })
        result = convert_dataset(ds, target_units="km", source_units="m")
        np.testing.assert_allclose(result["with_units"].values, [1.0])
        np.testing.assert_array_equal(result["no_units"].values, [1.0])

    def test_source_units_filters_variables(self):
        ds = xr.Dataset({
            "dist_m": xr.DataArray([1000.0], attrs={"units": "m"}),
            "dist_km": xr.DataArray([5.0], attrs={"units": "km"}),
        })
        # Only convert variables that have "m" as their current units
        result = convert_dataset(ds, target_units="km", source_units="m")
        np.testing.assert_allclose(result["dist_m"].values, [1.0])
        # dist_km should be unchanged because its units are "km" not "m"
        np.testing.assert_array_equal(result["dist_km"].values, [5.0])


    def test_no_conversion_returns_original_dataset(self):
        ds = xr.Dataset({
            "temp": xr.DataArray([273.15], attrs={"units": "K"}),
        })
        result = convert_dataset(ds, target_units="km", source_units=None)
        assert result is ds

    def test_updates_units_attr_on_converted_vars(self):
        ds = xr.Dataset({
            "dist": xr.DataArray([1000.0], attrs={"units": "m"}),
        })
        result = convert_dataset(ds, target_units="km", source_units="m")
        assert result["dist"].attrs["units"] == "km"

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            convert_dataset("not a dataset", target_units="km", source_units="m")

    def test_does_not_mutate_input(self):
        ds = xr.Dataset({
            "dist": xr.DataArray([1000.0], attrs={"units": "m"}),
        })
        original = ds["dist"].values.copy()
        convert_dataset(ds, target_units="km", source_units="m")
        np.testing.assert_array_equal(ds["dist"].values, original)
        assert ds["dist"].attrs["units"] == "m"

    def test_dict_target_units_per_variable(self):
        ds = xr.Dataset({
            "temp": xr.DataArray([273.15], attrs={"units": "K"}),
            "dist": xr.DataArray([1000.0], attrs={"units": "m"}),
            "wind": xr.DataArray([10.0], attrs={"units": "m/s"}),
        })
        result = convert_dataset(
            ds,
            target_units={"temp": "degC", "dist": "km"},
        )
        np.testing.assert_allclose(result["temp"].values, [0.0])
        np.testing.assert_allclose(result["dist"].values, [1.0])
        # wind not in dict -> unchanged
        np.testing.assert_array_equal(result["wind"].values, [10.0])

    def test_dict_source_units_overrides_attrs(self):
        ds = xr.Dataset({
            "dist": xr.DataArray([1.0], attrs={"units": "km"}),
        })
        # attrs say "km" but source dict overrides to "m"
        result = convert_dataset(
            ds,
            target_units="km",
            source_units={"dist": "m"},
        )
        np.testing.assert_allclose(result["dist"].values, [0.001])

    def test_dict_source_units_falls_back_to_attrs(self):
        ds = xr.Dataset({
            "dist": xr.DataArray([1000.0], attrs={"units": "m"}),
            "temp": xr.DataArray([273.15], attrs={"units": "K"}),
        })
        # Only override source for temp, dist falls back to attrs
        result = convert_dataset(
            ds,
            target_units={"dist": "km", "temp": "degC"},
            source_units={"temp": "K"},
        )
        np.testing.assert_allclose(result["dist"].values, [1.0])
        np.testing.assert_allclose(result["temp"].values, [0.0])

    def test_dict_both_source_and_target(self):
        ds = xr.Dataset({
            "temp": xr.DataArray([273.15, 300.0], attrs={"units": "K"}),
            "dist": xr.DataArray([1000.0, 2000.0], attrs={"units": "m"}),
        })
        result = convert_dataset(
            ds,
            target_units={"temp": "degC", "dist": "km"},
            source_units={"temp": "K", "dist": "m"},
        )
        np.testing.assert_allclose(result["temp"].values, [0.0, 26.85])
        np.testing.assert_allclose(result["dist"].values, [1.0, 2.0])

    def test_dict_target_var_not_in_dataset_ignored(self):
        ds = xr.Dataset({
            "dist": xr.DataArray([1000.0], attrs={"units": "m"}),
        })
        # Dict references a var that doesn't exist -> no error, just ignored
        result = convert_dataset(
            ds,
            target_units={"dist": "km", "nonexistent": "degC"},
            source_units=None,
        )
        np.testing.assert_allclose(result["dist"].values, [1.0])


# ---- convert_units (dispatcher) ----


class TestConvertUnits:
    def test_dispatches_numpy_array(self):
        data = np.array([1000.0, 2000.0])
        result = convert_units(data, target_units="km", source_units="m")
        np.testing.assert_allclose(result, [1.0, 2.0])

    def test_dispatches_dataarray(self):
        da = xr.DataArray([1000.0], attrs={"units": "m"})
        result = convert_units(da, target_units="km", source_units=None)
        assert isinstance(result, xr.DataArray)
        np.testing.assert_allclose(result.values, [1.0])

    def test_dispatches_dataset(self):
        ds = xr.Dataset({
            "dist": xr.DataArray([1000.0], attrs={"units": "m"}),
        })
        result = convert_units(ds, target_units="km", source_units="m")
        assert isinstance(result, xr.Dataset)
        np.testing.assert_allclose(result["dist"].values, [1.0])

    def test_source_units_optional_for_dataarray(self):
        da = xr.DataArray([1000.0], attrs={"units": "m"})
        result = convert_units(da, target_units="km", source_units=None)
        np.testing.assert_allclose(result.values, [1.0])

    def test_source_units_optional_for_dataset(self):
        ds = xr.Dataset({
            "dist": xr.DataArray([1000.0], attrs={"units": "m"}),
        })
        result = convert_units(ds, target_units="km", source_units=None)
        np.testing.assert_allclose(result["dist"].values, [1.0])

    def test_plain_list_treated_as_array(self):
        # Lists/tuples should be handled like arrays
        data = [1000.0, 2000.0]
        result = convert_units(data, target_units="km", source_units="m")
        np.testing.assert_allclose(result, [1.0, 2.0])

    def test_scalar_conversion(self):
        result = convert_units(1000.0, target_units="km", source_units="m")
        assert abs(result - 1.0) < 1e-10

    def test_dispatches_dataset_with_dict(self):
        ds = xr.Dataset({
            "temp": xr.DataArray([273.15], attrs={"units": "K"}),
            "dist": xr.DataArray([1000.0], attrs={"units": "m"}),
        })
        result = convert_units(
            ds,
            target_units={"temp": "degC", "dist": "km"},
        )
        assert isinstance(result, xr.Dataset)
        np.testing.assert_allclose(result["temp"].values, [0.0])
        np.testing.assert_allclose(result["dist"].values, [1.0])

    def test_dispatches_dataarray_with_dict(self):
        da = xr.DataArray([1000.0], attrs={"units": "m"}, name="dist")
        result = convert_units(da, target_units={"dist": "km"})
        assert isinstance(result, xr.DataArray)
        np.testing.assert_allclose(result.values, [1.0])


# ---- Unit type variants (str, pint.Unit, Units) ----


class TestUnitTypes:
    """Test that str, pint.Unit, and Units objects are all accepted."""

    # -- are_equal --

    def test_are_equal_pint_units(self):
        assert are_equal(ureg.meter, ureg.meter)

    def test_are_equal_units_objects(self):
        assert are_equal(Units.from_any("m"), Units.from_any("m"))

    def test_are_equal_mixed_str_and_pint(self):
        assert are_equal("m", ureg.meter)

    def test_are_equal_mixed_str_and_units(self):
        assert are_equal("m/s", Units.from_any("m/s"))

    # -- are_compatible --

    def test_are_compatible_pint_units(self):
        assert are_compatible(ureg.meter, ureg.kilometer)

    def test_are_compatible_units_objects(self):
        assert are_compatible(Units.from_any("m"), Units.from_any("km"))

    def test_are_compatible_mixed(self):
        assert are_compatible("m", ureg.kilometer)

    def test_are_compatible_incompatible_pint(self):
        assert not are_compatible(ureg.meter, ureg.kelvin)

    # -- convert_array with pint.Unit --

    def test_convert_array_pint_source(self):
        data = np.array([1000.0])
        result = convert_array(data, target_units="km", source_units=ureg.meter)
        np.testing.assert_allclose(result, [1.0])

    def test_convert_array_pint_target(self):
        data = np.array([1000.0])
        result = convert_array(data, target_units=ureg.kilometer, source_units="m")
        np.testing.assert_allclose(result, [1.0])

    def test_convert_array_pint_both(self):
        data = np.array([1000.0])
        result = convert_array(data, target_units=ureg.kilometer, source_units=ureg.meter)
        np.testing.assert_allclose(result, [1.0])

    # -- convert_array with Units objects --

    def test_convert_array_units_source(self):
        data = np.array([1000.0])
        result = convert_array(data, target_units="km", source_units=Units.from_any("m"))
        np.testing.assert_allclose(result, [1.0])

    def test_convert_array_units_target(self):
        data = np.array([1000.0])
        result = convert_array(data, target_units=Units.from_any("km"), source_units="m")
        np.testing.assert_allclose(result, [1.0])

    def test_convert_array_units_both(self):
        data = np.array([1000.0])
        result = convert_array(
            data,
            target_units=Units.from_any("km"),
            source_units=Units.from_any("m"),
        )
        np.testing.assert_allclose(result, [1.0])

    # -- convert_dataarray with pint.Unit --

    def test_convert_dataarray_pint_target(self):
        da = xr.DataArray([1000.0], attrs={"units": "m"})
        result = convert_dataarray(da, target_units=ureg.kilometer)
        np.testing.assert_allclose(result.values, [1.0])

    def test_convert_dataarray_pint_source(self):
        da = xr.DataArray([1000.0], attrs={"units": "m"})
        result = convert_dataarray(da, target_units="km", source_units=ureg.meter)
        np.testing.assert_allclose(result.values, [1.0])

    # -- convert_dataarray with Units objects --

    def test_convert_dataarray_units_target(self):
        da = xr.DataArray([1000.0], attrs={"units": "m"})
        result = convert_dataarray(da, target_units=Units.from_any("km"))
        np.testing.assert_allclose(result.values, [1.0])

    # -- convert_dataset with pint.Unit --

    def test_convert_dataset_pint_target(self):
        ds = xr.Dataset({
            "dist": xr.DataArray([1000.0], attrs={"units": "m"}),
        })
        result = convert_dataset(ds, target_units=ureg.kilometer, source_units=ureg.meter)
        np.testing.assert_allclose(result["dist"].values, [1.0])

    # -- convert_dataset with dict containing pint.Unit values --

    def test_convert_dataset_dict_pint_values(self):
        ds = xr.Dataset({
            "dist": xr.DataArray([1000.0], attrs={"units": "m"}),
            "temp": xr.DataArray([273.15], attrs={"units": "K"}),
        })
        result = convert_dataset(
            ds,
            target_units={"dist": ureg.kilometer, "temp": ureg.degC},
        )
        np.testing.assert_allclose(result["dist"].values, [1.0])
        np.testing.assert_allclose(result["temp"].values, [0.0])

    # -- convert_dataset with dict containing Units values --

    def test_convert_dataset_dict_units_values(self):
        ds = xr.Dataset({
            "dist": xr.DataArray([1000.0], attrs={"units": "m"}),
        })
        result = convert_dataset(
            ds,
            target_units={"dist": Units.from_any("km")},
        )
        np.testing.assert_allclose(result["dist"].values, [1.0])

    # -- convert_units dispatcher with mixed types --

    def test_convert_units_pint_units(self):
        data = np.array([1000.0])
        result = convert_units(data, target_units=ureg.kilometer, source_units=ureg.meter)
        np.testing.assert_allclose(result, [1.0])

    def test_convert_units_units_objects(self):
        data = np.array([1000.0])
        result = convert_units(
            data,
            target_units=Units.from_any("km"),
            source_units=Units.from_any("m"),
        )
        np.testing.assert_allclose(result, [1.0])
