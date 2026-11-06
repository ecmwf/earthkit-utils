#!/usr/bin/env python3

# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Tests of the convert contract.

- convertible data is converted, with no warning, whatever ``errors`` is
- equal units are a no-op returning the input itself
- data that cannot be converted is returned unchanged with a warning when
  ``errors="ignore"``, and raises when ``errors="raise"``
- xarray objects get their source units from the ``units`` attribute, which
  is updated only when the data is converted
- the input is never modified
"""

import logging

import numpy as np
import pint
import pytest
import xarray as xr
from earthkit.data import SimpleFieldList

from earthkit.utils.units import Units, are_compatible, convert
from earthkit.utils.units.units import get_registry

ureg = get_registry()

ERRORS = ["ignore", "raise"]

# (source_units, target_units, values, expected)
CONVERTIBLE = [
    ("m", "km", [1000.0, 2000.0], [1.0, 2.0]),
    ("degC", "degF", [0.0, 100.0], [32.0, 212.0]),
    ("K", "degC", [273.15, 283.15], [0.0, 10.0]),
    ("m/s", "km/h", [10.0], [36.0]),
    ("m s-1", "km h-1", [10.0], [36.0]),
    ("g m**-2", "kg m**-2", [1000.0], [1.0]),
    ("dimensionless", "%", [0.5], [50.0]),
]

# (source_units, target_units, error raised with errors="raise")
NOT_CONVERTIBLE = [
    ("m", "K", pint.DimensionalityError),
    ("m/s", "m", pint.DimensionalityError),
    ("code table", "%", ValueError),
    ("m", "dBZ", ValueError),
    ("gpm", "dBZ", ValueError),
]
# None source units mean dimensionless for arrays, but use the units attribute for xarray
ARRAY_NOT_CONVERTIBLE = [*NOT_CONVERTIBLE, (None, "m", pint.DimensionalityError)]


def dataarray(values=(1000.0, 2000.0), units="m", name="dist", **attrs):
    if units is not None:
        attrs["units"] = units
    return xr.DataArray(np.array(values), dims="x", coords={"x": np.arange(len(values))}, name=name, attrs=attrs)


@pytest.fixture
def ds():
    return xr.Dataset(
        {
            "dist": dataarray([1000.0, 2000.0], "m"),
            "temp": dataarray([273.15, 283.15], "K", name="temp", long_name="temperature"),
            "flag": dataarray([0.0, 1.0], None, name="flag"),
        },
        attrs={"title": "test"},
    )


@pytest.fixture
def no_warnings(caplog):
    with caplog.at_level(logging.WARNING):
        yield
    # at teardown, caplog.records only holds the records logged during teardown
    assert not caplog.get_records("call"), caplog.text


# ---- arrays: converting ----


@pytest.mark.parametrize("errors", ERRORS)
@pytest.mark.parametrize("source_units, target_units, values, expected", CONVERTIBLE)
@pytest.mark.usefixtures("no_warnings")
def test_array_converts(source_units, target_units, values, expected, errors):
    result = convert(np.array(values), target_units, source_units, errors=errors)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("make_units", [str, ureg.Unit, Units.from_any], ids=["str", "pint", "Units"])
def test_array_unit_types(make_units):
    result = convert(np.array([1000.0]), make_units("km"), make_units("m"))
    np.testing.assert_allclose(result, [1.0])


def test_array_keeps_shape_and_nan():
    data = np.array([[1000.0, np.nan, 3000.0], [4000.0, 5000.0, 6000.0]])
    result = convert(data, "km", "m")
    np.testing.assert_allclose(result, [[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])


@pytest.mark.parametrize("data", [np.float64(1000.0), np.array(1000.0)])
def test_array_scalar(data):
    assert convert(data, "km", "m") == pytest.approx(1.0)


@pytest.mark.parametrize("dtype, expected", [("float32", "float32"), ("float64", "float64"), ("int32", "float64")])
def test_array_dtype(dtype, expected):
    assert convert(np.ones(2, dtype=dtype), "km", "m").dtype == expected


def test_array_input_not_modified():
    data = np.array([1000.0, 2000.0])
    convert(data, "km", "m")
    np.testing.assert_array_equal(data, [1000.0, 2000.0])


@pytest.mark.parametrize("units", [("m", "m"), ("m/s", "m s-1"), ("dBZ", "dBZ")])
@pytest.mark.usefixtures("no_warnings")
def test_array_equal_units_returns_input(units):
    data = np.array([1.0])
    assert convert(data, *units, errors="raise") is data


# ---- arrays: not converting ----


@pytest.mark.parametrize("source_units, target_units", [case[:2] for case in ARRAY_NOT_CONVERTIBLE])
def test_array_not_convertible_ignore(source_units, target_units, caplog):
    data = np.array([1.0, 2.0])
    with caplog.at_level(logging.WARNING):
        assert convert(data, target_units, source_units, errors="ignore") is data
    assert "Cannot convert" in caplog.text
    np.testing.assert_array_equal(data, [1.0, 2.0])


@pytest.mark.parametrize("source_units, target_units, error", ARRAY_NOT_CONVERTIBLE)
def test_array_not_convertible_raise(source_units, target_units, error):
    with pytest.raises(error):
        convert(np.array([1.0, 2.0]), target_units, source_units, errors="raise")


@pytest.mark.parametrize("errors", ERRORS)
def test_array_dict_units_not_supported(errors):
    with pytest.raises(TypeError, match="Unsupported type for units"):
        convert(np.array([1.0]), {"dist": "km"}, "m", errors=errors)


# ---- unsupported input ----


@pytest.mark.parametrize(
    "data", [[1000.0], 1000.0, SimpleFieldList(np.array([1.0]))], ids=["list", "float", "fieldlist"]
)
def test_unsupported_data_type(data):
    with pytest.raises(TypeError, match="No dispatcher matched"):
        convert(data, "km", "m")


@pytest.mark.parametrize(
    "data",
    [np.array([1.0]), dataarray(), xr.Dataset({"dist": dataarray()}), xr.Dataset()],
    ids=["array", "dataarray", "dataset", "empty-dataset"],
)
def test_invalid_errors(data):
    with pytest.raises(ValueError, match="errors must be"):
        convert(data, "km", "m", errors="bogus")


# ---- xarray.DataArray: converting ----


@pytest.mark.parametrize("errors", ERRORS)
@pytest.mark.parametrize("source_units, target_units, values, expected", CONVERTIBLE)
@pytest.mark.usefixtures("no_warnings")
def test_dataarray_converts(source_units, target_units, values, expected, errors):
    da = dataarray(values, source_units, long_name="variable")
    result = convert(da, target_units, errors=errors)

    np.testing.assert_allclose(result.values, expected)
    assert result.attrs == {"units": target_units, "long_name": "variable"}
    assert result.name == da.name
    assert result.dims == da.dims
    xr.testing.assert_identical(result.coords.to_dataset(), da.coords.to_dataset())


@pytest.mark.parametrize(
    "target_units, expected_attr",
    [("km", "km"), ("kilometre", "kilometre"), (ureg.kilometer, "kilometer"), (Units.from_any("km"), "kilometer")],
)
def test_dataarray_units_attr(target_units, expected_attr):
    assert convert(dataarray(), target_units).attrs["units"] == expected_attr


@pytest.mark.usefixtures("no_warnings")
def test_dataarray_equal_units_only_relabels():
    result = convert(dataarray(units="m s-1"), "m/s", errors="raise")
    np.testing.assert_array_equal(result.values, [1000.0, 2000.0])
    assert result.attrs["units"] == "m/s"


def test_dataarray_multidimensional():
    da = xr.DataArray(np.full((2, 3), 1000.0), dims=("y", "x"), attrs={"units": "m"})
    result = convert(da, "km")
    assert result.dims == ("y", "x")
    np.testing.assert_allclose(result.values, np.ones((2, 3)))


def test_dataarray_input_not_modified():
    da = dataarray()
    convert(da, "km")
    xr.testing.assert_identical(da, dataarray())


@pytest.mark.parametrize("source_units", ["m", ureg.meter, {"dist": "m"}])
def test_dataarray_source_units_override_attrs(source_units):
    result = convert(dataarray(units="km"), "km", source_units)
    np.testing.assert_allclose(result.values, [1.0, 2.0])


def test_dataarray_source_units_dict_falls_back_to_attrs():
    result = convert(dataarray(), "km", {"other": "cm"})
    np.testing.assert_allclose(result.values, [1.0, 2.0])


def test_dataarray_source_units_without_units_attr():
    result = convert(dataarray(units=None), "km", "m")
    np.testing.assert_allclose(result.values, [1.0, 2.0])
    assert result.attrs["units"] == "km"


def test_dataarray_target_units_dict():
    result = convert(dataarray(), {"dist": "km", "other": "K"})
    np.testing.assert_allclose(result.values, [1.0, 2.0])
    assert result.attrs["units"] == "km"


@pytest.mark.parametrize("errors", ERRORS)
@pytest.mark.parametrize(
    "data",
    [np.array([1.0]), dataarray(), xr.Dataset({"dist": dataarray()})],
    ids=["array", "dataarray", "dataset"],
)
def test_none_target_units_raises(data, errors):
    with pytest.raises(TypeError, match="target_units must be provided"):
        convert(data, None, errors=errors)


@pytest.mark.parametrize("name", ["dist", None])
def test_dataarray_name_not_in_target_dict_raises_by_default(name):
    with pytest.raises(ValueError, match="No target units found"):
        convert(dataarray(name=name), {"other": "km"})


@pytest.mark.parametrize("name", ["dist", None])
def test_dataarray_name_not_in_target_dict_ignore(name, caplog):
    da = dataarray(name=name)
    with caplog.at_level(logging.WARNING):
        assert convert(da, {"other": "km"}, errors="ignore") is da
    assert "No target units found" in caplog.text


# ---- xarray.DataArray: not converting ----


@pytest.mark.parametrize("source_units, target_units", [case[:2] for case in NOT_CONVERTIBLE])
def test_dataarray_not_convertible_ignore(source_units, target_units, caplog):
    da = dataarray(units=source_units)
    with caplog.at_level(logging.WARNING):
        assert convert(da, target_units, errors="ignore") is da
    assert "Cannot convert units of 'dist'" in caplog.text
    assert da.attrs["units"] == source_units


@pytest.mark.parametrize("source_units, target_units, error", NOT_CONVERTIBLE)
def test_dataarray_not_convertible_raise(source_units, target_units, error):
    with pytest.raises(error):
        convert(dataarray(units=source_units), target_units, errors="raise")


def test_dataarray_no_source_units(caplog):
    da = dataarray(units=None)

    with caplog.at_level(logging.WARNING):
        assert convert(da, "km", errors="ignore") is da
    assert "No source units found for 'dist'" in caplog.text

    with pytest.raises(ValueError, match="No source units found for 'dist'"):
        convert(da, "km", errors="raise")


# ---- xarray.Dataset ----


@pytest.mark.parametrize("errors", ERRORS)
@pytest.mark.usefixtures("no_warnings")
def test_dataset_target_units_dict(ds, errors):
    result = convert(ds, {"dist": "km", "temp": "degC"}, errors=errors)

    np.testing.assert_allclose(result["dist"].values, [1.0, 2.0])
    np.testing.assert_allclose(result["temp"].values, [0.0, 10.0])
    assert result["dist"].attrs == {"units": "km"}
    assert result["temp"].attrs == {"units": "degC", "long_name": "temperature"}
    xr.testing.assert_identical(result["flag"], ds["flag"])
    assert list(result.data_vars) == list(ds.data_vars)
    xr.testing.assert_identical(result.coords.to_dataset(), ds.coords.to_dataset())
    assert result.attrs == {"title": "test"}


def test_dataset_input_not_modified(ds):
    original = ds.copy(deep=True)
    convert(ds, {"dist": "km", "temp": "degC"})
    xr.testing.assert_identical(ds, original)


@pytest.mark.parametrize("errors", ERRORS)
@pytest.mark.usefixtures("no_warnings")
def test_dataset_nothing_requested(ds, errors):
    xr.testing.assert_identical(convert(ds, {}, errors=errors), ds)


def test_dataset_single_target_units_ignore(ds, caplog):
    with caplog.at_level(logging.WARNING):
        result = convert(ds, "km", errors="ignore")

    np.testing.assert_allclose(result["dist"].values, [1.0, 2.0])
    xr.testing.assert_identical(result["temp"], ds["temp"])
    xr.testing.assert_identical(result["flag"], ds["flag"])
    assert "Cannot convert units of 'temp'" in caplog.text
    assert "No source units found for 'flag'" in caplog.text


@pytest.mark.parametrize(
    "target_units, error, match",
    [
        ("km", pint.DimensionalityError, "kelvin"),
        ({"dist": "km", "temp": "m"}, pint.DimensionalityError, "kelvin"),
        ({"flag": "km"}, ValueError, "No source units found for 'flag'"),
    ],
)
def test_dataset_raise(ds, target_units, error, match):
    with pytest.raises(error, match=match):
        convert(ds, target_units, errors="raise")


def test_dataset_source_units_dict(ds):
    result = convert(ds, {"dist": "km", "flag": "%"}, {"flag": "dimensionless"})
    np.testing.assert_allclose(result["dist"].values, [1.0, 2.0])
    np.testing.assert_allclose(result["flag"].values, [0.0, 100.0])
    assert result["flag"].attrs["units"] == "%"


def test_dataset_single_source_units_applies_to_all_variables(ds):
    result = convert(ds, {"dist": "km", "flag": "km"}, "m")
    np.testing.assert_allclose(result["dist"].values, [1.0, 2.0])
    np.testing.assert_allclose(result["flag"].values, [0.0, 0.001])


# ---- list target units ----

# (source_units, target_units, values, expected, chosen target units)
LIST_CONVERTIBLE = [
    ("m/s", ["km/h", "min"], [10.0], [36.0], "km/h"),
    ("h", ["km/h", "min"], [2.0], [120.0], "min"),
    ("h", ["min", "km/h"], [2.0], [120.0], "min"),
    ("K", ["km", "degC"], [273.15], [0.0], "degC"),
    ("km/h", ["m/s", "h"], [36.0], [10.0], "m/s"),
    ("m", ["km"], [1000.0], [1.0], "km"),
]

# target units that are always rejected, whatever errors is
AMBIGUOUS = [["m/s", "km/h"], ["K", "degC"], ["m", "m"], ["dBZ", "dBZ"], ["m", "s", "km"]]
UNSUPPORTED_IN_LIST = [["m", {"x": "km"}], ["m", 5], ["m", ["km"]]]


@pytest.mark.parametrize("errors", ERRORS)
@pytest.mark.parametrize("source_units, target_units, values, expected, chosen", LIST_CONVERTIBLE)
@pytest.mark.usefixtures("no_warnings")
def test_array_list_target_units(source_units, target_units, values, expected, chosen, errors):
    result = convert(np.array(values), target_units, source_units, errors=errors)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("make_units", [str, ureg.Unit, Units.from_any], ids=["str", "pint", "Units"])
def test_array_list_target_unit_types(make_units):
    result = convert(np.array([2.0]), [make_units("km/h"), make_units("min")], make_units("h"))
    np.testing.assert_allclose(result, [120.0])


@pytest.mark.parametrize("source_units, target_units", [("km/h", ["km h-1", "min"]), ("dBZ", ["m", "dBZ"])])
@pytest.mark.usefixtures("no_warnings")
def test_array_list_target_units_equal_returns_input(source_units, target_units):
    data = np.array([1.0])
    assert convert(data, target_units, source_units) is data


@pytest.mark.parametrize("target_units", [["km/h", "kg"], ["dBZ"], []])
def test_array_list_target_units_no_match_raises_by_default(target_units):
    with pytest.raises(ValueError, match="any of the target units"):
        convert(np.array([1.0]), target_units, "m")


def test_array_list_target_units_no_match_ignore(caplog):
    data = np.array([1.0])
    with caplog.at_level(logging.WARNING):
        assert convert(data, ["km/h", "kg"], "m", errors="ignore") is data
    assert "any of the target units" in caplog.text


@pytest.mark.parametrize("errors", ERRORS)
@pytest.mark.parametrize("target_units", AMBIGUOUS)
def test_array_list_target_units_ambiguous(target_units, errors):
    with pytest.raises(ValueError, match="Ambiguous target units"):
        convert(np.array([1.0]), target_units, "m", errors=errors)


@pytest.mark.parametrize("errors", ERRORS)
@pytest.mark.parametrize("target_units", UNSUPPORTED_IN_LIST)
def test_array_list_target_units_unsupported_type(target_units, errors):
    with pytest.raises(TypeError, match="Unsupported type for units"):
        convert(np.array([1.0]), target_units, "m", errors=errors)


def test_array_list_target_units_input_not_modified():
    data = np.array([1000.0, 2000.0])
    target_units = ["km", "s"]
    convert(data, target_units, "m")
    np.testing.assert_array_equal(data, [1000.0, 2000.0])
    assert target_units == ["km", "s"]


@pytest.mark.parametrize("errors", ERRORS)
@pytest.mark.parametrize("source_units, target_units, values, expected, chosen", LIST_CONVERTIBLE)
@pytest.mark.usefixtures("no_warnings")
def test_dataarray_list_target_units(source_units, target_units, values, expected, chosen, errors):
    da = dataarray(values, source_units, long_name="variable")
    result = convert(da, target_units, errors=errors)

    np.testing.assert_allclose(result.values, expected)
    assert result.attrs == {"units": chosen, "long_name": "variable"}
    assert result.name == da.name
    assert result.dims == da.dims


def test_dataarray_list_target_units_source_units_override():
    # the source units, not the units attribute, choose the target units
    result = convert(dataarray([36.0], units="h"), ["m/s", "h"], "km/h")
    np.testing.assert_allclose(result.values, [10.0])
    assert result.attrs["units"] == "m/s"


def test_dataarray_list_target_units_in_dict():
    result = convert(dataarray([2.0], units="h", name="t"), {"t": ["km/h", "min"]})
    np.testing.assert_allclose(result.values, [120.0])
    assert result.attrs["units"] == "min"


def test_dataarray_list_target_units_no_match_raises_by_default():
    with pytest.raises(ValueError, match="any of the target units"):
        convert(dataarray(), ["km/h", "kg"])


def test_dataarray_list_target_units_no_match_ignore(caplog):
    da = dataarray()
    with caplog.at_level(logging.WARNING):
        assert convert(da, ["km/h", "kg"], errors="ignore") is da
    assert "Cannot convert units of 'dist'" in caplog.text


@pytest.mark.parametrize("errors", ERRORS)
@pytest.mark.parametrize("target_units", AMBIGUOUS)
def test_dataarray_list_target_units_ambiguous(target_units, errors):
    with pytest.raises(ValueError, match="Ambiguous target units"):
        convert(dataarray(), target_units, errors=errors)


@pytest.mark.parametrize("errors", ERRORS)
@pytest.mark.parametrize("target_units", UNSUPPORTED_IN_LIST)
def test_dataarray_list_target_units_unsupported_type(target_units, errors):
    with pytest.raises(TypeError, match="Unsupported type for units"):
        convert(dataarray(), target_units, errors=errors)


def test_dataarray_list_target_units_input_not_modified():
    da = dataarray()
    convert(da, ["km", "s"])
    xr.testing.assert_identical(da, dataarray())


@pytest.mark.parametrize("errors", ERRORS)
@pytest.mark.usefixtures("no_warnings")
def test_dataset_list_target_units(ds, errors):
    # each variable gets the target units compatible with its own units
    result = convert(ds.drop_vars("flag"), ["km", "degC"], errors=errors)

    np.testing.assert_allclose(result["dist"].values, [1.0, 2.0])
    np.testing.assert_allclose(result["temp"].values, [0.0, 10.0])
    assert result["dist"].attrs == {"units": "km"}
    assert result["temp"].attrs == {"units": "degC", "long_name": "temperature"}
    assert result.attrs == {"title": "test"}


def test_dataset_list_target_units_single_source_units(ds):
    # a single source units applies to all variables, including those without a units attribute
    result = convert(ds, ["km", "s"], "m")
    for name in ds.data_vars:
        np.testing.assert_allclose(result[name].values, ds[name].values / 1000)
        assert result[name].attrs["units"] == "km"


def test_dataset_list_target_units_in_dict(ds):
    result = convert(ds, {"dist": ["km", "degC"], "temp": ["km", "degC"]})
    np.testing.assert_allclose(result["dist"].values, [1.0, 2.0])
    np.testing.assert_allclose(result["temp"].values, [0.0, 10.0])
    xr.testing.assert_identical(result["flag"], ds["flag"])


def test_dataset_list_target_units_mixed_dict_values(ds):
    result = convert(ds, {"dist": ["km", "s"], "temp": "degC"})
    np.testing.assert_allclose(result["dist"].values, [1.0, 2.0])
    np.testing.assert_allclose(result["temp"].values, [0.0, 10.0])


def test_dataset_list_target_units_no_match_raises_by_default(ds):
    with pytest.raises(ValueError, match="any of the target units"):
        convert(ds.drop_vars("flag"), ["km", "s"])


def test_dataset_list_target_units_no_match_ignore(ds, caplog):
    with caplog.at_level(logging.WARNING):
        result = convert(ds, ["km", "s"], errors="ignore")

    np.testing.assert_allclose(result["dist"].values, [1.0, 2.0])
    xr.testing.assert_identical(result["temp"], ds["temp"])
    xr.testing.assert_identical(result["flag"], ds["flag"])
    assert "Cannot convert units of 'temp'" in caplog.text
    assert "No source units found for 'flag'" in caplog.text


@pytest.mark.parametrize("errors", ERRORS)
@pytest.mark.parametrize("target_units", [["km", "m"], {"dist": ["km", "m"]}])
def test_dataset_list_target_units_ambiguous(ds, target_units, errors):
    with pytest.raises(ValueError, match="Ambiguous target units"):
        convert(ds, target_units, errors=errors)


def test_dataset_list_target_units_input_not_modified(ds):
    original = ds.copy(deep=True)
    convert(ds.drop_vars("flag"), ["km", "degC"])
    xr.testing.assert_identical(ds, original)


def test_dask_list_target_units_stays_lazy():
    dask_array = pytest.importorskip("dask.array")
    result = convert(_uncomputable(), ["km", "s"])

    assert isinstance(result.data, dask_array.Array)
    assert result.attrs["units"] == "km"


@pytest.mark.parametrize(
    "source_units, target_units",
    [(case[0], case[1]) for case in LIST_CONVERTIBLE] + [("m", ["km/h", "kg"]), ("dBZ", ["m", "dBZ"])],
)
def test_list_target_units_matches_are_compatible(source_units, target_units):
    # the chosen target units are the ones are_compatible accepts
    compatible = [unit for unit in target_units if are_compatible(source_units, unit)]
    try:
        result = convert(dataarray([1.0], units=source_units), target_units)
        assert compatible == [result.attrs["units"]]
    except ValueError:
        assert compatible == []


# ---- dict units on a Dataset with many variables ----


@pytest.fixture
def ds_many():
    return xr.Dataset(
        {
            "u": dataarray([10.0, 20.0], "m/s", name="u"),
            "t": dataarray([273.15, 283.15], "K", name="t", long_name="temperature"),
            "p": dataarray([100000.0, 50000.0], "Pa", name="p"),
            "z": dataarray([1000.0, 2000.0], "m", name="z"),
            "refl": dataarray([10.0, 20.0], "dBZ", name="refl"),
            "flag": dataarray([0.0, 1.0], None, name="flag"),
        },
        attrs={"title": "test"},
    )


def assert_converted(result, ds, expected):
    """Check the variables in ``expected`` (name -> (values, units attribute)) and that the others are unchanged."""
    assert list(result.data_vars) == list(ds.data_vars)
    assert result.attrs == ds.attrs
    xr.testing.assert_identical(result.coords.to_dataset(), ds.coords.to_dataset())
    for name in ds.data_vars:
        if name in expected:
            values, units = expected[name]
            np.testing.assert_allclose(result[name].values, values)
            assert result[name].attrs == {**ds[name].attrs, "units": units}
        else:
            xr.testing.assert_identical(result[name], ds[name])


# (target_units, source_units, expected: name -> (values, units attribute))
DICT_CASES = {
    "all-converted": (
        {"u": "km/h", "t": "degC", "p": "hPa", "z": "km"},
        None,
        {
            "u": ([36.0, 72.0], "km/h"),
            "t": ([0.0, 10.0], "degC"),
            "p": ([1000.0, 500.0], "hPa"),
            "z": ([1.0, 2.0], "km"),
        },
    ),
    "subset": (
        {"u": "km/h", "p": "hPa"},
        None,
        {"u": ([36.0, 72.0], "km/h"), "p": ([1000.0, 500.0], "hPa")},
    ),
    "mixed-unit-types": (
        {"u": ureg.Unit("km/h"), "t": Units.from_any("degC"), "p": ["hPa", "km"], "z": "km"},
        None,
        {
            "u": ([36.0, 72.0], str(ureg.Unit("km/h"))),
            "t": ([0.0, 10.0], str(Units.from_any("degC"))),
            "p": ([1000.0, 500.0], "hPa"),
            "z": ([1.0, 2.0], "km"),
        },
    ),
    "list-values": (
        {"u": ["km/h", "hPa"], "p": ["km/h", "hPa"], "t": ["km", "degC"]},
        None,
        {"u": ([36.0, 72.0], "km/h"), "p": ([1000.0, 500.0], "hPa"), "t": ([0.0, 10.0], "degC")},
    ),
    "list-target": (
        ["km/h", "hPa", "degC"],
        {"z": "km/h", "refl": "km/h", "flag": "km/h"},
        {
            "u": ([36.0, 72.0], "km/h"),
            "t": ([0.0, 10.0], "degC"),
            "p": ([1000.0, 500.0], "hPa"),
            "z": ([1000.0, 2000.0], "km/h"),
            "refl": ([10.0, 20.0], "km/h"),
            "flag": ([0.0, 1.0], "km/h"),
        },
    ),
    "source-dict-partial": (
        {"u": "m/s", "z": "km"},
        {"u": "km/h"},
        {"u": ([10 / 3.6, 20 / 3.6], "m/s"), "z": ([1.0, 2.0], "km")},
    ),
    "source-dict-for-unrequested-variable": (
        {"z": "km"},
        {"u": "km/h", "t": "degC"},
        {"z": ([1.0, 2.0], "km")},
    ),
    "source-dict-for-variable-without-units": (
        {"flag": "%", "z": "km"},
        {"flag": "dimensionless"},
        {"flag": ([0.0, 100.0], "%"), "z": ([1.0, 2.0], "km")},
    ),
    "single-source-overrides-requested": (
        {"u": "km", "z": "km"},
        "m",
        {"u": ([0.01, 0.02], "km"), "z": ([1.0, 2.0], "km")},
    ),
    "equal-units-only-relabel": (
        {"u": "m s-1", "refl": "dBZ"},
        None,
        {"u": ([10.0, 20.0], "m s-1"), "refl": ([10.0, 20.0], "dBZ")},
    ),
    "source-key-not-in-dataset": (
        {"u": "km/h"},
        {"nonexistent": "K"},
        {"u": ([36.0, 72.0], "km/h")},
    ),
    "none-value": (
        {"u": None, "z": "km"},
        None,
        {"z": ([1.0, 2.0], "km")},
    ),
    "empty-dict": ({}, None, {}),
}


@pytest.mark.parametrize("errors", ERRORS)
@pytest.mark.parametrize("target_units, source_units, expected", DICT_CASES.values(), ids=DICT_CASES.keys())
@pytest.mark.usefixtures("no_warnings")
def test_dataset_dict_units(ds_many, target_units, source_units, expected, errors):
    result = convert(ds_many, target_units, source_units, errors=errors)
    assert_converted(result, ds_many, expected)


def test_dataset_dict_units_input_not_modified(ds_many):
    original = ds_many.copy(deep=True)
    target_units = {"u": "km/h", "t": ["degC", "km"], "z": "km"}
    source_units = {"u": "m/s"}
    convert(ds_many, target_units, source_units)
    xr.testing.assert_identical(ds_many, original)
    assert target_units == {"u": "km/h", "t": ["degC", "km"], "z": "km"}
    assert source_units == {"u": "m/s"}


def test_dataset_single_target_units_with_source_dict(ds_many, caplog):
    # a single target applies to every variable: u (overridden to m) and z convert, the others cannot
    with caplog.at_level(logging.WARNING):
        result = convert(ds_many, "km", {"u": "m"}, errors="ignore")

    assert_converted(result, ds_many, {"u": ([0.01, 0.02], "km"), "z": ([1.0, 2.0], "km")})
    for name in ["t", "p", "refl"]:
        assert f"Cannot convert units of '{name}'" in caplog.text
    assert "No source units found for 'flag'" in caplog.text


# (target_units, source_units, error raised by default, match, the variable that cannot be converted)
DICT_NOT_CONVERTIBLE = {
    "incompatible": ({"u": "km/h", "t": "km", "z": "km"}, None, pint.DimensionalityError, "kelvin", "t"),
    "no-source-units": ({"u": "km/h", "flag": "km", "z": "km"}, None, ValueError, "'flag'", "flag"),
    "unrecognised": ({"u": "km/h", "refl": "mm/h", "z": "km"}, None, ValueError, "unrecognised", "refl"),
    "overridden-source-incompatible": (
        {"u": "km/h", "p": "hPa", "z": "km"},
        {"p": "K"},
        pint.DimensionalityError,
        "kelvin",
        "p",
    ),
    "no-list-match": ({"u": "km/h", "t": ["km", "hPa"], "z": "km"}, None, ValueError, "any of the target units", "t"),
    "key-not-in-dataset": (
        {"u": "km/h", "z": "km", "nonexistent": "degC"},
        None,
        ValueError,
        "not in the Dataset",
        "nonexistent",
    ),
}


@pytest.mark.parametrize(
    "target_units, source_units, error, match, failing", DICT_NOT_CONVERTIBLE.values(), ids=DICT_NOT_CONVERTIBLE.keys()
)
def test_dataset_dict_units_not_convertible_raises_by_default(
    ds_many, target_units, source_units, error, match, failing
):
    with pytest.raises(error, match=match):
        convert(ds_many, target_units, source_units)


@pytest.mark.parametrize(
    "target_units, source_units, error, match, failing", DICT_NOT_CONVERTIBLE.values(), ids=DICT_NOT_CONVERTIBLE.keys()
)
def test_dataset_dict_units_not_convertible_ignore(ds_many, target_units, source_units, error, match, failing, caplog):
    # the other requested variables are still converted
    with caplog.at_level(logging.WARNING):
        result = convert(ds_many, target_units, source_units, errors="ignore")

    assert_converted(result, ds_many, {"u": ([36.0, 72.0], "km/h"), "z": ([1.0, 2.0], "km")})
    assert f"'{failing}'" in caplog.text


@pytest.mark.parametrize("errors", ERRORS)
@pytest.mark.parametrize(
    "target_units, source_units, error, match",
    [
        ({"u": "km/h", "z": 5}, None, TypeError, "Unsupported type for units"),
        ({"u": "km/h", "z": {"nested": "km"}}, None, TypeError, "Unsupported type for units"),
        ({"u": "km/h", "z": "km"}, {"z": 5}, TypeError, "Unsupported type for units"),
        ({"u": "km/h", "z": ["km", "m"]}, None, ValueError, "Ambiguous target units"),
        ({"u": "km/h", "z": ["km", {"nested": "km"}]}, None, TypeError, "Unsupported type for units"),
    ],
    ids=["int-target", "nested-dict-target", "int-source", "ambiguous-list", "unsupported-in-list"],
)
def test_dataset_dict_units_invalid_always_raises(ds_many, target_units, source_units, error, match, errors):
    with pytest.raises(error, match=match):
        convert(ds_many, target_units, source_units, errors=errors)


@pytest.mark.parametrize("errors", ERRORS)
@pytest.mark.usefixtures("no_warnings")
def test_dataarray_dict_units_picks_own_name(errors):
    da = dataarray([10.0, 20.0], "m/s", name="u")
    result = convert(da, {"u": "km/h", "t": "degC", "z": "km"}, {"t": "K", "z": "m"}, errors=errors)
    np.testing.assert_allclose(result.values, [36.0, 72.0])
    assert result.attrs["units"] == "km/h"


# ---- are_compatible ----


@pytest.mark.parametrize(
    "unit_1, unit_2, expected",
    [
        ("m", "km", True),
        ("degC", "K", True),
        ("m s-1", "km/h", True),
        ("dimensionless", "%", True),
        ("m/s", "m s-1", True),
        ("dBZ", "dBZ", True),
        (None, None, True),
        ("m", "K", False),
        ("m", "dBZ", False),
        ("code table", "%", False),
        (None, "m", False),
    ],
)
def test_are_compatible(unit_1, unit_2, expected):
    assert are_compatible(unit_1, unit_2) is expected


@pytest.mark.parametrize("make_units", [str, ureg.Unit, Units.from_any], ids=["str", "pint", "Units"])
def test_are_compatible_unit_types(make_units):
    assert are_compatible(make_units("m"), make_units("km"))
    assert not are_compatible(make_units("m"), make_units("K"))


@pytest.mark.parametrize(
    "source_units, target_units",
    [case[:2] for case in CONVERTIBLE + ARRAY_NOT_CONVERTIBLE] + [("dBZ", "dBZ"), ("m/s", "m s-1")],
)
def test_are_compatible_matches_convert(source_units, target_units):
    try:
        convert(np.array([1.0]), target_units, source_units, errors="raise")
        converted = True
    except (ValueError, pint.DimensionalityError):
        converted = False
    assert are_compatible(source_units, target_units) is converted


# ---- dask ----


def _uncomputable(units="m"):
    """A dask-backed DataArray that raises if its data is ever computed."""
    dask = pytest.importorskip("dask")
    dask_array = pytest.importorskip("dask.array")

    def compute():
        raise RuntimeError("data was computed")

    data = dask_array.from_delayed(dask.delayed(compute)(), shape=(2,), dtype="float32")
    return xr.DataArray(data, dims="x", name="dist", attrs={"units": units})


def test_dask_stays_lazy():
    dask_array = pytest.importorskip("dask.array")
    result = convert(_uncomputable(), "km")

    assert isinstance(result.data, dask_array.Array)
    assert result.dtype == np.float32
    assert result.attrs["units"] == "km"


def test_dask_converts():
    pytest.importorskip("dask")
    result = convert(dataarray().chunk(), "km")
    np.testing.assert_allclose(result.values, [1.0, 2.0])


def test_dask_not_convertible_ignore():
    da = _uncomputable()
    assert convert(da, "K", errors="ignore") is da


def test_dask_not_convertible_raise_without_computing():
    with pytest.raises(ValueError, match="DimensionalityError"):
        convert(_uncomputable(), "K", errors="raise")
