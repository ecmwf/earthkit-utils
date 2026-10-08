#!/usr/bin/env python3

# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Tests of the convert_units contract.

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

from earthkit.utils.units import Units, convert_units
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
    result = convert_units(np.array(values), target_units, source_units, errors=errors)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("make_units", [str, ureg.Unit, Units.from_any], ids=["str", "pint", "Units"])
def test_array_unit_types(make_units):
    result = convert_units(np.array([1000.0]), make_units("km"), make_units("m"))
    np.testing.assert_allclose(result, [1.0])


def test_array_keeps_shape_and_nan():
    data = np.array([[1000.0, np.nan, 3000.0], [4000.0, 5000.0, 6000.0]])
    result = convert_units(data, "km", "m")
    np.testing.assert_allclose(result, [[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])


@pytest.mark.parametrize("data", [np.float64(1000.0), np.array(1000.0)])
def test_array_scalar(data):
    assert convert_units(data, "km", "m") == pytest.approx(1.0)


@pytest.mark.parametrize("dtype, expected", [("float32", "float32"), ("float64", "float64"), ("int32", "float64")])
def test_array_dtype(dtype, expected):
    assert convert_units(np.ones(2, dtype=dtype), "km", "m").dtype == expected


def test_array_input_not_modified():
    data = np.array([1000.0, 2000.0])
    convert_units(data, "km", "m")
    np.testing.assert_array_equal(data, [1000.0, 2000.0])


@pytest.mark.parametrize("units", [("m", "m"), ("m/s", "m s-1"), ("dBZ", "dBZ"), (None, None)])
@pytest.mark.usefixtures("no_warnings")
def test_array_equal_units_returns_input(units):
    data = np.array([1.0])
    assert convert_units(data, *units, errors="raise") is data


# ---- arrays: not converting ----


@pytest.mark.parametrize("source_units, target_units", [case[:2] for case in ARRAY_NOT_CONVERTIBLE])
def test_array_not_convertible_ignore(source_units, target_units, caplog):
    data = np.array([1.0, 2.0])
    with caplog.at_level(logging.WARNING):
        assert convert_units(data, target_units, source_units) is data
    assert "Cannot convert" in caplog.text
    np.testing.assert_array_equal(data, [1.0, 2.0])


@pytest.mark.parametrize("source_units, target_units, error", ARRAY_NOT_CONVERTIBLE)
def test_array_not_convertible_raise(source_units, target_units, error):
    with pytest.raises(error):
        convert_units(np.array([1.0, 2.0]), target_units, source_units, errors="raise")


@pytest.mark.parametrize("errors", ERRORS)
def test_array_dict_units_not_supported(errors):
    with pytest.raises(ValueError, match="Unsupported type for units"):
        convert_units(np.array([1.0]), {"dist": "km"}, "m", errors=errors)


# ---- unsupported input ----


@pytest.mark.parametrize(
    "data", [[1000.0], 1000.0, SimpleFieldList(np.array([1.0]))], ids=["list", "float", "fieldlist"]
)
def test_unsupported_data_type(data):
    with pytest.raises(TypeError, match="No dispatcher matched"):
        convert_units(data, "km", "m")


@pytest.mark.parametrize(
    "data",
    [np.array([1.0]), dataarray(), xr.Dataset({"dist": dataarray()}), xr.Dataset()],
    ids=["array", "dataarray", "dataset", "empty-dataset"],
)
def test_invalid_errors(data):
    with pytest.raises(ValueError, match="errors must be"):
        convert_units(data, "km", "m", errors="bogus")


# ---- xarray.DataArray: converting ----


@pytest.mark.parametrize("errors", ERRORS)
@pytest.mark.parametrize("source_units, target_units, values, expected", CONVERTIBLE)
@pytest.mark.usefixtures("no_warnings")
def test_dataarray_converts(source_units, target_units, values, expected, errors):
    da = dataarray(values, source_units, long_name="variable")
    result = convert_units(da, target_units, errors=errors)

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
    assert convert_units(dataarray(), target_units).attrs["units"] == expected_attr


@pytest.mark.usefixtures("no_warnings")
def test_dataarray_equal_units_only_relabels():
    result = convert_units(dataarray(units="m s-1"), "m/s", errors="raise")
    np.testing.assert_array_equal(result.values, [1000.0, 2000.0])
    assert result.attrs["units"] == "m/s"


def test_dataarray_multidimensional():
    da = xr.DataArray(np.full((2, 3), 1000.0), dims=("y", "x"), attrs={"units": "m"})
    result = convert_units(da, "km")
    assert result.dims == ("y", "x")
    np.testing.assert_allclose(result.values, np.ones((2, 3)))


def test_dataarray_input_not_modified():
    da = dataarray()
    convert_units(da, "km")
    xr.testing.assert_identical(da, dataarray())


@pytest.mark.parametrize("source_units", ["m", ureg.meter, {"dist": "m"}])
def test_dataarray_source_units_override_attrs(source_units):
    result = convert_units(dataarray(units="km"), "km", source_units)
    np.testing.assert_allclose(result.values, [1.0, 2.0])


def test_dataarray_source_units_dict_falls_back_to_attrs():
    result = convert_units(dataarray(), "km", {"other": "cm"})
    np.testing.assert_allclose(result.values, [1.0, 2.0])


def test_dataarray_source_units_without_units_attr():
    result = convert_units(dataarray(units=None), "km", "m")
    np.testing.assert_allclose(result.values, [1.0, 2.0])
    assert result.attrs["units"] == "km"


def test_dataarray_target_units_dict():
    result = convert_units(dataarray(), {"dist": "km", "other": "K"})
    np.testing.assert_allclose(result.values, [1.0, 2.0])
    assert result.attrs["units"] == "km"


@pytest.mark.parametrize("errors", ERRORS)
@pytest.mark.parametrize("target_units", [None, {"other": "km"}])
@pytest.mark.usefixtures("no_warnings")
def test_dataarray_no_target_units_returns_input(target_units, errors):
    da = dataarray()
    assert convert_units(da, target_units, errors=errors) is da


# ---- xarray.DataArray: not converting ----


@pytest.mark.parametrize("source_units, target_units", [case[:2] for case in NOT_CONVERTIBLE])
def test_dataarray_not_convertible_ignore(source_units, target_units, caplog):
    da = dataarray(units=source_units)
    with caplog.at_level(logging.WARNING):
        assert convert_units(da, target_units) is da
    assert "Cannot convert units of 'dist'" in caplog.text
    assert da.attrs["units"] == source_units


@pytest.mark.parametrize("source_units, target_units, error", NOT_CONVERTIBLE)
def test_dataarray_not_convertible_raise(source_units, target_units, error):
    with pytest.raises(error):
        convert_units(dataarray(units=source_units), target_units, errors="raise")


def test_dataarray_no_source_units(caplog):
    da = dataarray(units=None)

    with caplog.at_level(logging.WARNING):
        assert convert_units(da, "km") is da
    assert "No source units found for 'dist'" in caplog.text

    with pytest.raises(ValueError, match="No source units found for 'dist'"):
        convert_units(da, "km", errors="raise")


# ---- xarray.Dataset ----


@pytest.mark.parametrize("errors", ERRORS)
@pytest.mark.usefixtures("no_warnings")
def test_dataset_target_units_dict(ds, errors):
    result = convert_units(ds, {"dist": "km", "temp": "degC"}, errors=errors)

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
    convert_units(ds, {"dist": "km", "temp": "degC"})
    xr.testing.assert_identical(ds, original)


@pytest.mark.parametrize("errors", ERRORS)
@pytest.mark.parametrize("target_units", [None, {"other": "km"}])
@pytest.mark.usefixtures("no_warnings")
def test_dataset_nothing_requested(ds, target_units, errors):
    xr.testing.assert_identical(convert_units(ds, target_units, errors=errors), ds)


def test_dataset_single_target_units_ignore(ds, caplog):
    with caplog.at_level(logging.WARNING):
        result = convert_units(ds, "km")

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
        convert_units(ds, target_units, errors="raise")


def test_dataset_source_units_dict(ds):
    result = convert_units(ds, {"dist": "km", "flag": "%"}, {"flag": "dimensionless"})
    np.testing.assert_allclose(result["dist"].values, [1.0, 2.0])
    np.testing.assert_allclose(result["flag"].values, [0.0, 100.0])
    assert result["flag"].attrs["units"] == "%"


def test_dataset_single_source_units_applies_to_all_variables(ds):
    result = convert_units(ds, {"dist": "km", "flag": "km"}, "m")
    np.testing.assert_allclose(result["dist"].values, [1.0, 2.0])
    np.testing.assert_allclose(result["flag"].values, [0.0, 0.001])


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
    result = convert_units(_uncomputable(), "km")

    assert isinstance(result.data, dask_array.Array)
    assert result.dtype == np.float32
    assert result.attrs["units"] == "km"


def test_dask_converts():
    pytest.importorskip("dask")
    result = convert_units(dataarray().chunk(), "km")
    np.testing.assert_allclose(result.values, [1.0, 2.0])


def test_dask_not_convertible_ignore():
    da = _uncomputable()
    assert convert_units(da, "K") is da


def test_dask_not_convertible_raise_without_computing():
    with pytest.raises(ValueError, match="DimensionalityError"):
        convert_units(_uncomputable(), "K", errors="raise")
