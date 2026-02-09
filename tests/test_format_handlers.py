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

from earthkit.utils.decorators import format_handler

# from earthkit.data import from_source, from_object


TEST_NUMPY = np.array([1, 2, 3])
TEST_DATAARRAY = xr.DataArray(TEST_NUMPY, name="test", dims=["x"], coords={"x": [0, 1, 2]})
TEST_DATASET = xr.Dataset({"test": TEST_DATAARRAY})


@pytest.mark.parametrize(
    "in_data",
    [
        TEST_NUMPY,
        TEST_DATAARRAY,
        TEST_DATASET,
    ],
)
def test_format_handler_numpy(in_data):
    @format_handler()
    def _numpy_handler(data: np.ndarray):
        return data

    assert isinstance(_numpy_handler(in_data), np.ndarray)


@pytest.mark.parametrize(
    "in_data",
    [
        TEST_NUMPY,
        TEST_DATAARRAY,
        TEST_DATASET,
    ],
)
def test_format_handler_xarray(in_data):
    @format_handler()
    def _xarray_handler(data: xr.DataArray):
        return data

    assert isinstance(_xarray_handler(in_data), xr.DataArray)


@pytest.mark.parametrize(
    "in_data",
    [
        TEST_NUMPY,
        TEST_DATAARRAY,
        TEST_DATASET,
    ],
)
def test_format_handler_multiple_accepted(in_data):
    """All in_data types are accepted, therefore no tranformation should be attempted."""

    @format_handler()
    def _xarray_handler(data: xr.DataArray | np.ndarray | xr.Dataset):
        return data

    assert isinstance(_xarray_handler(in_data), type(in_data))


@pytest.mark.parametrize(
    "in_data",
    [
        TEST_NUMPY,
        TEST_DATAARRAY,
        TEST_DATASET,
    ],
)
@pytest.mark.parametrize(
    "param",
    [
        1,
        "2",
        # 3.  # TODO: Not yet implemented
    ],
)
def test_format_handler_multiple_args_kwargs(in_data, param):
    """All in_data types are accepted, therefore no tranformation should be attempted."""

    @format_handler()
    def _xarray_handler(
        data1: xr.DataArray,
        data2: np.ndarray,
        param1: int,
        param2: str,
    ):
        return data1, data2, param1, param2

    out_data1, out_data2, _out_param1, _out_param2 = _xarray_handler(in_data, in_data, param, param)
    assert isinstance(out_data1, xr.DataArray)
    assert isinstance(out_data2, np.ndarray)

    # TODO: The following are not yet implemented.
    # assert isinstance(out_param1, int)
    # assert isinstance(out_param2, str)



def test_format_handler_convert_types():
    """Only convert data if it is an xarray.DataArray or xarray.Dataset."""

    @format_handler(convert_types={"data": (xr.DataArray, xr.Dataset)})
    def _xarray_handler(data: xr.DataArray):
        return data

    assert isinstance(_xarray_handler(TEST_DATAARRAY), xr.DataArray)
    assert isinstance(_xarray_handler(TEST_DATASET), xr.DataArray)
    assert isinstance(_xarray_handler(TEST_NUMPY), np.ndarray)


@pytest.mark.parametrize(
    "in_data",
    [
        TEST_NUMPY,
        TEST_DATAARRAY,
        TEST_DATASET,
    ],
)
def test_format_handler_with_kwarg_types(in_data):
    """All in_data types are accepted, therefore no tranformation should be attempted."""

    @format_handler(kwarg_types={"data": xr.DataArray})
    def _xarray_handler(data):
        return data

    assert isinstance(_xarray_handler(in_data), xr.DataArray)


def test_format_handler_no_earthkit_data(caplog, monkeypatch):
    import sys
    import types

    # Force earthkit.data to appear uninstalled by providing a dummy earthkit package
    fake_earthkit = types.ModuleType("earthkit")

    # Remove any real modules that may already be loaded
    monkeypatch.delitem(sys.modules, "earthkit.data", raising=False)
    monkeypatch.delitem(sys.modules, "earthkit.data.translators", raising=False)

    # Insert dummy earthkit without a data submodule
    monkeypatch.setitem(sys.modules, "earthkit", fake_earthkit)

    @format_handler()
    def _handler(data: xr.DataArray):
        return data

    _handler(TEST_NUMPY)
    # Check that a warning/error was logged when incompatible type passed
    assert any(
        "earthkit.data is not available" in record.getMessage()
        or "input object type does not match the expected function type" in record.getMessage()
        for record in caplog.records
    )
