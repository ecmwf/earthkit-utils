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

from earthkit.utils.parameters import PARAMS
from earthkit.utils.parameters._params import _PARAMS
from earthkit.utils.parameters._params import _ParamDB

REQUIRED_KEYS = {"variable", "units", "long_name", "standard_name", "cf_units"}
ALL_KEYS = REQUIRED_KEYS | {"paramId", "edition"}


def test_params_is_param_db():
    assert isinstance(PARAMS, _ParamDB)
    assert PARAMS.params is _PARAMS


@pytest.mark.parametrize("name", sorted(_PARAMS))
def test_param_item_keys(name):
    item = PARAMS.get(name)
    assert REQUIRED_KEYS <= set(item) <= ALL_KEYS


@pytest.mark.parametrize("name", sorted(_PARAMS))
def test_param_item_key_order(name):
    keys = list(_PARAMS[name])
    assert keys.index("long_name") < keys.index("standard_name")


@pytest.mark.parametrize("name", sorted(_PARAMS))
def test_param_item_values(name):
    item = PARAMS.get(name)
    assert isinstance(item["variable"], str) and item["variable"]
    assert isinstance(item["units"], str) and item["units"]
    assert isinstance(item["long_name"], str) and item["long_name"]
    # cf_units is defined if and only if standard_name is defined
    assert (item["standard_name"] is None) == (item["cf_units"] is None)
    if item["standard_name"] is not None:
        assert isinstance(item["standard_name"], str)
        assert isinstance(item["cf_units"], str)
    if "paramId" in item:
        assert isinstance(item["paramId"], int)
        assert "edition" in item


def test_get():
    item = PARAMS.get("2m_temperature")
    assert item["variable"] == "2t"
    assert item["units"] == "K"
    assert item["paramId"] == 167
    assert item["long_name"] == "2 metre temperature"
    assert item["standard_name"] == "air_temperature"


def test_get_unknown():
    with pytest.raises(KeyError):
        PARAMS.get("not_a_parameter")


def test_field_parameter_metadata():
    assert PARAMS.field_parameter_metadata("2m_temperature") == {
        "parameter.variable": "2t",
        "parameter.units": "K",
        "parameter.standard_name": "air_temperature",
    }


def test_field_parameter_metadata_no_standard_name():
    assert PARAMS.field_parameter_metadata("lcl_pressure") == {
        "parameter.variable": "p_lcl",
        "parameter.units": "Pa",
        "parameter.standard_name": None,
    }


def test_field_parameter_metadata_unknown():
    with pytest.raises(KeyError):
        PARAMS.field_parameter_metadata("not_a_parameter")


def test_xarray_variable_and_metadata_cf_units():
    variable, meta = PARAMS.xarray_variable_and_metadata("wind_speed")
    assert variable == "ws"
    assert meta == {
        "long_name": "Wind speed",
        "units": "m s-1",
        "standard_name": "wind_speed",
    }


def test_xarray_variable_and_metadata_fallback_units():
    # no cf_units: falls back to "units"
    variable, meta = PARAMS.xarray_variable_and_metadata("lcl_temperature")
    assert variable == "t_lcl"
    assert meta == {
        "long_name": "Lifting condensation level temperature",
        "units": "K",
        "standard_name": None,
    }


@pytest.mark.parametrize("name", sorted(_PARAMS))
def test_xarray_variable_and_metadata_all(name):
    item = PARAMS.get(name)
    variable, meta = PARAMS.xarray_variable_and_metadata(name)
    assert variable == item["variable"]
    assert meta["long_name"] == item["long_name"]
    assert meta["standard_name"] == item["standard_name"]
    assert meta["units"] == (item["cf_units"] or item["units"])


def test_xarray_variable_and_metadata_unknown():
    with pytest.raises(KeyError):
        PARAMS.xarray_variable_and_metadata("not_a_parameter")
