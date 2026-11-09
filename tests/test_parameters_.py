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
from earthkit.utils.parameters._params import _REQUIRED_KEYS

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
    # cf_units must be defined when standard_name is, but can also exist without it
    if item["standard_name"] is not None:
        assert isinstance(item["standard_name"], str)
        assert item["cf_units"] is not None
    if item["cf_units"] is not None:
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
        "long_name": PARAMS.get("lcl_temperature")["long_name"],
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


# --------------------------------------------------------------------------------------
# Caller-supplied definitions
# --------------------------------------------------------------------------------------


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


def test_required_keys():
    assert _REQUIRED_KEYS == ("variable", "units", "long_name", "standard_name", "cf_units")


def test_get_with_definitions_new_name():
    definitions = {"my_parameter": _definition()}
    assert PARAMS.get("my_parameter", definitions) == definitions["my_parameter"]


def test_get_with_definitions_overrides_builtin():
    definitions = {"2m_temperature": _definition(variable="t2")}
    assert PARAMS.get("2m_temperature", definitions)["variable"] == "t2"


def test_get_with_definitions_other_names_unchanged():
    definitions = {"2m_temperature": _definition(variable="t2")}
    # names which are not in `definitions` still come from _PARAMS
    assert PARAMS.get("10m_wind_speed", definitions) is _PARAMS["10m_wind_speed"]
    # an empty or None `definitions` changes nothing
    assert PARAMS.get("2m_temperature", {}) is _PARAMS["2m_temperature"]
    assert PARAMS.get("2m_temperature", None) is _PARAMS["2m_temperature"]


def test_params_not_modified_by_definitions():
    before = {name: dict(item) for name, item in _PARAMS.items()}
    definitions = {
        "2m_temperature": _definition(variable="t2"),
        "my_parameter": _definition(),
    }
    PARAMS.get("2m_temperature", definitions)
    PARAMS.field_parameter_metadata("2m_temperature", definitions)
    PARAMS.xarray_variable_and_metadata("2m_temperature", definitions)

    assert _PARAMS == before
    assert list(_PARAMS) == list(before)
    assert "my_parameter" not in _PARAMS
    # the override only applied to that call
    assert PARAMS.get("2m_temperature")["variable"] == "2t"
    assert PARAMS.get("2m_temperature") is _PARAMS["2m_temperature"]


def test_field_parameter_metadata_with_definitions():
    definitions = {
        "my_parameter": _definition(variable="my", units="m", standard_name="height", cf_units="m"),
        "2m_temperature": _definition(variable="t2", units="C", standard_name="air_temperature", cf_units="C"),
    }
    assert PARAMS.field_parameter_metadata("my_parameter", definitions) == {
        "parameter.variable": "my",
        "parameter.units": "m",
        "parameter.standard_name": "height",
    }
    assert PARAMS.field_parameter_metadata("2m_temperature", definitions) == {
        "parameter.variable": "t2",
        "parameter.units": "C",
        "parameter.standard_name": "air_temperature",
    }


def test_xarray_variable_and_metadata_with_definitions():
    definitions = {
        "my_parameter": _definition(variable="my", long_name="My parameter"),
        "2m_temperature": _definition(
            variable="t2",
            long_name="Temperature at 2 m",
            standard_name="air_temperature",
            cf_units="C",
        ),
    }
    variable, meta = PARAMS.xarray_variable_and_metadata("my_parameter", definitions)
    assert variable == "my"
    assert meta == {
        "long_name": "My parameter",
        # cf_units is None: falls back to "units"
        "units": "K",
        "standard_name": None,
    }

    variable, meta = PARAMS.xarray_variable_and_metadata("2m_temperature", definitions)
    assert variable == "t2"
    assert meta == {
        "long_name": "Temperature at 2 m",
        "units": "C",
        "standard_name": "air_temperature",
    }


def test_get_unknown_with_definitions():
    definitions = {"my_parameter": _definition()}
    with pytest.raises(KeyError):
        PARAMS.get("not_a_parameter", definitions)
    with pytest.raises(KeyError):
        PARAMS.field_parameter_metadata("not_a_parameter", definitions)
    with pytest.raises(KeyError):
        PARAMS.xarray_variable_and_metadata("not_a_parameter", definitions)


def test_check_definitions_none():
    assert PARAMS.check_definitions(None) is None


def test_check_definitions_empty():
    assert PARAMS.check_definitions({}) is None


def test_check_definitions_valid():
    assert PARAMS.check_definitions({"my_parameter": _definition()}) is None
    # the optional keys are accepted
    assert PARAMS.check_definitions({"my_parameter": _definition(paramId=123, edition=("1", "2"))}) is None
    # any number of entries, with extra keys
    assert PARAMS.check_definitions({"my_parameter": _definition(other=1), "another": _definition()}) is None
    # a definition overriding a built-in one is a definition like any other
    assert PARAMS.check_definitions({"2m_temperature": _definition()}) is None


@pytest.mark.parametrize("missing", sorted(_REQUIRED_KEYS))
def test_check_definitions_missing_key(missing):
    item = _definition()
    del item[missing]
    with pytest.raises(ValueError) as excinfo:
        PARAMS.check_definitions({"my_parameter": item})
    message = str(excinfo.value)
    assert missing in message
    assert "my_parameter" in message


def test_check_definitions_missing_keys():
    with pytest.raises(ValueError) as excinfo:
        PARAMS.check_definitions({"my_parameter": {}})
    message = str(excinfo.value)
    for key in _REQUIRED_KEYS:
        assert key in message


@pytest.mark.parametrize("definitions", ["2m_temperature", 1, ["my_parameter"], ({"my_parameter": _definition()},)])
def test_check_definitions_not_a_dict(definitions):
    with pytest.raises(TypeError) as excinfo:
        PARAMS.check_definitions(definitions)
    assert "definitions" in str(excinfo.value)


@pytest.mark.parametrize("item", [None, "not a definition", 1, ["variable"]])
def test_check_definitions_entry_not_a_dict(item):
    with pytest.raises(TypeError) as excinfo:
        PARAMS.check_definitions({"my_parameter": item})
    assert "my_parameter" in str(excinfo.value)


def test_check_definitions_entry_not_a_dict_names_entry():
    with pytest.raises(TypeError) as excinfo:
        PARAMS.check_definitions({"my_parameter": _definition(), "other_parameter": None})
    assert "other_parameter" in str(excinfo.value)
