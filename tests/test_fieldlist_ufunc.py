# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
"""Tests for ``fieldlist_ufunc``.

The fields are built in memory, from GRIB messages generated with eccodes samples,
so these tests do not download or read any data.
"""

import copy

import numpy as np
import pytest

pytest.importorskip("earthkit.data")
eccodes = pytest.importorskip("eccodes")

import earthkit.data as ekd  # noqa: E402

from earthkit.utils.decorators._fieldlist_ufunc import fieldlist_ufunc  # noqa: E402
from earthkit.utils.parameters import PARAMS  # noqa: E402
from earthkit.utils.parameters._params import _PARAMS, _REQUIRED_KEYS  # noqa: E402

# ecCodes parameters used as input fields: "167" is 2t, "168" is 2d
PARAM_2T = 167
PARAM_2D = 168


def _grib_message(param_id, values):
    """Generate a GRIB message in memory with the given parameter and values."""
    handle = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    try:
        eccodes.codes_set(handle, "Ni", 4)
        eccodes.codes_set(handle, "Nj", 3)
        eccodes.codes_set(handle, "param", str(param_id))
        eccodes.codes_set_array(handle, "values", np.asarray(values, dtype="float64"))
        return eccodes.codes_get_message(handle)
    finally:
        eccodes.codes_release(handle)


def make_field(param_id=PARAM_2T, n=12, offset=0.0):
    """Build a Field in memory, with ``parameter.variable`` and ``metadata.paramId`` set."""
    values = np.arange(n, dtype="float64") + offset
    try:
        message = _grib_message(param_id, values)
    except Exception as e:  # eccodes samples are not available
        pytest.skip(f"cannot generate a GRIB message in memory: {e}")
    return ekd.from_source("memory", message).to_fieldlist()[0]


def run(func, groups, **fieldlist_ufunc_kwargs):
    """Apply ``func`` to ``groups``, as Fields and as FieldLists.

    ``groups`` is a tuple of lists of Fields, one list per argument of ``func``. All
    the lists have the same length. The result of every call is returned as a list:
    first the Field computed from the first field of each group, then the fields of the
    FieldList computed from all of them. This exercises both code paths.
    """
    n = len(groups[0])
    assert all(len(g) == n for g in groups), "groups must have the same length"

    result = fieldlist_ufunc(func, *[g[0] for g in groups], fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)
    assert isinstance(result, ekd.Field)

    fields = fieldlist_ufunc(
        func,
        *[ekd.FieldList.from_fields(g) for g in groups],
        fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs,
    )
    assert isinstance(fields, ekd.FieldList)
    assert len(fields) == n

    return [result, *fields]


def test_input_field_metadata():
    """Check that the generated input fields carry the metadata the tests rely on."""
    f = make_field()
    assert f.get("parameter.variable") == "2t"
    assert f.get("metadata.paramId") == PARAM_2T


# --------------------------------------------------------------------------------------
# Built-in parameter names
# --------------------------------------------------------------------------------------


def test_builtin_default_parameter():
    f = make_field()
    for r in run(lambda x: x + 1, ([f],), default_parameter="temperature"):
        assert np.allclose(r.values, f.values + 1)
        assert r.get("parameter.variable") == "t"
        assert r.get("parameter.units") == "K"


def test_builtin_parameters_mapping():
    f = make_field()
    for r in run(lambda x: x * 2, ([f],), parameters={"2t": "2m_dewpoint"}):
        assert np.allclose(r.values, f.values * 2)
        assert r.get("parameter.variable") == "2d"
        assert r.get("parameter.units") == "K"


def test_builtin_param_ids_mapping():
    f = make_field()
    for r in run(lambda x: x - 1, ([f],), param_ids={PARAM_2T: "pressure"}):
        assert np.allclose(r.values, f.values - 1)
        assert r.get("parameter.variable") == "pres"
        assert r.get("parameter.units") == "Pa"


def test_builtin_two_fieldlists():
    a = make_field()
    b = make_field(PARAM_2D, offset=10.0)
    results = run(lambda x, y: x + y, ([a, b], [b, a]), default_parameter="potential_temperature")
    assert np.allclose(results[0].values, a.values + b.values)
    assert np.allclose(results[1].values, a.values + b.values)
    assert np.allclose(results[2].values, b.values + a.values)
    for r in results:
        assert r.get("parameter.variable") == "pt"
        assert r.get("parameter.units") == "K"


def test_input_fields_are_not_modified():
    f = make_field()
    run(lambda x: x + 1, ([f],), default_parameter="temperature")
    assert f.get("parameter.variable") == "2t"
    assert f.get("metadata.paramId") == PARAM_2T


def test_no_name_determined():
    f = make_field()
    with pytest.raises(ValueError) as excinfo:
        fieldlist_ufunc(lambda x: x, f)
    assert "default_parameter" in str(excinfo.value)


def test_old_default_variable_no_longer_used():
    # the removed "default_variable" key is ignored: alone it cannot determine the name
    f = make_field()
    with pytest.raises(ValueError, match="Could not determine parameter name"):
        fieldlist_ufunc(lambda x: x, f, fieldlist_ufunc_kwargs={"default_variable": "temperature"})


def test_unknown_name():
    f = make_field()
    with pytest.raises(KeyError):
        fieldlist_ufunc(lambda x: x, f, fieldlist_ufunc_kwargs={"default_parameter": "not_a_parameter"})


# --------------------------------------------------------------------------------------
# Caller supplied definitions
# --------------------------------------------------------------------------------------


def definition(**overrides):
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


CUSTOM = {"my_parameter": definition(variable="my", units="Pa")}


def test_custom_default_parameter():
    f = make_field()
    for r in run(lambda x: x + 1, ([f],), default_parameter="my_parameter", parameter_definitions=CUSTOM):
        assert np.allclose(r.values, f.values + 1)
        assert r.get("parameter.variable") == "my"
        assert r.get("parameter.units") == "Pa"


def test_custom_parameters_mapping():
    f = make_field()
    for r in run(
        lambda x: x + 1,
        ([f],),
        parameters={"2t": "my_parameter"},
        parameter_definitions=CUSTOM,
    ):
        assert r.get("parameter.variable") == "my"
        assert r.get("parameter.units") == "Pa"


def test_custom_param_ids_mapping():
    f = make_field()
    for r in run(
        lambda x: x + 1,
        ([f],),
        param_ids={PARAM_2T: "my_parameter"},
        parameter_definitions=CUSTOM,
    ):
        assert r.get("parameter.variable") == "my"
        assert r.get("parameter.units") == "Pa"


def test_custom_unknown_name():
    f = make_field()
    with pytest.raises(KeyError):
        fieldlist_ufunc(
            lambda x: x,
            f,
            fieldlist_ufunc_kwargs={"default_parameter": "another_parameter", "parameter_definitions": CUSTOM},
        )


def test_definitions_override_builtin():
    f = make_field()
    definitions = {"temperature": definition(variable="t_overridden", units="Pa")}
    for r in run(lambda x: x + 1, ([f],), default_parameter="temperature", parameter_definitions=definitions):
        assert r.get("parameter.variable") == "t_overridden"
        assert r.get("parameter.units") == "Pa"

    # the override only applied to that call
    assert PARAMS.get("temperature") is _PARAMS["temperature"]
    assert _PARAMS["temperature"]["variable"] == "t"


def test_definitions_not_used():
    f = make_field()
    # a name which is not in the definitions still comes from the parameter database
    definitions = {"my_parameter": definition()}
    for r in run(lambda x: x + 1, ([f],), default_parameter="temperature", parameter_definitions=definitions):
        assert r.get("parameter.variable") == "t"


def test_params_not_modified():
    before = {name: dict(item) for name, item in _PARAMS.items()}
    f = make_field()
    definitions = {
        "my_parameter": definition(),
        "temperature": definition(variable="t_overridden"),
    }
    run(lambda x: x + 1, ([f],), default_parameter="temperature", parameter_definitions=definitions)
    assert _PARAMS == before
    assert list(_PARAMS) == list(before)


# --------------------------------------------------------------------------------------
# Checking the definitions
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("definitions", ["my_parameter", 1, ["my_parameter"], ({"my_parameter": definition()},)])
def test_definitions_not_a_dict(definitions):
    f = make_field()
    with pytest.raises(TypeError) as excinfo:
        fieldlist_ufunc(
            lambda x: x,
            f,
            fieldlist_ufunc_kwargs={"default_parameter": "temperature", "parameter_definitions": definitions},
        )
    assert "definitions" in str(excinfo.value)


@pytest.mark.parametrize("item", [None, "not a definition", 1, ["variable"]])
def test_definitions_entry_not_a_dict(item):
    f = make_field()
    with pytest.raises(TypeError) as excinfo:
        fieldlist_ufunc(
            lambda x: x,
            f,
            fieldlist_ufunc_kwargs={"default_parameter": "temperature", "parameter_definitions": {"my": item}},
        )
    assert "my" in str(excinfo.value)


@pytest.mark.parametrize("missing", sorted(_REQUIRED_KEYS))
def test_definitions_missing_key(missing):
    f = make_field()
    item = definition()
    del item[missing]
    with pytest.raises(ValueError) as excinfo:
        fieldlist_ufunc(
            lambda x: x,
            f,
            fieldlist_ufunc_kwargs={"default_parameter": "temperature", "parameter_definitions": {"my": item}},
        )
    assert missing in str(excinfo.value)


def test_invalid_definitions_when_unused():
    f = make_field()
    # the parameter database is used, the invalid definitions are still checked
    with pytest.raises(ValueError):
        fieldlist_ufunc(
            lambda x: x,
            f,
            fieldlist_ufunc_kwargs={
                "default_parameter": "temperature",
                "parameter_definitions": {"my": {"units": "K"}},
            },
        )


def test_invalid_definitions_empty_fieldlist():
    # an empty FieldList does not skip the check
    with pytest.raises(TypeError):
        fieldlist_ufunc(
            lambda x: x,
            ekd.FieldList.from_fields([]),
            fieldlist_ufunc_kwargs={"default_parameter": "temperature", "parameter_definitions": "my_parameter"},
        )


def test_valid_definitions_empty_fieldlist():
    fl = fieldlist_ufunc(
        lambda x: x,
        ekd.FieldList.from_fields([]),
        fieldlist_ufunc_kwargs={"default_parameter": "my_parameter", "parameter_definitions": CUSTOM},
    )
    assert isinstance(fl, ekd.FieldList)
    assert len(fl) == 0


def test_definitions_none():
    f = make_field()
    for r in run(lambda x: x + 1, ([f],), default_parameter="temperature", parameter_definitions=None):
        assert r.get("parameter.variable") == "t"


# --------------------------------------------------------------------------------------
# The callers arguments are left alone
# --------------------------------------------------------------------------------------


def test_fieldlist_ufunc_kwargs_not_modified():
    f = make_field()
    definitions = {"my_parameter": definition()}
    kwargs = {
        "parameters": {"2t": "my_parameter"},
        "param_ids": {PARAM_2D: "temperature"},
        "default_parameter": "my_parameter",
        "parameter_definitions": definitions,
    }
    before = copy.deepcopy(kwargs)

    fieldlist_ufunc(lambda x: x + 1, f, fieldlist_ufunc_kwargs=kwargs)

    assert kwargs == before
    assert list(kwargs) == list(before)
    # the nested objects are still the ones given by the caller
    assert kwargs["parameter_definitions"] is definitions
    assert definitions == before["parameter_definitions"]


def test_definitions_not_modified():
    f = make_field()
    definitions = {"my_parameter": definition()}
    before = copy.deepcopy(definitions)

    run(lambda x: x + 1, ([f],), default_parameter="my_parameter", parameter_definitions=definitions)

    assert definitions == before
    assert list(definitions) == list(before)
