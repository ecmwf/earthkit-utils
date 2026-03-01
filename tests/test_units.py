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

from earthkit.utils.units import Units


@pytest.mark.parametrize(
    "input_units,expected_value,pint_object",
    [
        ("m/s", "meter / second", True),
        ("m s-1", "meter / second", True),
        ("m s^-1", "meter / second", True),
        ("m/s2", "meter / second ** 2", True),
        ("kg m**-2", "kilogram / meter ** 2", True),
        ("kilogram / meter ** 2", "kilogram / meter ** 2", True),
        ("invalid", "invalid", False),
    ],
)
def test_units_to_str(input_units, expected_value, pint_object):
    r = Units.from_any(input_units)
    assert str(r) == expected_value, f"{str(r)}"
    if pint_object:
        assert r.to_pint() is not None
    else:
        assert r.to_pint() is None


@pytest.mark.parametrize(
    "units,pint_object",
    [
        (["m/s", "m s-1", "m s**-1"], True),
        (["m/s2", "meter / second ** 2"], True),
        (["degC", "celsius"], True),
        (["K", "kelvin"], True),
        (["(0 - 1)", "percent", "%"], True),
        (["kg m**-2", "kilogram / meter ** 2"], True),
        (["kilogram / meter ** 2", "kilogram / meter ** 2"], True),
        (["invalid", "invalid"], False),
    ],
)
def test_units_equal(units, pint_object):
    units_str = units.copy()
    units = [Units.from_any(u) for u in units]

    first = units[0]

    # compare units to units
    for u in units[1:]:
        assert u == first
        if pint_object:
            assert u.to_pint() == first.to_pint()
            assert u.to_pint() is not None
        assert str(u) == str(first)

    # compare units to str
    for u, u_str in zip(units, units_str):
        assert u_str == u
        assert u == u_str

    # compare first units to all str
    for u in units_str:
        assert first == u
