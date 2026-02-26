# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import re

import pint
from pint import UnitRegistry

ureg = UnitRegistry()
Q_ = ureg.Quantity


UNIT_STR_ALIASES = {"(0 - 1)": "percent"}


def _pintify(unit_str):
    """
    Convert a unit string to a Pint-compatible unit.

    For example, it converts "m s-1" to "m.s^-1".

    Parameters
    ----------
    unit_str : str
        The unit string to convert.
    """

    if unit_str is None:
        unit_str = "dimensionless"

    if unit_str in UNIT_STR_ALIASES:
        unit_str = UNIT_STR_ALIASES[unit_str]

    # Replace spaces with dots
    unit_str = unit_str.replace(" ", ".")

    # Insert ^ between characters and numbers (including negative numbers)
    unit_str = re.sub(r"([a-zA-Z])(-?\d+)", r"\1^\2", unit_str)

    try:
        result = ureg(unit_str).units
    except pint.errors.UndefinedUnitError:
        result = unit_str
    return result


def are_equal(unit_1, unit_2):
    """
    Check if two units are equivalent.

    Parameters
    ----------
    unit_1 : str
        The first unit.
    unit_2 : str
        The second unit.

    Returns
    -------
    bool
        True if the units are equivalent, False otherwise.
    """
    return _pintify(unit_1) == _pintify(unit_2)


def convert(data, source_units, target_units):
    """
    Convert data from one set of units to another.

    Parameters
    ----------
    data : numpy.ndarray
        The data to convert.
    source_units : str
        The units of the data.
    target_units : str
        The units to convert to.

    Returns
    -------
    numpy.ndarray
        The converted data.
    """
    source_units = _pintify(source_units)
    target_units = _pintify(target_units)

    return (data * source_units).to(target_units).magnitude
