# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any, TypeAlias

from earthkit.utils.decorators import dispatch

from .units import Units

ArrayLike: TypeAlias = Any


def convert_units(
    data: ArrayLike,
    target_units=None,
    source_units=None,
    errors: str = "ignore",
) -> ArrayLike:
    """Convert the units of the data.

    Parameters
    ----------
    data : array
        The data to convert.
    target_units : str, pint.Unit or Units
        The units to convert to.
    source_units : str, pint.Unit or Units
        The units of the data.
    errors : {"ignore", "raise"}, default "ignore"
        What to do when the conversion cannot be performed. With ``"ignore"``
        a warning is logged and the data is returned unchanged. With ``"raise"``
        a ``ValueError`` is raised.

    Returns
    -------
    array
        The converted data.

    Implementations
    ---------------
    :func:`convert_units` calls one of the following implementations depending on the type of ``data``:

    - :py:func:`earthkit.utils.units.array.convert_units` for arrays
    """
    dispatched = dispatch(convert_units, array=True, xarray=False, fieldlist=False)
    return dispatched(data, target_units, source_units, errors=errors)


def are_equal(unit_1, unit_2) -> bool:
    """
    Check if two units are equivalent.

    Parameters
    ----------
    unit_1 : str, pint.Unit, Units or None
        The first unit.
    unit_2 : str, pint.Unit, Units or None
        The second unit.

    Returns
    -------
    bool
        True if the units are equivalent, False otherwise.
    """
    return Units.from_any(unit_1) == Units.from_any(unit_2)
