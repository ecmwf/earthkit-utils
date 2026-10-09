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
    data : array, xarray.DataArray or xarray.Dataset
        The data to convert.
    target_units : str, pint.Unit, Units or dict
        The units to convert to. A dict, mapping DataArray/variable names to
        units, is only supported for xarray objects.
    source_units : str, pint.Unit, Units or dict
        The units of the data. Required for arrays. For xarray objects it
        defaults to the ``units`` attribute; see the xarray implementation.
    errors : {"ignore", "raise"}, default "ignore"
        What to do when the conversion cannot be performed. With ``"ignore"``
        a warning is logged and the data is returned unchanged. With ``"raise"``
        the error is raised.

    Returns
    -------
    array, xarray.DataArray or xarray.Dataset
        The converted data, of the same type as ``data``.

    Implementations
    ---------------
    :func:`convert_units` calls one of the following implementations depending on the type of ``data``:

    - :py:func:`earthkit.utils.units.array.convert_units` for arrays
    - :py:func:`earthkit.utils.units.xarray.convert_units` for xarray.DataArray and xarray.Dataset
    """
    dispatched = dispatch(convert_units, array=True, xarray=True, fieldlist=False)
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


def are_compatible(unit_1, unit_2) -> bool:
    """
    Check if data can be converted from one unit to another.

    Units are compatible when they are equal, including equal units that Pint
    does not recognise (e.g. "dBZ"), or when both are recognised by Pint and
    have the same dimensionality. This matches when :func:`convert_units` succeeds.

    Parameters
    ----------
    unit_1 : str, pint.Unit, Units or None
        The units to convert from.
    unit_2 : str, pint.Unit, Units or None
        The units to convert to.

    Returns
    -------
    bool
        True if the units are compatible, False otherwise.
    """
    unit_1, unit_2 = Units.from_any(unit_1), Units.from_any(unit_2)
    if unit_1 == unit_2:
        return True
    if unit_1.to_pint() is None or unit_2.to_pint() is None:
        return False
    return unit_1.to_pint().is_compatible_with(unit_2.to_pint())
