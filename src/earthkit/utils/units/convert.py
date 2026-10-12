# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import warnings
from typing import Any, TypeAlias

import numpy as np

from earthkit.utils.decorators import dispatch

from .units import Units

ArrayLike: TypeAlias = Any


def convert(
    data: ArrayLike,
    target_units,
    source_units=None,
    errors: str = "raise",
) -> ArrayLike:
    """Convert the units of the data.

    Parameters
    ----------
    data : array, xarray.DataArray or xarray.Dataset
        The data to convert.
    target_units : str, pint.Unit, Units, list or dict
        The units to convert to. With a list, the data is converted to the one
        unit in it compatible with its source units; two compatible units in
        the list are ambiguous and always raise. A dict, mapping
        DataArray/variable names to units, is only supported for xarray objects.
    source_units : str, pint.Unit, Units or dict
        The units of the data. Required for arrays. For xarray objects it
        defaults to the ``units`` attribute; see the xarray implementation.
    errors : {"raise", "ignore"}, default "raise"
        What to do when the conversion cannot be performed. With ``"ignore"``
        a warning is logged and the data is returned unchanged. With ``"raise"``
        the error is raised.

    Returns
    -------
    array, xarray.DataArray or xarray.Dataset
        The converted data, of the same type as ``data``.

    Implementations
    ---------------
    :func:`convert` calls one of the following implementations depending on the type of ``data``:

    - :py:func:`earthkit.utils.units.array.convert` for arrays
    - :py:func:`earthkit.utils.units.xarray.convert` for xarray.DataArray and xarray.Dataset
    """
    if target_units is None:
        raise TypeError("target_units must be provided")
    dispatched = dispatch(convert, array=True, xarray=True, fieldlist=False)
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


def convert_units(
    data: ArrayLike,
    target_units=None,
    source_units=None,
    errors: str = "ignore",
) -> ArrayLike:
    """Deprecated, use :func:`earthkit.utils.units.convert` instead.

    .. warning:: Deprecated API. This function will be removed in a future release.
    """
    warnings.warn(
        "convert_units is deprecated and will be removed in a future release. "
        "Use earthkit.utils.units.convert instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return convert(*_legacy_args(data, target_units, source_units), errors)


def convert_array(
    data: ArrayLike,
    target_units=None,
    source_units=None,
    errors: str = "ignore",
) -> ArrayLike:
    """Deprecated, use :func:`earthkit.utils.units.convert` instead.

    .. warning:: Deprecated API. This function will be removed in a future release.
    """
    warnings.warn(
        "convert_array is deprecated and will be removed in a future release. "
        "Use earthkit.utils.units.convert instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    import earthkit.utils.units.array.convert as arr_convert

    return arr_convert(data, target_units, source_units, errors)


def convert_dataarray(
    data: ArrayLike,
    target_units=None,
    source_units=None,
    errors: str = "ignore",
) -> ArrayLike:
    """Deprecated, use :func:`earthkit.utils.units.convert` instead.

    .. warning:: Deprecated API. This function will be removed in a future release.
    """
    warnings.warn(
        "convert_dataarray is deprecated and will be removed in a future release. "
        "Use earthkit.utils.units.convert instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return convert(data, target_units, source_units, errors)


def convert_dataset(
    data: ArrayLike,
    target_units=None,
    source_units=None,
    errors: str = "ignore",
) -> ArrayLike:
    """Deprecated, use :func:`earthkit.utils.units.convert` instead.

    .. warning:: Deprecated API. This function will be removed in a future release.
    """
    warnings.warn(
        "convert_dataset is deprecated and will be removed in a future release. "
        "Use earthkit.utils.units.convert instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return convert(*_legacy_args(data, target_units, source_units), errors)


def _legacy_args(data, target_units, source_units):
    """Return the arguments for :func:`convert` that give the behaviour of the deprecated functions.

    Lists and numbers are converted as arrays, and for a Dataset a single ``source_units``
    selects the variables whose ``units`` attribute matches it, instead of applying to all of them.
    """
    if isinstance(data, (list, tuple, int, float)):
        return np.asarray(data), target_units, source_units

    import xarray as xr

    if (
        xr is not None
        and isinstance(data, xr.Dataset)
        and source_units is not None
        and not isinstance(source_units, dict)
    ):
        target_units = {
            name: target_units.get(name) if isinstance(target_units, dict) else target_units
            for name, var in data.data_vars.items()
            if are_equal(var.attrs.get("units"), source_units)
        }
    return data, target_units, source_units
