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
    filter_by_source_units: bool = False,
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
    source_units : str, pint.Unit, Units, list or dict
        The units of the data. Required for arrays. For xarray objects it
        defaults to the ``units`` attribute; see the xarray implementation.
        A list is only accepted with ``filter_by_source_units``.
    errors : {"raise", "ignore"}, default "raise"
        What to do when the conversion cannot be performed. With ``"ignore"``
        a warning is logged and the data is returned unchanged. With ``"raise"``
        the error is raised.
    filter_by_source_units : bool, default False
        When True, ``source_units`` selects the data to convert instead of
        overriding its units metadata: only the data whose own units (e.g. the
        ``units`` attribute of an xarray object) equal ``source_units``, or one of
        the units in a ``source_units`` list, is converted. The rest is returned
        unchanged without a warning or error. ``source_units`` must then be a
        unit or a list of units, not ``None`` or a dict.
        Has no effect on arrays, whose units are ``source_units``.

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
    return dispatched(data, target_units, source_units, errors=errors, filter_by_source_units=filter_by_source_units)


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
    filter_by_source_units: bool = True,
) -> ArrayLike:
    """Deprecated, use :func:`earthkit.utils.units.convert` instead.

    .. warning:: Deprecated API. This function will be removed in a future release.

    ``filter_by_source_units`` defaults to True for backwards compatibility: for an
    xarray.Dataset, a single ``source_units`` selects the variables to convert. Use
    False to apply ``source_units`` to all the variables, the default of
    :func:`earthkit.utils.units.convert`.
    """
    warnings.warn(
        "convert_units is deprecated and will be removed in a future release. "
        "Use earthkit.utils.units.convert instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    data, filter_by_source_units = _legacy_args(data, source_units, filter_by_source_units)
    return convert(data, target_units, source_units, errors=errors, filter_by_source_units=filter_by_source_units)


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
    filter_by_source_units: bool = True,
) -> ArrayLike:
    """Deprecated, use :func:`earthkit.utils.units.convert` instead.

    .. warning:: Deprecated API. This function will be removed in a future release.

    ``filter_by_source_units`` defaults to True for backwards compatibility: for an
    xarray.Dataset, a single ``source_units`` selects the variables to convert. Use
    False to apply ``source_units`` to all the variables, the default of
    :func:`earthkit.utils.units.convert`.
    """
    warnings.warn(
        "convert_dataset is deprecated and will be removed in a future release. "
        "Use earthkit.utils.units.convert instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    data, filter_by_source_units = _legacy_args(data, source_units, filter_by_source_units)
    return convert(data, target_units, source_units, errors=errors, filter_by_source_units=filter_by_source_units)


def _legacy_args(data, source_units, filter_by_source_units):
    """Return the data and source units filter that give the behaviour of the deprecated functions.

    Lists and numbers are converted as arrays, and the source units filter only applies
    to a Dataset with a single ``source_units`` (the deprecated functions silently ignored
    it otherwise, while :func:`convert` rejects a ``None`` or dict ``source_units``).
    """
    if isinstance(data, (list, tuple, int, float)):
        return np.asarray(data), False

    import xarray as xr

    return data, (
        filter_by_source_units
        and isinstance(data, xr.Dataset)
        and source_units is not None
        and not isinstance(source_units, dict)
    )
