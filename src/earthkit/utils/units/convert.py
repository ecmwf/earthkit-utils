# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import sys
from typing import TYPE_CHECKING, Any, TypeAlias, Union

import pint

from .units import Q_, StrUnits, Units

LOG = logging.getLogger(__name__)

ArrayLike: TypeAlias = Any
UnitLike: TypeAlias = Union[str, pint.Unit, Units]
UnitSpec: TypeAlias = Union[UnitLike, dict[str, UnitLike], None]

if TYPE_CHECKING:
    import xarray  # type: ignore[import]


def is_module_loaded(module_name):
    return module_name in sys.modules


def are_equal(unit_1: UnitLike | None, unit_2: UnitLike | None) -> bool:
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
    return Units.from_any(unit_1) == Units.from_any(unit_2)


def convert_array(
    data: ArrayLike,
    target_units: UnitSpec = None,
    source_units: UnitSpec = None,
) -> ArrayLike:
    """
    Convert data from one set of units to another.

    Parameters
    ----------
    data : array-like
        The data to convert.
    target_units : str
        The units to convert to.
    source_units : str
        The units of the data.

    Returns
    -------
    array-like
        The converted data, or the original data if conversion is not possible.
    """
    if source_units is None or target_units is None:
        LOG.warning("source_units and target_units must both be provided to convert array data")
        return data
    if isinstance(target_units, dict) or isinstance(source_units, dict):
        LOG.warning("target_units and source_units as dictionaries are not supported for array objects")
        return data

    source_parsed = Units.from_any(source_units)
    target_parsed = Units.from_any(target_units)

    # If either unit is unrecognised by pint, we cannot convert
    source_pint = source_parsed.to_pint()
    target_pint = target_parsed.to_pint()
    if source_pint is None or target_pint is None:
        LOG.warning("Cannot convert between unrecognised units: %s -> %s", source_units, target_units)
        return data

    # No-op if units are the same
    if source_parsed == target_parsed:
        return data

    try:
        return Q_(data, source_pint).to(target_pint).magnitude
    except pint.errors.DimensionalityError:
        LOG.warning("Cannot convert incompatible units: %s -> %s", source_units, target_units)
        return data


def convert_dataarray(
    data: "xarray.DataArray",
    target_units: UnitSpec = None,
    source_units: UnitSpec = None,
) -> "xarray.DataArray":
    """
    Convert the units of an xarray.DataArray.

    Parameters
    ----------
    data : xarray.DataArray
        The DataArray to convert.
    target_units : str or dict, optional
        The units to convert to. If a dict, looks up the DataArray's
        ``name`` in the mapping. If the name is not found, no conversion
        is performed.
    source_units : str or dict, optional
        The units of the data. If a dict, looks up the DataArray's
        ``name`` in the mapping, falling back to ``data.attrs["units"]``.
        If a str, used directly as the source units.
        If None, tries to read from ``data.attrs["units"]``.

    Returns
    -------
    xarray.DataArray
        The converted DataArray, or the original if conversion is not possible.
    """
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("xarray is required to convert DataArray units") from exc

    if not isinstance(data, xr.DataArray):
        raise TypeError("data must be an xarray.DataArray")

    # Resolve target units
    if isinstance(target_units, dict):
        target_units_resolved = target_units.get(data.name)
    else:
        target_units_resolved = target_units
    if target_units_resolved is None:
        return data

    # Resolve source units
    if isinstance(source_units, dict):
        source_units_resolved = source_units.get(data.name, data.attrs.get("units"))
    elif isinstance(source_units, str):
        source_units_resolved = source_units
    else:
        source_units_resolved = data.attrs.get("units")
    if source_units_resolved is None:
        LOG.warning(f"No source units found for DataArray '{data.name}', cannot convert")
        return data

    converted = convert_array(data.data, target_units_resolved, source_units_resolved)

    # If convert_array returned the same object, data was not converted
    if converted is data.data:
        return data

    result = data.copy(deep=False)
    result.data = converted

    if target_units_resolved is not None:
        result.attrs = dict(result.attrs)
        result.attrs["units"] = str(target_units_resolved)

    return result


def are_compatible(unit_1: UnitLike | None, unit_2: UnitLike | None) -> bool:
    """Check if two units are dimensionally compatible."""
    unit_1_parsed = Units.from_any(unit_1)
    unit_2_parsed = Units.from_any(unit_2)

    if isinstance(unit_1_parsed, StrUnits) or isinstance(unit_2_parsed, StrUnits):
        return False

    unit_1_pint = unit_1_parsed.to_pint()
    unit_2_pint = unit_2_parsed.to_pint()

    try:
        Q_(1, unit_1_pint).to(unit_2_pint)
    except pint.errors.DimensionalityError:
        return False
    return True


def convert_dataset(
    data: "xarray.Dataset",
    target_units: UnitSpec = None,
    source_units: UnitSpec = None,
) -> "xarray.Dataset":
    """
    Convert the units of variables in an xarray.Dataset.

    Parameters
    ----------
    data : xarray.Dataset
        The Dataset to convert.
    target_units : str
        The units to convert to.
    source_units : str, optional
        The units to match. If None, any variable with units compatible
        with ``target_units`` will be converted. If provided, only variables
        whose current units match ``source_units`` will be converted.

    Returns
    -------
    xarray.Dataset
        The converted Dataset.
    """
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("xarray is required to convert Dataset units") from exc

    if not isinstance(data, xr.Dataset):
        raise TypeError("data must be an xarray.Dataset")

    result = None

    for name, da in data.data_vars.items():
        # Get source units for this variable, checking in order:
        source_units_for_var = da.attrs.get("units")
        if isinstance(source_units, dict):
            source_units_for_var = source_units.get(name, source_units_for_var)
        elif isinstance(source_units, str):
            if not are_equal(source_units, source_units_for_var):
                continue
        # No source units found for variable, skip it
        if source_units_for_var is None:
            continue

        # Get target units for this variable, checking in order:
        if isinstance(target_units, dict):
            target_units_for_var = target_units.get(name)
        else:
            target_units_for_var = target_units
        if target_units_for_var is None:
            continue

        if not are_compatible(source_units_for_var, target_units_for_var):
            continue

        if result is None:
            result = data.copy(deep=False)

        result[name] = convert_dataarray(da, target_units_for_var, source_units_for_var)

    return data if result is None else result


def convert_units(
    data: ArrayLike,
    target_units: UnitSpec = None,
    source_units: UnitSpec = None,
) -> ArrayLike:
    """
    Convert units for arrays, xarray.DataArray, or xarray.Dataset objects.

    Parameters
    ----------
    data : array-like or xarray.DataArray or xarray.Dataset
        The data to convert.
    target_units : str or dict, optional
        The units to convert to. If a dict, maps variable/DataArray names
        to target unit strings.
    source_units : str or dict, optional
        The units of the data. If a dict, maps variable/DataArray names
        to source unit strings. If a str and ``data`` is a Dataset, acts
        as a filter (only variables whose current units match are converted).
        If None and ``data`` is a DataArray, tries to read from
        ``data.attrs["units"]``. If ``data`` is a Dataset and
        ``source_units`` is None, variables with units compatible with
        ``target_units`` will be converted.

    Returns
    -------
    array-like or xarray.DataArray or xarray.Dataset
        The converted data.
    """
    if is_module_loaded("xarray"):
        import xarray as xr

        if isinstance(data, xr.DataArray):
            return convert_dataarray(data, target_units, source_units)
        if isinstance(data, xr.Dataset):
            return convert_dataset(data, target_units, source_units)

    return convert_array(data, target_units, source_units)
