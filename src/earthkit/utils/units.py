# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import re
from typing import Any
from typing import TYPE_CHECKING
from typing import TypeAlias
import logging

import pint
from pint import UnitRegistry

LOG = logging.getLogger(__name__)

ArrayLike: TypeAlias = Any

if TYPE_CHECKING:
    import xarray  # type: ignore[import]

ureg: UnitRegistry = UnitRegistry()
Q_: type[pint.Quantity] = ureg.Quantity


UNIT_STR_ALIASES: dict[str, str] = {
    # "(0 - 1)" denotes a fractional value between 0 and 1, i.e. a pure, dimensionless ratio.
    "(0 - 1)": "dimensionless",
}


def _pintify(unit_str: str | None) -> str | pint.Unit:
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


def are_equal(unit_1: str | None, unit_2: str | None) -> bool:
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


def convert_array(
    data: ArrayLike,
    target_units: str | None = None,
    source_units: str | None = None,
    raises: bool = True
) -> ArrayLike:
    """
    Convert data from one set of units to another.

    Parameters
    ----------
    data : numpy.ndarray
        The data to convert.
    target_units : str
        The units to convert to.
    source_units : str
        The units of the data.

    Returns
    -------
    numpy.ndarray
        The converted data.
    """
    if source_units is None or target_units is None:
        if raises:
            raise ValueError("source_units and target_units must both be provided to convert array data")
        else:
            LOG.warning("source_units and target_units must both be provided to convert array data")
            return data

    target_units = _pintify(target_units)
    source_units = _pintify(source_units)

    return (data * source_units).to(target_units).magnitude


def convert_dataarray(
    data: "xarray.DataArray",
    target_units: str | None,
    source_units: str | None,
    raises: bool = True,
) -> "xarray.DataArray":
    """
    Convert the units of an xarray.DataArray.

    Parameters
    ----------
    data : xarray.DataArray
        The DataArray to convert.
    target_units : str
        The units to convert to.
    source_units : str
        The units of the data. If None, tries to read from ``data.attrs["units"]``.

    Returns
    -------
    xarray.DataArray
        The converted DataArray.
    """
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("xarray is required to convert DataArray units") from exc

    if not isinstance(data, xr.DataArray):
        raise TypeError("data must be an xarray.DataArray")

    if source_units is None:
        source_units = data.attrs.get("units")

    converted = convert_array(data.data, target_units, source_units)
    result = data.copy(deep=False)
    result.data = converted

    if target_units is not None:
        result.attrs = dict(result.attrs)
        result.attrs["units"] = str(_pintify(target_units))

    return result


def _are_compatible(unit_1: str | None, unit_2: str | None) -> bool:
    unit_1_parsed = _pintify(unit_1)
    unit_2_parsed = _pintify(unit_2)

    if isinstance(unit_1_parsed, str) or isinstance(unit_2_parsed, str):
        return False

    try:
        (1 * unit_1_parsed).to(unit_2_parsed)
    except pint.errors.DimensionalityError:
        return False
    return True


def convert_dataset(
    data: "xarray.Dataset",
    target_units: str | None,
    source_units: str | None,
) -> "xarray.Dataset":
    """
    Convert the units of variables in an xarray.Dataset.

    Parameters
    ----------
    data : xarray.Dataset
        The Dataset to convert.
    target_units : str
        The units to convert to.
    source_units : str
        The units to match. If None, any variable with units compatible
        with ``target_units`` will be converted.

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

    result = data.copy(deep=False)

    for name, da in data.data_vars.items():
        var_units = da.attrs.get("units")
        if var_units is None:
            continue

        if source_units is None:
            if not _are_compatible(var_units, target_units):
                continue
        else:
            if not are_equal(var_units, source_units):
                continue

        result[name] = convert_dataarray(da, target_units, var_units)

    return result


def convert_units(
    data: ArrayLike,
    target_units: str | None,
    source_units: str | None,
) -> ArrayLike:
    """
    Convert units for arrays, xarray.DataArray, or xarray.Dataset objects.

    Parameters
    ----------
    data : array-like or xarray.DataArray or xarray.Dataset
        The data to convert.
    target_units : str
        The units to convert to.
    source_units : str
        The units of the data. If None and ``data`` is a DataArray,
        tries to read from ``data.attrs["units"]``. If ``data`` is a Dataset and
        ``source_units`` is None, variables with units compatible with
        ``target_units`` will be converted.

    Returns
    -------
    array-like or xarray.DataArray or xarray.Dataset
        The converted data.
    """
    try:
        import xarray as xr
    except ImportError:
        xr = None

    if xr is not None:
        if isinstance(data, xr.DataArray):
            return convert_dataarray(data, target_units, source_units)
        if isinstance(data, xr.Dataset):
            return convert_dataset(data, target_units, source_units)

    return convert_array(data, target_units, source_units)
