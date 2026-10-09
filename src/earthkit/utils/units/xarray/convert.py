# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import xarray as xr

from .. import array

LOG = logging.getLogger(__name__)


def convert_units(
    data: xr.DataArray | xr.Dataset,
    target_units=None,
    source_units=None,
    errors: str = "ignore",
) -> xr.DataArray | xr.Dataset:
    """Convert the units of an xarray.DataArray or of each data variable of an xarray.Dataset.

    The ``units`` attribute of each converted DataArray/variable is set to its target units.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        The data to convert.
    target_units : str, pint.Unit, Units or dict
        The units to convert to. A dict maps DataArray/variable names to units,
        and the DataArrays/variables not in it are left unchanged.
    source_units : str, pint.Unit, Units or dict, optional
        The units of the data, overriding the ``units`` attribute. A dict maps
        DataArray/variable names to units, falling back to the ``units`` attribute.
    errors : {"ignore", "raise"}, default "ignore"
        What to do when a DataArray/variable cannot be converted, including when
        it has no source units. With ``"ignore"`` a warning is logged and it is
        returned unchanged, with ``"raise"`` the error is raised.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The converted data.
    """
    import pint

    if errors not in ("ignore", "raise"):
        raise ValueError(f"errors must be 'ignore' or 'raise', got {errors!r}")

    if isinstance(data, xr.Dataset):
        return data.assign({
            name: convert_units(var, target_units, source_units, errors) for name, var in data.data_vars.items()
        })

    target_units = _get(target_units, data.name)
    if target_units is None:
        return data

    source_units = _get(source_units, data.name)
    if source_units is None:
        source_units = data.attrs.get("units")
    if source_units is None:
        message = f"No source units found for '{data.name}'"
        if errors == "raise":
            raise ValueError(message)
        LOG.warning(message)
        return data

    try:
        # For dask data, the units are checked when dask infers the output dtype
        result = xr.apply_ufunc(
            array.convert_units,
            data,
            kwargs={"target_units": target_units, "source_units": source_units, "errors": "raise"},
            keep_attrs=True,
            dask="parallelized",
        )
    except (ValueError, pint.errors.PintError) as exc:
        if errors == "raise":
            raise
        LOG.warning(f"Cannot convert units of '{data.name}': {exc}")
        return data

    result.attrs["units"] = str(target_units)
    return result


def _get(units, name):
    """Return the units for ``name`` from a dict of units, or the units themselves."""
    return units.get(name) if isinstance(units, dict) else units
