# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from .units import Units

def convert_units_numpy(
    data: "xarray.DataArray",
    from_units: str,
    to_units: str,
) -> "xarray.DataArray":
    """Convert the units of a xarray DataArray using NumPy.

    Parameters
    ----------
    data : xarray.DataArray
        The data to convert.
    from_units : str
        The current units of the data.
    to_units : str
        The desired units of the data.

    Returns
    -------
    xarray.DataArray
        The converted data.

    """
    from .units import Units

    converter = Units()
    return converter.convert_numpy(
        data, from_units, to_units
    )
