# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import typing as T
from functools import partial

import array_api_compat

from .backend import _BACKENDS
from .backend import _CUPY  # noqa: F401
from .backend import _DEFAULT_BACKEND
from .backend import _JAX  # noqa: F401
from .backend import _NUMPY  # noqa: F401
from .backend import _TORCH  # noqa: F401
from .backend import get_backend  # noqa: F401
from .convert import array_to_numpy  # noqa: F401
from .convert import convert_array  # noqa: F401
from .device import to_device  # noqa: F401


# TODO: maybe this is not necessary
def other_namespace(xp: T.Any) -> T.Any:
    """Return the patched version of an array-api-compat namespace."""
    if not hasattr(xp, "histogram2d"):
        from .compute import histogram2d

        xp.histogram2d = partial(histogram2d, xp)
    if not hasattr(xp, "polyval"):
        from .compute import polyval

        xp.polyval = partial(polyval, xp)
    if not hasattr(xp, "percentile"):
        from .compute import percentile

        xp.percentile = partial(percentile, xp)

    if not hasattr(xp, "seterr"):
        from .compute import seterr

        xp.seterr = partial(seterr, xp)

    return xp


def array_namespace(*args: T.Any) -> T.Any:
    """Return the array namespace of the arguments.

    Parameters
    ----------
    *args: tuple
        Scalar or array-like arguments.

    Returns
    -------
    xp: module
        The array-api-compat namespace of the arguments. The namespace
        returned from array_api_compat.array_namespace(*args) is patched with
        extra/modified methods. When only a scalar is passed, the numpy namespace
        is returned.

    Notes
    -----
    The array namespace is extended with the following methods when necessary:
        - polyval: evaluate a polynomial (available in numpy)
        - percentile: compute the n-th percentile of the data along the
          specified axis (available in numpy)
        - histogram2d: compute a 2D histogram (available in numpy)
    Some other methods may be reimplemented for a given namespace to ensure correct
    behaviour. E.g. sign() for torch.
    """
    arrays = [a for a in args if hasattr(a, "shape")]
    if not arrays:
        return _DEFAULT_BACKEND.namespace
    else:
        xp = array_api_compat.array_namespace(*arrays)
        for b in _BACKENDS:
            if b.match_namespace(xp):
                return b.namespace

        return xp


# This is experimental and may not be needed in the future.
def array_namespace_xarray(data_object: T.Any) -> T.Any:
    """Attempt to infer the array namespace from the data object.

    Parameters
    ----------
    data_object : T.Any
        The data object from which to infer the array namespace.

    Returns
    -------
        The inferred array namespace.

    Raises
    ------
    TypeError
        If the array namespace cannot be inferred from the data object.
    """
    from earthkit.utils.module import is_module_loaded

    if not is_module_loaded("xarray"):
        raise TypeError("xarray is not installed, cannot infer array namespace from data object.")

    import xarray as xr

    if isinstance(data_object, xr.DataArray):
        print(f"data_object: {type(data_object.data)}")
        return array_namespace(data_object.data)
    elif isinstance(data_object, xr.Dataset):
        data_vars = list(data_object.data_vars)
        if data_vars:
            first = array_namespace(data_object[data_vars[0]].data)
            if all(array_namespace(data_object[var].data) is first for var in data_vars[1:]):
                return first
            else:
                raise TypeError(
                    "Data object contains variables with different array namespaces, "
                    "cannot infer a single xp for computation."
                )
        return None

    raise TypeError(
        "data_object must be an xarray.DataArray or xarray.Dataset, " f"got {type(data_object)} instead."
    )
