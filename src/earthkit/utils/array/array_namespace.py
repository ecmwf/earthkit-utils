import typing as T

import array_api_compat

from .namespace.namespace import PatchedNamespace

# TODO: avoid using testing internals
from .testing.backend import _BACKENDS
from .testing.backend import _DEFAULT_BACKEND


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

        return PatchedNamespace(xp)


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
