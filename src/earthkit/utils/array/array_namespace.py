import typing as T

import array_api_compat

from .namespace import _DEFAULT_NAMESPACE
from .namespace import _NAMESPACES
from .namespace import UnknownPatchedNamespace


def _get_array_name(xp):
    name = xp.__name__
    if "jax" in name:
        return "jax"
    elif "numpy" in name:
        return "numpy"
    elif "cupy" in name:
        return "cupy"
    elif "torch" in name:
        return "torch"
    else:
        return None


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
        return _DEFAULT_NAMESPACE
    else:
        xp = array_api_compat.array_namespace(*arrays)
        namespace = _NAMESPACES.get(_get_array_name(xp))
        if namespace is None:
            namespace = UnknownPatchedNamespace(xp)
        return namespace
