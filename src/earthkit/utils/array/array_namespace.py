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


def _get_namespace_from_array(*arrays):
    xp = array_api_compat.array_namespace(*arrays)
    namespace = _NAMESPACES.get(_get_array_name(xp))
    if namespace is None:
        namespace = UnknownPatchedNamespace(xp)
    return namespace


def array_namespace(*args: T.Any) -> T.Any:
    """Return the array namespace of the arguments.

    Parameters
    ----------
    *args: tuple
        Scalar, string or array-like arguments.

    Returns
    -------
    xp: module
        The patched array namespace of the arguments. The namespace
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

    arrays = [a for a in args if array_api_compat.is_array_api_obj(a)]
    if not arrays:
        # TODO: decide if we want to support this or not
        # i.e. array_namespace("numpy")
        # array_namespace(np)
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, str):
                xp = _NAMESPACES[arg]
            else:
                if hasattr(arg, "asarray"):
                    xp = _get_namespace_from_array(arg.asarray(0))
                elif hasattr(arg, "array"):
                    xp = _get_namespace_from_array(arg.array(0))
                else:
                    xp = _DEFAULT_NAMESPACE
        else:
            xp = _DEFAULT_NAMESPACE
    else:
        xp = _get_namespace_from_array(*arrays)

    return xp
