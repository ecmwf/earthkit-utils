# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import array_api_compat

from earthkit.utils.array.array_namespace import _get_array_name

from .backends import _BACKENDS
from .backends import UnknownArrayBackend


def backend_from_array(array, raise_exception=True):
    """Return the array backend of an array-like object."""
    xp = array_api_compat.array_namespace(array)
    for b in _BACKENDS.values():
        if b.match_namespace(xp):
            return b

    if raise_exception:
        raise ValueError(f"Can't find namespace for array type={type(array)}")

    return xp


def backend_from_name(name, raise_exception=True):
    r = _BACKENDS.get(name, None)
    if raise_exception and r is None:
        raise ValueError(f"Unknown array backend name={name}")
    return r


def backend_from_module(module, raise_exception=True):
    import inspect

    r = None
    if inspect.ismodule(module):
        name = _get_array_name(module)
        r = _BACKENDS.get(name, None)
    if raise_exception and r is None:
        raise ValueError(f"Unknown array backend module={module}")
    return r


def get_backend(data):
    import inspect

    if isinstance(data, UnknownArrayBackend):
        return data
    elif isinstance(data, str):
        return backend_from_name(data, raise_exception=True)
    elif inspect.ismodule(data):
        return backend_from_module(data, raise_exception=True)
    else:
        return backend_from_array(data)
