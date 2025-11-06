# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
from importlib import import_module

from earthkit.utils.array.namespace import _NAMESPACES

LOG = logging.getLogger(__name__)


def _modules_installed(*modules):
    for module in modules:
        try:
            import_module(module)
        except ImportError:
            return False
    return True


NO_TORCH = not _modules_installed("torch")
NO_CUPY = not _modules_installed("cupy")
NO_JAX = not _modules_installed("jax")
NO_XARRAY = not _modules_installed("xarray")
if not NO_CUPY:
    try:
        import cupy as cp

        a = cp.ones(2)
    except Exception:
        NO_CUPY = True

ARRAY_NAMES = ["numpy"]
if not NO_TORCH:
    ARRAY_NAMES.append("torch")
if not NO_CUPY:
    ARRAY_NAMES.append("cupy")
# if not NO_JAX:
#     ARRAY_NAMES.append("jax")


def _get_namespace_devices(names):
    devices = []
    namespaces = []
    for name in names:
        xp = _NAMESPACES[name]
        xp_devices = xp.devices()
        n_devices = len(xp_devices)
        namespaces += [xp] * n_devices
        devices += xp_devices
    return list(zip(namespaces, devices))


NAMESPACE_DEVICES = _get_namespace_devices(ARRAY_NAMES)


# def match_dtype(array, backend, dtype):
#     """Return True if the dtype of an array matches the specified dtype."""
#     if dtype is not None:
#         dtype = backend.make_dtype(dtype)
#         r = array.dtype == dtype if dtype is not None else False
#         return r


# def check_array_type(array, expected_backend, dtype=None):
#     from earthkit.utils.array import get_backend

#     b1 = get_backend(array)
#     b2 = get_backend(expected_backend)

#     assert b1 == b2, f"{b1=}, {b2=}"

#     expected_dtype = dtype
#     if expected_dtype is not None:
#         assert match_dtype(array, b2, expected_dtype), f"{array.dtype}, {expected_dtype=}"
#         # assert b2.match_dtype(array, expected_dtype), f"{array.dtype}, {expected_dtype=}"


# def get_array_backend(backend, skip=None, raise_on_missing=True):
#     if backend is None:
#         backend = "numpy"

#     if isinstance(backend, list):
#         res = []
#         for b in backend:
#             b = get_array_backend(b, raise_on_missing=raise_on_missing)
#             if b:
#                 res.append(b)
#         return res

#     if isinstance(backend, str):
#         b = _ARRAY_BACKENDS_BY_NAME.get(backend)
#         if b is None:
#             if raise_on_missing:
#                 raise ValueError(f"Unknown array backend: {backend}")
#         return b

#     return backend


# def skip_array_backend(backends, skip):
#     if not isinstance(backends, (list, tuple)):
#         backends = [backends]
#     if not isinstance(skip, (list, tuple)):
#         skip = [skip]

#     if not skip:
#         return backends

#     backends = get_array_backend(backends)
#     skip = get_array_backend(skip, raise_on_missing=False)
#     if not skip:
#         return backends

#     res = []
#     for b in backends:
#         if b not in skip:
#             res.append(b)
#     return res
