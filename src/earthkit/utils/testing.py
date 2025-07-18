# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
import os
from importlib import import_module

from earthkit.utils.array.backend import backend_from_name

LOG = logging.getLogger(__name__)


_ROOT_DIR = top = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if not os.path.exists(os.path.join(_ROOT_DIR, "tests", "data")):
    _ROOT_DIR = "./"


def modules_installed(*modules):
    for module in modules:
        try:
            import_module(module)
        except ImportError:
            return False
    return True


def MISSING(*modules):
    return not modules_installed(*modules)


NO_TORCH = not modules_installed("torch")
NO_CUPY = not modules_installed("cupy")
NO_JAX = not modules_installed("jax")
if not NO_CUPY:
    try:
        import cupy as cp

        a = cp.ones(2)
    except Exception:
        NO_CUPY = True

ARRAY_BACKENDS = ["numpy"]
if not NO_TORCH:
    ARRAY_BACKENDS.append("torch")

if not NO_CUPY:
    ARRAY_BACKENDS.append("cupy")


ARRAY_BACKENDS = [backend_from_name(b) for b in ARRAY_BACKENDS]
_ARRAY_BACKENDS_BY_NAME = {b.name: b for b in ARRAY_BACKENDS}

NO_XARRAY = not modules_installed("xarray")


def match_dtype(array, backend, dtype):
    """Return True if the dtype of an array matches the specified dtype."""
    if dtype is not None:
        dtype = backend.make_dtype(dtype)
        r = array.dtype == dtype if dtype is not None else False
        return r


def check_array_type(array, expected_backend, dtype=None):
    from earthkit.utils.array import get_backend

    b1 = get_backend(array)
    b2 = get_backend(expected_backend)

    assert b1 == b2, f"{b1=}, {b2=}"

    expected_dtype = dtype
    if expected_dtype is not None:
        assert match_dtype(array, b2, expected_dtype), f"{array.dtype}, {expected_dtype=}"
        # assert b2.match_dtype(array, expected_dtype), f"{array.dtype}, {expected_dtype=}"


def get_array_namespace(backend):
    if backend is None:
        backend = "numpy"

    from earthkit.utils.array import get_backend

    return get_backend(backend).namespace


def get_array_backend(backend, skip=None, raise_on_missing=True):
    if backend is None:
        backend = "numpy"

    if isinstance(backend, list):
        res = []
        for b in backend:
            b = get_array_backend(b, raise_on_missing=raise_on_missing)
            if b:
                res.append(b)
        return res

    if isinstance(backend, str):
        b = _ARRAY_BACKENDS_BY_NAME.get(backend)
        if b is None:
            if raise_on_missing:
                raise ValueError(f"Unknown array backend: {backend}")
        return b

    return backend


def skip_array_backend(backends, skip):
    if not isinstance(backends, (list, tuple)):
        backends = [backends]
    if not isinstance(skip, (list, tuple)):
        skip = [skip]

    if not skip:
        return backends

    backends = get_array_backend(backends)
    skip = get_array_backend(skip, raise_on_missing=False)
    if not skip:
        return backends

    res = []
    for b in backends:
        if b not in skip:
            res.append(b)
    return res
