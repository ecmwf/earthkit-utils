# (C) Copyright 2020 ECMWF.
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

NO_JAX = not modules_installed("jax")


def check_array_type(array, expected_backend, dtype=None):
    from earthkit.utils.array import get_backend

    b1 = get_backend(array)
    b2 = get_backend(expected_backend)

    assert b1 == b2, f"{b1=}, {b2=}"

    expected_dtype = dtype
    if expected_dtype is not None:
        assert b2.match_dtype(array, expected_dtype), f"{array.dtype}, {expected_dtype=}"


def get_array_namespace(backend):
    if backend is None:
        backend = "numpy"

    from earthkit.utils.array import get_backend

    return get_backend(backend).namespace


ARRAY_BACKENDS = ["numpy"]
if not NO_TORCH:
    ARRAY_BACKENDS.append("torch")

if not NO_CUPY:
    ARRAY_BACKENDS.append("cupy")


def main(path):
    import sys

    import pytest

    # Parallel does not work on darwin, gets RuntimeError: context has already been set
    # because pytest-parallel changes the context from `spawn` to `fork`

    args = ["-p", "no:parallel", "-E", "release"]

    if len(sys.argv) > 1 and sys.argv[1] == "--no-debug":
        args += ["-o", "log_cli=False"]
    else:
        logging.basicConfig(level=logging.DEBUG)
        args += ["-o", "log_cli=True"]

    args += [path]

    sys.exit(pytest.main(args))
