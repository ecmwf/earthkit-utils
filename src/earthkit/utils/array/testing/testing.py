# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

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
        xp_devices = xp.__array_namespace_info__().devices()
        if name == "torch":
            xp_devices = [x for x in xp_devices if x.type != "meta"]
        n_devices = len(xp_devices)
        namespaces += [xp] * n_devices
        devices += xp_devices
    return list(zip(namespaces, devices))


NAMESPACE_DEVICES = _get_namespace_devices(ARRAY_NAMES)
