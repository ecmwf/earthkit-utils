# (C) Copyright 2023- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import importlib
import pkgutil
import threading
import os

_lock = threading.RLock()

__path__ = pkgutil.extend_path(__path__, __name__)

modules = set()

for path in __path__:
    if not os.path.isdir(path):
        continue
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path) and not name.startswith('_') and name not in {"importlib", "pkgutil", "threading", "os"}:
            spec = importlib.util.find_spec(f"{__name__}.{name}")
            if spec is not None:
                modules.add(name)

__all__ = tuple(sorted(modules))

print(__all__)
print(__path__)

try:
    from earthkit._version import __version__
except:
    __version__ = -1

def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"No such submodule: {name}")

    with _lock:
        if name in globals():
            return globals()[name]
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
    return mod


def __dir__():
    return tuple(globals()) + __all__
