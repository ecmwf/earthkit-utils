# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from .unknown import UnknownPatchedNamespace


class PatchedCupyNamespace(UnknownPatchedNamespace):

    def __init__(self, xp=None):
        import array_api_compat.cupy as cp

        super().__init__(cp)

    @property
    def _earthkit_array_namespace_name(self):
        return "cupy"

    def polyval(self, *args, **kwargs):
        from cupy.polynomial.polynomial import polyval

        return polyval(*args, **kwargs)

    def asarray(self, *args, **kwargs):
        device = kwargs.pop("device", None)
        if device is not None:
            if isinstance(device, str) and device.startswith("cuda"):
                _, _, idx = device.partition(":")
                dev_id = int(idx) if idx else 0
            else:
                dev_id = device
            with self._xp.cuda.Device(dev_id):
                self._xp.asarray(*args, **kwargs)
        else:
            self._xp.asarray(*args, **kwargs)
