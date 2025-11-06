# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from .unknown import UnknownPatchedNamespace


class PatchedTorchNamespace(UnknownPatchedNamespace):

    def __init__(self):
        super().__init__(None)

    def _set_xp(self):
        import array_api_compat.torch as torch

        self._xp = torch

    @property
    def _earthkit_array_namespace_name(self):
        return "torch"

    def sign(self, x, *args, **kwargs):
        """Reimplement the sign function to handle NaNs.

        The problem is that torch.sign returns 0 for NaNs, but the array API
        standard requires NaNs to be propagated.
        """
        x = self.xp.asarray(x)
        r = self.xp.sign(x, *args, **kwargs)
        r = self.xp.asarray(r)
        r[self.xp.isnan(x)] = self.xp.nan
        return r

    def percentile(self, a, q, axis=None, **kwargs):
        return self._xp.quantile(a, q / 100, dim=axis, **kwargs)

    def size(self, x):
        """Return the size of an array."""
        x = self.xp.asarray(x)
        return x.numel()

    def shape(self, x):
        """Return the shape of an array."""
        x = self.xp.asarray(x)
        return tuple(x.shape)

    def to_device(self, x, device, **kwargs):
        return x.to(device, **kwargs)

    def devices(self):
        import torch

        devices = []
        if torch.cpu.is_available():
            for i in range(torch.cpu.device_count()):
                devices.append(f"cpu:{i}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
        if torch.backends.mps.is_available():
            for i in range(torch.mps.device_count()):
                devices.append(f"mps:{i}")
        if torch.backends.xpu.is_available():
            for i in range(torch.xpu.device_count()):
                devices.append(f"xpu:{i}")
        if torch.backends.mtia.is_available():
            for i in range(torch.mtia.device_count()):
                devices.append(f"mtia:{i}")

        return devices
