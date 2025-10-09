from functools import cached_property

import array_api_compat

from .abstract import ArrayBackend


class TorchBackend(ArrayBackend):
    name = "torch"
    module_name = "torch"

    def _make_sample(self):
        import torch

        return torch.ones(2)

    def match_namespace(self, xp):
        return array_api_compat.is_torch_namespace(xp)

    @cached_property
    def namespace(self):
        """Return the patched version of the array-api-compat torch namespace."""
        from earthkit.utils.array.namespace.torch import PatchedTorchNamespace

        return PatchedTorchNamespace()

    @cached_property
    def compat_namespace(self):
        """Return the array-api-compat torch namespace."""
        import array_api_compat.torch as xp

        return xp

    @cached_property
    def raw_namespace(self):
        import torch

        return torch

    def to_numpy(self, v):
        return v.cpu().numpy()

    def from_numpy(self, v):
        import torch

        return torch.from_numpy(v)

    def from_other(self, v, **kwargs):
        import torch

        return torch.tensor(v, **kwargs)

    @cached_property
    def _dtypes(self):
        import torch

        return {"float64": torch.float64, "float32": torch.float32}
