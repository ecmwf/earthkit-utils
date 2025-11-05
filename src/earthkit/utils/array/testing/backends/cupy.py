from functools import cached_property

import array_api_compat

from .unknown import UnknownArrayBackend


class CupyBackend(UnknownArrayBackend):
    name = "cupy"
    module_name = "cupy"

    def _make_sample(self):
        import cupy

        return cupy.ones(2)

    def match_namespace(self, xp):
        return array_api_compat.is_cupy_namespace(xp)

    @cached_property
    def namespace(self):
        """Return the patched version of the array-api-compat numpy namespace."""
        from earthkit.utils.array.namespace.cupy import PatchedCupyNamespace

        return PatchedCupyNamespace()

    @cached_property
    def compat_namespace(self):
        """Return the array-api-compat cupy namespace."""
        import array_api_compat.cupy as xp

        return xp

    @cached_property
    def raw_namespace(self):
        import cupy

        return cupy

    def from_numpy(self, v, **kwargs):
        return self.from_other(v)

    def to_numpy(self, v):
        return v.get()

    def from_other(self, v, **kwargs):
        import cupy as cp

        return cp.array(v, **kwargs)

    @cached_property
    def _dtypes(self):
        import cupy as cp

        return {"float64": cp.float64, "float32": cp.float32}
