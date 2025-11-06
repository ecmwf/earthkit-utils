from functools import cached_property

import array_api_compat

from .unknown import UnknownArrayBackend


class NumpyBackend(UnknownArrayBackend):
    name = "numpy"
    module_name = "numpy"

    def _make_sample(self):

        import numpy as np

        return np.ones(2)

    def match_namespace(self, xp):
        return array_api_compat.is_numpy_namespace(xp)

    @cached_property
    def namespace(self):
        """Return the patched version of the array-api-compat numpy namespace."""
        from earthkit.utils.array.namespace import NAMESPACES

        return NAMESPACES["numpy"]

    @cached_property
    def compat_namespace(self):
        """Return the array-api-compat numpy namespace."""
        import array_api_compat.numpy as xp

        return xp

    @cached_property
    def raw_namespace(self):
        import numpy as np

        return np

    def to_numpy(self, v):
        return v

    def from_numpy(self, v, **kwargs):
        return v

    def from_other(self, v, **kwargs):
        import numpy as np

        if not kwargs and isinstance(v, np.ndarray):
            return v

        return np.array(v, **kwargs)

    def to_numpy_dtype(self, dtype):
        dtype = self.dtype_to_str(dtype)
        if dtype is None:
            return None
        else:
            return self.make_dtype(dtype)

    @cached_property
    def _dtypes(self):
        import numpy

        return {"float64": numpy.float64, "float32": numpy.float32}
