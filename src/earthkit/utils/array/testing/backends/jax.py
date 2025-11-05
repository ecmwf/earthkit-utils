from functools import cached_property

import array_api_compat

from .unknown import UnknownArrayBackend


class JaxBackend(UnknownArrayBackend):
    name = "jax"
    module_name = "jax"

    def _make_sample(self):
        import jax.numpy as jarray

        return jarray.ones(2)

    def match_namespace(self, xp):
        return array_api_compat.is_jax_namespace(xp)

    @cached_property
    def namespace(self):
        """Return the of the array-api-compat jax namespace."""
        from earthkit.utils.array.namespace.unknown import UnknownPatchedNamespace

        return UnknownPatchedNamespace(self.compat_namespace)

    @cached_property
    def compat_namespace(self):
        # jnp is array-api-compliant (see jax-ml/jax#22818)
        return self.raw_namespace

    @cached_property
    def raw_namespace(self):
        import jax.numpy as jnp

        return jnp

    def to_numpy(self, v):
        import numpy as np

        return np.array(v)

    def from_numpy(self, v, **kwargs):
        return self.from_other(v, **kwargs)

    def from_other(self, v, **kwargs):
        import jax.numpy as jnp

        return jnp.array(v, **kwargs)

    @cached_property
    def _dtypes(self):
        import jax.numpy as jnp

        return {"float64": jnp.float64, "float32": jnp.float32}
