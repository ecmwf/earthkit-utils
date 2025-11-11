from .unknown import UnknownPatchedNamespace


class PatchedJaxNamespace(UnknownPatchedNamespace):

    def __init__(self):
        super().__init__(None)

    def _set_xp(self):
        import jax.numpy as jnp

        self._xp = jnp

    @property
    def _earthkit_array_namespace_name(self):
        return "jax"

    def percentile(self, a, q, axis=None):
        return self.xp.percentile(a, q, axis=axis)

    def quantile(self, a, q, axis=None):
        return self.xp.quantile(a, q, axis=axis)

    def devices(self):
        import jax

        return jax.devices()
