# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from earthkit.utils.decorators import thread_safe_cached_property

from .unknown import UnknownPatchedNamespace


class PatchedJaxNamespace(UnknownPatchedNamespace):

    def __init__(self):
        super().__init__(None)

    @thread_safe_cached_property
    def xp(self):
        import jax.numpy as jnp

        return jnp

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
