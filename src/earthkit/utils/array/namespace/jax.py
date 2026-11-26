# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from typing import Any

from earthkit.utils.array.namespace.unknown import UnknownPatchedNamespace
from earthkit.utils.decorators import thread_safe_cached_property


class PatchedJaxNamespace(UnknownPatchedNamespace):
    def __init__(self) -> None:
        super().__init__(None)

    @thread_safe_cached_property
    def xp(self) -> Any:
        import jax.numpy as jnp

        return jnp

    @property
    def _earthkit_array_namespace_name(self) -> str:
        return "jax"

    def percentile(self, a: Any, q: Any, axis: int | None = None) -> Any:
        return self.xp.percentile(a, q, axis=axis)

    def quantile(self, a: Any, q: Any, axis: int | None = None) -> Any:
        return self.xp.quantile(a, q, axis=axis)

    def rad2deg(self, x: Any) -> Any:
        return self.xp.rad2deg(x)

    def deg2rad(self, x: Any) -> Any:
        return self.xp.deg2rad(x)

    def choice(self, a: Any, size: int, replace: bool = True, generator: Any = None) -> Any:
        from random import randint

        import jax.random

        generator = jax.random.PRNGKey(randint(0, 10000)) if generator is None else generator
        return jax.random.choice(generator, a, shape=(size,), replace=replace)
