# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from earthkit.utils.array.namespace.unknown import UnknownPatchedNamespace
from earthkit.utils.decorators import thread_safe_cached_property


class PatchedNumpyNamespace(UnknownPatchedNamespace):
    def __init__(self) -> None:
        super().__init__(None)

    @thread_safe_cached_property
    def xp(self) -> Any:
        import array_api_compat.numpy as np

        return np

    @property
    def _earthkit_array_namespace_name(self) -> str:
        return "numpy"

    def polyval(self, *args: Any, **kwargs: Any) -> Any:
        from numpy.polynomial.polynomial import polyval

        return polyval(*args, **kwargs)

    def percentile(self, a: Any, q: Any, axis: int | None = None) -> Any:
        return self.xp.percentile(a, q, axis=axis)

    def quantile(self, a: Any, q: Any, axis: int | None = None) -> Any:
        return self.xp.quantile(a, q, axis=axis)

    def histogram2d(self, x: Any, y: Any, *, bins: int | Sequence[int] = 10) -> Any:
        return self.xp.histogram2d(x, y, bins=bins)

    def histogramdd(self, x: Any, *, bins: int | Sequence[int] = 10) -> Any:
        return self.xp.histogramdd(x, bins=bins)

    def isclose(self, x: Any, y: Any, *, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False) -> Any:
        return self.xp.isclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def allclose(self, x: Any, y: Any, *, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False) -> Any:
        return self.xp.allclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def rad2deg(self, x: Any) -> Any:
        return self.xp.rad2deg(x)

    def deg2rad(self, x: Any) -> Any:
        return self.xp.deg2rad(x)

    def choice(self, a: Any, size: int, replace: bool = True, generator: Any = None) -> Any:
        rng = self.xp.random.default_rng() if generator is None else generator
        return rng.choice(a, size=size, replace=replace)
