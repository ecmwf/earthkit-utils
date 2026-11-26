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


class PatchedCupyNamespace(UnknownPatchedNamespace):
    def __init__(self) -> None:
        super().__init__(None)

    @thread_safe_cached_property
    def xp(self) -> Any:
        import array_api_compat.cupy as cp

        return cp

    @property
    def _earthkit_array_namespace_name(self) -> str:
        return "cupy"

    def polyval(self, *args: Any, **kwargs: Any) -> Any:
        from cupy.polynomial.polynomial import polyval

        return polyval(*args, **kwargs)

    def percentile(self, a: Any, q: Any, axis: int | None = None) -> Any:
        return self.xp.percentile(a, q, axis=axis)

    def quantile(self, a: Any, q: Any, axis: int | None = None) -> Any:
        return self.xp.quantile(a, q, axis=axis)

    def histogram2d(self, x: Any, y: Any, *, bins: int | Sequence[int] = 10) -> Any:
        return self.xp.histogram2d(x, y, bins=bins)

    def histogramdd(self, x: Any, *, bins: int | Sequence[int] = 10) -> Any:
        return self.xp.histogramdd(x, bins=bins)

    def asarray(self, *args: Any, **kwargs: Any) -> Any:
        device = kwargs.pop("device", None)
        if device is not None:
            if isinstance(device, str) and device.startswith("cuda"):
                _, _, idx = device.partition(":")
                dev_id = int(idx) if idx else 0
            else:
                dev_id = device
            with self.xp.cuda.Device(dev_id):
                return self.xp.asarray(*args, **kwargs)
        else:
            return self.xp.asarray(*args, **kwargs)

    def to_device(self, x: Any, device: Any, **kwargs: Any) -> Any:
        return self.asarray(x, device=device, **kwargs)

    def rad2deg(self, x: Any) -> Any:
        return self.xp.rad2deg(x)

    def deg2rad(self, x: Any) -> Any:
        return self.xp.deg2rad(x)

    def choice(self, a: Any, size: int, replace: bool = True, generator: Any = None) -> Any:
        rng = self.xp.default_rng() if generator is None else generator
        return rng.choice(a, size=size, replace=replace)
