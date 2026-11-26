# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from typing import Any

from earthkit.utils.array.converter.unknown import FromUnknownConverter


class FromTorchConverter(FromUnknownConverter):
    def __init__(self, xp_target: Any) -> None:
        super().__init__(xp_target)

    def to_numpy(self, array: Any, **kwargs: Any) -> Any:
        return array.cpu().numpy()

    def to_cupy(self, array: Any, **kwargs: Any) -> Any:
        # TODO: add device handling
        # (below only works if tensor device is cuda)
        import cupy as cp

        return cp.from_dlpack(array)

    def to_torch(self, array: Any, **kwargs: Any) -> Any:
        # TODO: add device handling?
        return array

    # def to_jax(self, array, **kwargs):
    #     return
