# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

from typing import Any


class FromUnknownConverter:
    def __init__(self, xp_target: Any) -> None:
        # TODO: check if we ever will need source also
        # self.xp_source = xp_source
        self.xp_target = xp_target

    def to(self, array: Any, target_backend: str, **kwargs: Any) -> Any:
        method_name = f"to_{target_backend}"
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            return method(array, **kwargs)
        else:
            return self.to_unknown(array, **kwargs)

    def _default_convert(self, xp: Any, array: Any, **kwargs: Any) -> Any:
        """Default attempt to convert from other backend to this backend.

        Process is the following:
        1. Attempts to use dlpack if available.
        2. Tries to convert using xp.asarray naively.
        3. Tries to convert using xp.array naively.
        4. Converts to numpy, then tries to convert using xp.asarray on the new numpy array.
        5. Converts to numpy, then tries to convert using xp.array on the new numpy array.
        """
        if hasattr(array, "__dlpack__") and hasattr(xp, "from_dlpack"):
            return xp.from_dlpack(array, **kwargs)
        else:
            try:
                return xp.asarray(array, **kwargs)
            except Exception:
                try:
                    return xp.array(array, **kwargs)
                except Exception:
                    numpy_array = self.to_numpy(array)
                    try:
                        return xp.asarray(numpy_array, **kwargs)
                    except Exception:
                        return xp.array(numpy_array, **kwargs)

    def to_unknown(self, array: Any, **kwargs: Any) -> Any:
        return self._default_convert(self.xp_target, array, **kwargs)

    def to_numpy(self, array: Any, **kwargs: Any) -> Any:
        from earthkit.utils.array.namespace import _NUMPY_NAMESPACE

        return self._default_convert(_NUMPY_NAMESPACE, array, **kwargs)

    def to_torch(self, array: Any, **kwargs: Any) -> Any:
        from earthkit.utils.array.namespace import _TORCH_NAMESPACE

        return self._default_convert(_TORCH_NAMESPACE, array, **kwargs)

    def to_cupy(self, array: Any, **kwargs: Any) -> Any:
        from earthkit.utils.array.namespace import _CUPY_NAMESPACE

        return self._default_convert(_CUPY_NAMESPACE, array, **kwargs)

    def to_jax(self, array: Any, **kwargs: Any) -> Any:
        return self._default_convert(self.xp_target, array, **kwargs)
