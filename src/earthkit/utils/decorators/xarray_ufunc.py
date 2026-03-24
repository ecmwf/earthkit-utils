# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
from __future__ import annotations


def _infer_output_count(func) -> int:
    try:
        import inspect
        from typing import get_args
        from typing import get_origin
    except ImportError:
        return 1

    try:
        annotation = inspect.signature(func).return_annotation
    except (ValueError, TypeError):
        return 1

    if annotation is inspect.Signature.empty:
        return 1

    origin = get_origin(annotation)
    if origin is tuple:
        args = get_args(annotation)
        if args and args[-1] is not Ellipsis:
            return len(args)
    return 1


def xarray_ufunc(func, *args, **kwargs):
    import xarray as xr

    xarray_ufunc_kwargs = kwargs.pop("xarray_ufunc_kwargs", None) or {}
    merged = {
        "dask": "parallelized",
        "keep_attrs": True,
    }
    if xarray_ufunc_kwargs:
        merged.update(xarray_ufunc_kwargs)

    if "output_dtypes" not in merged:
        output_count = _infer_output_count(func)
        merged["output_dtypes"] = [float] * output_count

    if "output_core_dims" not in merged and len(merged["output_dtypes"]) > 1:
        output_core_dims = [args[0].dims for _ in merged["output_dtypes"]]
        merged["output_core_dims"] = output_core_dims

    if "input_core_dims" not in merged and len(merged["output_dtypes"]) > 1:
        input_core_dims = [x.dims for x in args]
        merged["input_core_dims"] = input_core_dims

    return xr.apply_ufunc(
        func,
        *args,
        kwargs=kwargs,
        **merged,
    )
