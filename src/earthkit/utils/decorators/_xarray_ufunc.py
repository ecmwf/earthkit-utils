# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
from __future__ import annotations

from earthkit.utils.parameters._params import PARAMS

OUTPUT_METADATA_KEYS = ["long_name", "standard_name", "units"]


def _infer_output_count(func) -> int:
    import inspect
    from typing import get_args, get_origin

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


def _resolve_param_info(name, raise_on_unknown):
    param_name, param_attrs = PARAMS.xarray_variable_and_metadata(name)
    param_attrs = {k: v for k, v in param_attrs.items() if v is not None}
    if not param_name and raise_on_unknown:
        raise ValueError(f"Unknown output parameter '{name}' specified in xarray_ufunc_kwargs")
    return param_name, param_attrs


def _apply_param_metadata(da, param_name, param_attrs):
    import xarray as xr

    if isinstance(da, xr.DataArray) and param_name is not None:
        da = da.rename(param_name)
        for k, v in param_attrs.items():
            if v is not None:
                da.attrs.update({k: v})
            elif k in da.attrs:
                del da.attrs[k]

        for k in OUTPUT_METADATA_KEYS:
            if param_attrs.get(k) is None and k in da.attrs:
                del da.attrs[k]

    return da


def xarray_ufunc(func, *args, xarray_ufunc_kwargs=None, **kwargs):
    import xarray as xr

    if xarray_ufunc_kwargs is None:
        xarray_ufunc_kwargs = {}

    # kwargs for xarray.apply_ufunc
    xarray_apply_ufunc_kwargs = xarray_ufunc_kwargs.pop("xarray_apply_ufunc_kwargs", {})

    # determine the metadata to set on the resulting object(s). "variables" can either be a
    # single parameter name (single-output functions) or a tuple/list of parameter names
    # (multi-output functions), one per output in the same order.
    param_lookup_name = xarray_ufunc_kwargs.get("variables")
    param_info = None
    if param_lookup_name:
        raise_on_unknown = xarray_ufunc_kwargs.get("raise_on_unknown", True)
        is_multi_output = isinstance(param_lookup_name, (tuple, list))
        if is_multi_output:
            param_info = [
                _resolve_param_info(name, raise_on_unknown) if name else (None, {}) for name in param_lookup_name
            ]
        else:
            param_info = [_resolve_param_info(param_lookup_name, raise_on_unknown)] if param_lookup_name else []

    merged = {
        "dask": "parallelized",
        "keep_attrs": True,
    }
    merged.update(xarray_apply_ufunc_kwargs)

    if "output_dtypes" not in merged:
        output_count = _infer_output_count(func)
        merged["output_dtypes"] = [float] * output_count

    if "output_core_dims" not in merged and len(merged["output_dtypes"]) > 1:
        output_core_dims = [args[0].dims for _ in merged["output_dtypes"]]
        merged["output_core_dims"] = output_core_dims

    if "input_core_dims" not in merged and len(merged["output_dtypes"]) > 1:
        input_core_dims = [x.dims for x in args]
        merged["input_core_dims"] = input_core_dims

    r = xr.apply_ufunc(
        func,
        *args,
        kwargs=kwargs,
        **merged,
    )

    # set the name and attributes on the resulting DataArray(s) if applicable
    if param_info:
        if isinstance(r, tuple):
            r = tuple(
                _apply_param_metadata(da, param_name, param_attrs)
                for da, (param_name, param_attrs) in zip(r, param_info)
            )
        else:
            param_name, param_attrs = param_info[0]
            r = _apply_param_metadata(r, param_name, param_attrs)

    return r
