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
    """Infer the number of outputs of ``func`` from its return annotation.

    Returns the number of elements for a fixed-length ``tuple[...]`` annotation. In all
    other cases (no annotation, non-tuple annotation, variable-length tuple or a signature
    that cannot be inspected) 1 is returned.
    """
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


def _normalise_attrs(attrs, count, is_multi_output):
    """Return a list of ``count`` attribute dicts (or None) from the user specified ``attrs``.

    ``attrs`` can be None, a dict (used for all outputs) or, for multi-output functions,
    a list/tuple with one dict (or None) per output. Values set to None remove the
    corresponding attribute from the result.
    """
    if attrs is None:
        return [None] * count
    if isinstance(attrs, dict):
        return [attrs] * count
    if is_multi_output and isinstance(attrs, (tuple, list)):
        if len(attrs) != count:
            raise ValueError("Length of 'variables' and 'attrs' must match for multi-output functions.")
        if all(e is None or isinstance(e, dict) for e in attrs):
            return list(attrs)
    if is_multi_output:
        raise ValueError("For multi-output functions, 'attrs' must be a dict or a list/tuple of dicts.")
    raise ValueError("For single-output functions, 'attrs' must be a dict.")


def _resolve_param_info(name, raise_on_unknown, attrs=None):
    """Return the variable name and attributes for the output described by parameter ``name``.

    Parameters
    ----------
    name : str
        Name of the parameter in the parameter database.
    raise_on_unknown : bool
        If True, raise a ValueError when the parameter has no variable name. Note that an
        unknown ``name`` currently raises a KeyError from the parameter database first.
    attrs : dict, optional
        Extra attributes to add to the ones defined by the parameter. They take precedence
        over the parameter metadata. Values set to None remove the attribute from the result.

    Returns
    -------
    tuple
        The variable name and the dict of attributes. Metadata with a None value is omitted.
    """
    param_name, param_attrs = PARAMS.xarray_variable_and_metadata(name)
    param_attrs = {k: v for k, v in param_attrs.items() if v is not None}
    if attrs is not None:
        param_attrs.update(attrs)
    if not param_name and raise_on_unknown:
        raise ValueError(f"Unknown output parameter '{name}' specified in xarray_ufunc_kwargs")
    return param_name, param_attrs


def _apply_param_metadata(da, param_name, param_attrs):
    """Rename ``da`` to ``param_name`` and update its attributes with ``param_attrs``.

    Attributes with a None value in ``param_attrs`` are removed. The output metadata keys
    (``long_name``, ``standard_name`` and ``units``) that are not set in ``param_attrs`` are
    also removed, so that they are not inherited from the input. ``da`` is returned unchanged
    when it is not a DataArray or ``param_name`` is None.
    """
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
    """Apply ``func`` to xarray objects with :func:`xarray.apply_ufunc`.

    Parameters
    ----------
    func : callable
        The function to apply. When it returns a fixed-length tuple (as given by its return
        annotation) the result is a tuple of the same length.
    *args
        Positional arguments passed to ``func``. They are expected to be xarray objects.
    xarray_ufunc_kwargs : dict, optional
        Options controlling the call. The following keys are supported:

        - "xarray_apply_ufunc_kwargs": dict, keyword arguments passed to
          :func:`xarray.apply_ufunc`. They override the defaults (``dask="parallelized"``,
          ``keep_attrs=True``). When missing, ``output_dtypes`` is inferred from the return
          annotation of ``func`` and, for multiple outputs, the core dimensions are taken
          from the inputs.
        - "variables": str or list/tuple of str, name(s) of the parameter(s) in the parameter
          database used to set the name and the ``long_name``, ``standard_name`` and
          ``units`` attributes of the result. Use a list/tuple with one item per output
          for multi-output functions. An empty item leaves that output untouched.
        - "attrs": dict or list/tuple of dicts, extra attributes added to the result. A single
          dict is applied to all outputs; a list/tuple must have one item (a dict or None)
          per output. Values set to None remove the attribute. Requires "variables".
        - "raise_on_unknown": bool, raise a ValueError when a parameter has no variable name
          (default: True). Note that a parameter missing from the parameter database raises
          a KeyError.
    **kwargs
        Keyword arguments passed to ``func``.

    Returns
    -------
    xarray.DataArray or tuple of xarray.DataArray
        The result of ``func``.

    Raises
    ------
    ValueError
        If "attrs" is specified without "variables" or if it has an invalid type or length.
    KeyError
        If a parameter in "variables" is not in the parameter database.
    """
    import xarray as xr

    if xarray_ufunc_kwargs is None:
        xarray_ufunc_kwargs = {}

    # kwargs for xarray.apply_ufunc
    xarray_apply_ufunc_kwargs = xarray_ufunc_kwargs.pop("xarray_apply_ufunc_kwargs", {})

    # determine the metadata to set on the resulting object(s). "variables" can either be a
    # single parameter name (single-output functions) or a tuple/list of parameter names
    # (multi-output functions), one per output in the same order.
    param_lookup_name = xarray_ufunc_kwargs.get("variables")
    attrs = xarray_ufunc_kwargs.get("attrs")
    param_info = None
    if param_lookup_name:
        raise_on_unknown = xarray_ufunc_kwargs.get("raise_on_unknown", True)
        is_multi_output = isinstance(param_lookup_name, (tuple, list))
        names = list(param_lookup_name) if is_multi_output else [param_lookup_name]
        attrs_list = _normalise_attrs(attrs, len(names), is_multi_output)
        param_info = [
            _resolve_param_info(name, raise_on_unknown, attrs=e) if name else (None, {})
            for name, e in zip(names, attrs_list)
        ]
    elif attrs is not None:
        raise ValueError("'attrs' in xarray_ufunc_kwargs requires 'variables' to be specified")

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
