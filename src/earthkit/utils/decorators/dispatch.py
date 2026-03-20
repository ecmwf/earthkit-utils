# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
from __future__ import annotations

import logging
import sys
from abc import ABCMeta
from abc import abstractmethod
from functools import wraps
from importlib import import_module
from inspect import signature
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    import xarray as xr  # noqa: F401

    from earthkit.data import FieldList  # noqa: F401

LOG = logging.getLogger(__name__)


def is_module_loaded(module_name):
    return module_name in sys.modules


def _is_xarray(obj: Any) -> bool:
    if not is_module_loaded("xarray"):
        return False

    try:
        import xarray as xr

    except ImportError:
        return False

    return isinstance(obj, (xr.DataArray, xr.Dataset))


def _is_fieldlist(obj: Any) -> bool:
    if not is_module_loaded("earthkit.data"):
        return False

    try:
        from earthkit.data import FieldList

    except ImportError:
        return False

    return isinstance(obj, FieldList)


def _is_array(obj: Any) -> bool:
    import array_api_compat

    return array_api_compat.is_array_api_obj(obj)


def is_array_like(obj: Any) -> bool:
    """Check if the object is array-like, i.e., if it belongs to a known array namespace or is a scalar or list that can be converted to an array."""
    import numpy as np

    try:
        np.asarray(obj)
        return True
    # TODO: Improve this exception handling
    except Exception:
        return False


class DataDispatcher(metaclass=ABCMeta):
    """
    A dispatcher class to route function calls based on input data types.
    """

    @staticmethod
    @abstractmethod
    def match(obj: Any) -> bool:
        pass

    @abstractmethod
    def dispatch(self, func: str, module: str, *args: Any, **kwargs: Any) -> Any:
        pass


class XArrayDispatcher(DataDispatcher):
    @staticmethod
    def match(obj: Any) -> bool:
        return _is_xarray(obj)

    def dispatch(self, func, module, *args, **kwargs):
        module = import_module(module + ".xarray")
        return getattr(module, func)(*args, **kwargs)


class FieldListDispatcher(DataDispatcher):
    @staticmethod
    def match(obj: Any) -> bool:
        return _is_fieldlist(obj)

    def dispatch(self, func, module, *args, **kwargs):
        module = import_module(module + ".fieldlist")
        return getattr(module, func)(*args, **kwargs)


class ArrayDispatcher(DataDispatcher):
    @staticmethod
    def match(obj: Any) -> bool:
        return _is_array(obj)

    def dispatch(self, func, module, *args, **kwargs):
        module = import_module(module + ".array")
        return getattr(module, func)(*args, **kwargs)


class ArrayLikeDispatcher(ArrayDispatcher):
    @staticmethod
    def match(obj: Any) -> bool:
        return is_array_like(obj)


def dispatch(
    func: callable,
    match: int | str = 0,
    xarray: bool = True,
    fieldlist: bool = True,
    array: bool = False,
    array_like: None | bool = None,
):
    """
    Decorator to dispatch function calls based on input data types.
    The dispatch will attempt to route the call to the appropriate
    implementation based on the type of the specified argument.
    The implementations are assumed to live in submodules named after the data
    type (e.g., .xarray, .fieldlist, .array) with the same function name as
    the toplevel function.

    This wrapper should be applied inline as:

        def func(...):
            return dispatch(func, match=..., xarray=..., fieldlist=..., array=..., array_like=...)(...)

    Parameters
    ----------
    func: function
        The toplevel function to be decorated. If None, a decorator factory
        is returned that expects the function to decorate.
    match: int or str
        The index or name of the argument to check for dispatching. Default is 0 (the first argument).
    xarray: bool
        Whether to include the xarray dispatcher. Default is True.
    fieldlist: bool
        Whether to include the FieldList dispatcher. Default is True.
    array: bool
        Whether to include the array dispatcher. Default is False.
    array_like: bool or None
        Whether to include the array-like dispatcher.
        If None (default), it will be set to the same value as `array`.

    Returns
    -------
    function
        The decorated function with dispatching capability.
    """

    if array_like is None:
        if array is True:
            array_like = True
        else:
            array_like = False

    def _make_wrapper(_func):
        DISPATCHERS = []
        if xarray:
            DISPATCHERS.append(XArrayDispatcher())
        if fieldlist:
            DISPATCHERS.append(FieldListDispatcher())
        if array:
            DISPATCHERS.append(ArrayDispatcher())
        if array_like:
            DISPATCHERS.append(ArrayLikeDispatcher())

        sig = signature(_func)

        params = list(sig.parameters)
        if isinstance(match, int):
            try:
                param_name = params[match]
            except IndexError as e:
                raise ValueError(
                    f"'match' index {match} is invalid for function {_func.__name__} with {len(params)} arguments"
                ) from e
        elif isinstance(match, str):
            if match in params:
                param_name = match
            else:
                raise ValueError(
                    f"'match' parameter name {match} is not in the function signature of {_func.__name__}"
                )
        else:
            raise TypeError(f"'match' must be an integer index or a string parameter name, got {type(match)}")

        @wraps(_func)
        def wrapper(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            obj_to_check = bound_args.arguments[param_name]

            module_name = _func.__module__
            parent_module, sep, _ = module_name.rpartition(".")
            _module = parent_module if parent_module else module_name
            for dispatcher in DISPATCHERS:
                try:
                    _matched = dispatcher.match(obj_to_check)
                except Exception as e:
                    LOG.debug(f"Dispatcher {dispatcher.__class__.__name__} failed to match due to error: {e}")
                    continue
                if _matched:
                    return dispatcher.dispatch(_func.__name__, _module, *args, **kwargs)
            raise TypeError(
                f"No dispatcher matched for function {_func.__name__} with argument {param_name} of type {type(obj_to_check)}, and no default dispatcher specified."
            )

        return wrapper

    # Called as dispatch(func, ...)
    return _make_wrapper(func)
