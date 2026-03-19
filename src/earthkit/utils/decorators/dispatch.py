# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
from __future__ import annotations

import sys
from abc import ABCMeta
from abc import abstractmethod
from functools import wraps
from importlib import import_module
from inspect import signature
from typing import Any

from earthkit.utils.array import array_namespace


def is_module_loaded(module_name):
    return module_name in sys.modules


def _is_xarray(obj: Any) -> bool:
    if not is_module_loaded("xarray"):
        return False

    try:
        import xarray as xr

        return isinstance(obj, (xr.DataArray, xr.Dataset))
    except (ImportError, RuntimeError, SyntaxError):
        return False


def _is_fieldlist(obj: Any) -> bool:
    if not is_module_loaded("earthkit.data"):
        return False

    try:
        from earthkit.data import FieldList

        return isinstance(obj, FieldList)
    except ImportError:
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
        try:
            xp = array_namespace(obj)
        except KeyError:
            return False

        try:
            xp.asarray(obj)
            return True
        except Exception:
            return False

    def dispatch(self, func, module, *args, **kwargs):
        module = import_module(module + ".array")
        return getattr(module, func)(*args, **kwargs)


_DISPATCHERS = [XArrayDispatcher(), FieldListDispatcher(), ArrayDispatcher()]


def dispatch(func, match=0, xarray=True, fieldlist=True, array=False):
    """
    Decorator to dispatch function calls based on input data types.
    The dispatch will attempt to route the call to the appropriate
    implementation based on the type of the specified argument.
    The implementations are assumed to live in submodules named after the data
    type (e.g., .xarray, .fieldlist, .array) with the same function name as
    the toplevel function.

    Parameters
    ----------
    func: function
        The toplevel function to be decorated.
    match: int or str
        The index or name of the argument to check for dispatching. Default is 0 (the first argument).
    xarray: bool
        Whether to include the xarray dispatcher. Default is True.
    fieldlist: bool
        Whether to include the FieldList dispatcher. Default is True.
    array: bool
        Whether to include the array dispatcher. Default is False.

    Returns
    -------
    function
        The decorated function with dispatching capability.
    """
    DISPATCHERS = []
    if xarray:
        DISPATCHERS.append(_DISPATCHERS[0])
    if fieldlist:
        DISPATCHERS.append(_DISPATCHERS[1])
    if array:
        DISPATCHERS.append(_DISPATCHERS[2])

    sig = signature(func)

    params = list(sig.parameters)
    if isinstance(match, int):
        try:
            param_name = params[match]
        except IndexError as e:
            raise ValueError(
                f"'match' index {match} is invalid for function {func.__name__} with  {len(params)} arguments"
            ) from e
    elif isinstance(match, str):
        if match in params:
            param_name = match
        else:
            raise ValueError(
                f"'match' parameter name {match} is not in the function signature of {func.__name__}"
            )
    else:
        raise TypeError(f"'match' must be an integer index or a string parameter name, got {type(match)}")

    @wraps(func)
    def wrapper(*args, **kwargs):
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        obj_to_check = bound_args.arguments[param_name]

        _module = ".".join(func.__module__.split(".")[:-1])
        for dispatcher in DISPATCHERS:
            if dispatcher.match(obj_to_check):
                return dispatcher.dispatch(func.__name__, _module, *args, **kwargs)
        raise TypeError(f"No matching dispatcher found for the input type: {type(obj_to_check)}")

    return wrapper
