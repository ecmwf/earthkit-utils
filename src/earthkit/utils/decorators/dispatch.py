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

    # from earthkit.utils.array import array_namespace

    # try:
    #     xp = array_namespace(obj)
    # except (KeyError, TypeError):
    #     return False

    # try:
    #     return isinstance(obj, type(xp.asarray(obj)))
    # except Exception:
    #     return False


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


_DISPATCHERS = [XArrayDispatcher(), FieldListDispatcher(), ArrayDispatcher()]


def dispatch(
    func=None, match=0, xarray=True, fieldlist=True, array=False, default_dispatcher=ArrayDispatcher()
):
    """
    Decorator to dispatch function calls based on input data types.
    The dispatch will attempt to route the call to the appropriate
    implementation based on the type of the specified argument.
    The implementations are assumed to live in submodules named after the data
    type (e.g., .xarray, .fieldlist, .array) with the same function name as
    the toplevel function.

    This decorator can be used either without arguments:

        @dispatch
        def func(...):
            ...

    or with configuration arguments:

        @dispatch(match=1, array=True)
        def func(...):
            ...

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
    default_dispatcher: DataDispatcher or None
        The default dispatcher to use if no dispatchers match. If None, a TypeError is raised when no dispatchers match. Default is ArrayDispatcher().

    Returns
    -------
    function
        The decorated function with dispatching capability.
    """

    def _make_wrapper(f):
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
                try:
                    _matched = dispatcher.match(obj_to_check)
                except Exception as e:
                    LOG.debug(f"Dispatcher {dispatcher.__class__.__name__} failed to match due to error: {e}")
                    continue
                if _matched:
                    return dispatcher.dispatch(func.__name__, _module, *args, **kwargs)
            if default_dispatcher is None:
                raise TypeError(
                    f"No dispatcher matched for function {func.__name__} with argument {param_name} of type {type(obj_to_check)}, and no default dispatcher specified."
                )
            LOG.warning(
                f"No dispatcher matched for function {func.__name__} with argument {param_name} of type {type(obj_to_check)}. "
                f"Using default dispatcher {default_dispatcher.__class__.__name__}."
            )
            return default_dispatcher.dispatch(func.__name__, _module, *args, **kwargs)

        return wrapper

    if func is None:
        # Called as @dispatch(match=..., ...)
        def decorator(real_func):
            return _make_wrapper(real_func)

        return decorator

    # Called as @dispatch or dispatch(func, ...)
    return _make_wrapper(func)
