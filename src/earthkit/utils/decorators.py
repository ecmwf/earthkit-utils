# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import threading


class thread_safe_cached_property:
    """A thread-safe cached property decorator.

    It was implemented because a the functools.cached_property is not thread-safe
    from Python 3.12 onwards.

    Parameters
    ----------
    method: property method
        The property method to be decorated.

    The :obj:`__get__` method of the decorator only runs on lookups. On first call
    it gets the underlying property's value and stores it as the hidden ``name`` attribute
    of the instance it was called on. Subsequent calls return the cached value, i.e. the
    hidden ``name`` attribute.
    """

    def __init__(self, method):
        self.method = method
        self.name = f"_c_{method.__name__}"
        self.lock = threading.Lock()

    def __get__(self, instance, owner=None):
        if instance is None:
            return self

        # not all objects have __dict__ (e.g. class defines slots)
        try:
            cache = instance.__dict__
        except AttributeError:
            msg = f"No '__dict__' is available on {type(instance).__name__!r}"
            raise TypeError(msg) from None

        # avoid using hasattr/getattr as they may be overridden in the instance
        # and may have side effects (infinite recursion etc.)
        if self.name in cache:
            return cache[self.name]

        with self.lock:
            if self.name in cache:
                return cache[self.name]
            value = self.method(instance)
            cache[self.name] = value
            return value
