# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Function decorators and wrappers for use in the downstream EarthKit packages."""

from earthkit.utils.decorators._dispatch import dispatch
from earthkit.utils.decorators.experimental import ExperimentalWarning, experimental
from earthkit.utils.decorators.format_handlers import format_handler
from earthkit.utils.decorators.thread_handlers import thread_safe_cached_property
from earthkit.utils.decorators.xarray_ufunc import xarray_ufunc

__all__ = [
    "ExperimentalWarning",
    "experimental",
    "thread_safe_cached_property",
    "format_handler",
    "dispatch",
    "xarray_ufunc",
]
