# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Function decorators and wrappers for use in the downstream EarthKit packages.
"""

from .dispatch import dispatch
from .experimental import ExperimentalWarning
from .experimental import experimental
from .format_handlers import format_handler
from .thread_handlers import thread_safe_cached_property
from .xarray_ufunc import xarray_ufunc

__all__ = [
    "ExperimentalWarning",
    "experimental",
    "thread_safe_cached_property",
    "format_handler",
    "dispatch",
    "xarray_ufunc",
]
