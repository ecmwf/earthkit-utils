# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from .thread_handlers import thread_safe_cached_property

try:
    from earthkit.data.decorators import format_handler
except ImportError:
    from .format_handler import format_handler

__all__ = [
    "thread_safe_cached_property",
    "format_handler",
]
