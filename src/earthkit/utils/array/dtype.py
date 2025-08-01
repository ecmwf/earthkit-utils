# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from .backend import _BACKENDS


def to_numpy_dtype(dtype, default=None):
    for b in _BACKENDS:
        v = b.to_numpy_dtype(dtype)
        if v is not None:
            return v
    return default
