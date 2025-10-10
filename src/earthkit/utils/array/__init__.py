# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from .array_backend import array_namespace
from .array_backend import array_namespace_xarray
from .backend import _BACKENDS  # noqa: F401
from .backend import _CUPY
from .backend import _DEFAULT_BACKEND
from .backend import _JAX
from .backend import _NUMPY
from .backend import _TORCH
from .backend import get_backend
from .convert import array_to_numpy
from .convert import convert_array
from .device import to_device  # noqa: F401
