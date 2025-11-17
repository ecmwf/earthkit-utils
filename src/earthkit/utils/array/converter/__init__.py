# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from .cupy import FromCupyConverter
from .jax import FromJaxConverter
from .numpy import FromNumpyConverter
from .torch import FromTorchConverter
from .unknown import FromUnknownConverter

_CONVERTERS = {
    "numpy": FromNumpyConverter,
    "cupy": FromCupyConverter,
    "torch": FromTorchConverter,
    "jax": FromJaxConverter,
}

_NUMPY_CONVERTER = _CONVERTERS["numpy"]
_CUPY_CONVERTER = _CONVERTERS["cupy"]
_TORCH_CONVERTER = _CONVERTERS["torch"]
_JAX_CONVERTER = _CONVERTERS["jax"]

_DEFAULT_CONVERTER = _NUMPY_CONVERTER

# for backwards compatibility
_CONVERTERS["pytorch"] = _TORCH_CONVERTER
