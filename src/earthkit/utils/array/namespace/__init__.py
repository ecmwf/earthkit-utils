# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from .cupy import PatchedCupyNamespace
from .jax import PatchedJaxNamespace
from .numpy import PatchedNumpyNamespace
from .torch import PatchedTorchNamespace
from .unknown import UnknownPatchedNamespace

_NAMESPACES = {
    "numpy": PatchedNumpyNamespace(),
    "cupy": PatchedCupyNamespace(),
    "torch": PatchedTorchNamespace(),
    "jax": PatchedJaxNamespace(),
}

_NUMPY_NAMESPACE = _NAMESPACES["numpy"]
_CUPY_NAMESPACE = _NAMESPACES["cupy"]
_TORCH_NAMESPACE = _NAMESPACES["torch"]
_JAX_NAMESPACE = _NAMESPACES["jax"]

_DEFAULT_NAMESPACE = _NUMPY_NAMESPACE

# for backwards compatibility
_NAMESPACES["pytorch"] = _TORCH_NAMESPACE
