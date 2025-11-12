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
