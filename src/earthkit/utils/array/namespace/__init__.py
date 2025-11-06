from .cupy import PatchedCupyNamespace
from .numpy import PatchedNumpyNamespace
from .torch import PatchedTorchNamespace
from .unknown import UnknownPatchedNamespace

_NAMESPACES = {
    "numpy": PatchedNumpyNamespace(),
    "cupy": PatchedCupyNamespace(),
    "torch": PatchedTorchNamespace(),
}

_NUMPY_NAMESPACE = _NAMESPACES["numpy"]
_CUPY_NAMESPACE = _NAMESPACES["cupy"]
_TORCH_NAMESPACE = _NAMESPACES["torch"]

_DEFAULT_NAMESPACE = _NUMPY_NAMESPACE
