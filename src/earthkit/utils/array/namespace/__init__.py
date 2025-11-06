from .cupy import PatchedCupyNamespace
from .numpy import PatchedNumpyNamespace
from .torch import PatchedTorchNamespace
from .unknown import UnknownPatchedNamespace

NAMESPACES = {
    "numpy": PatchedNumpyNamespace(),
    "cupy": PatchedCupyNamespace(),
    "torch": PatchedTorchNamespace(),
}

DEFAULT_NAMESPACE = NAMESPACES["numpy"]
