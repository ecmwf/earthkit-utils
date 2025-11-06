from .cupy import CupyBackend
from .jax import JaxBackend
from .numpy import NumpyBackend
from .torch import TorchBackend
from .unknown import UnknownArrayBackend

_BACKENDS = {
    "numpy": NumpyBackend(),
    "cupy": CupyBackend(),
    "torch": TorchBackend(),
    "jax": JaxBackend(),
}

_NUMPY_BACKEND = _BACKENDS["numpy"]
_CUPY_BACKEND = _BACKENDS["cupy"]
_TORCH_BACKEND = _BACKENDS["torch"]
_JAX_BACKEND = _BACKENDS["jax"]

_DEFAULT_BACKEND = _NUMPY_BACKEND
