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
