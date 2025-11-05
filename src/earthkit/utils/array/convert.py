from .array_namespace import array_namespace
from .converter import FromCupyConverter
from .converter import FromJaxConverter
from .converter import FromNumpyConverter
from .converter import FromTorchConverter
from .converter import FromUnknownConverter
from .namespace import PatchedCupyNamespace
from .namespace import PatchedNumpyNamespace
from .namespace import PatchedTorchNamespace
from .namespace import UnknownPatchedNamespace

converters = {
    "numpy": FromNumpyConverter,
    "cupy": FromCupyConverter,
    "torch": FromTorchConverter,
    "jax": FromJaxConverter,
}

namespaces = {
    "numpy": PatchedNumpyNamespace,
    "cupy": PatchedCupyNamespace,
    "torch": PatchedTorchNamespace,
}


def _get_array_name(xp):
    name = xp.__name__
    if "jax" in name:
        return "jax"
    elif "numpy" in name:
        return "numpy"
    elif "cupy" in name:
        return "cupy"
    elif "torch" in name:
        return "torch"
    else:
        return None


def _get_converter(source_array_backend):
    if isinstance(source_array_backend, UnknownPatchedNamespace):
        return converters.get(_get_array_name(source_array_backend), FromUnknownConverter)
    elif isinstance(source_array_backend, str):
        return converters[source_array_backend]
    else:
        raise ValueError(f"Unknown array backend: {source_array_backend}")


def _get_namespace(array_backend):
    if isinstance(array_backend, str):
        return namespaces[array_backend]()
    else:
        return namespaces.get(_get_array_name(array_backend), UnknownPatchedNamespace)(array_backend)


def convert(array, *, device=None, array_backend=None, **kwargs):
    """
    Return a copy/view of array moved to device.

    Parameters
    ----------
    array : array
        The array to be moved to the specified device.
    device : backend-specific device spec or str
        The device to which the array should be moved. For example,
        "cpu", "cuda:0", etc.
    array_backend : str or ArrayBackend
        The backend to use for the conversion. If None, the following logic
        is applied:
        - if the device is "cpu", it will use the numpy backend
        - otherwise it will use the backend of the array ``v``, but if that
          backend is numpy, it will use the cupy backend.
    *args, **kwargs : forwarded to the underlying backend call
    """

    # TODO: dtype conversion support also?

    if array_backend is None and device is None:
        return array

    source_xp = array_namespace(array)
    source_name = _get_array_name(source_xp)

    if array_backend is None:
        if device == "cpu":
            array_backend = PatchedNumpyNamespace()
            device = None
        elif source_name == "numpy":  # and device != "cpu" -> handled above
            array_backend = PatchedCupyNamespace()

    if array_backend is not None:
        target_xp = _get_namespace(array_backend)
        converter = _get_converter(source_xp)
        converter_instance = converter(target_xp)
        target_name = _get_array_name(target_xp)
        # TODO: decide if we want to pass device here, or later
        # currently, do it later
        array = converter_instance.to(array, target_name)

    if device is not None:
        xp = array_namespace(array)
        # TODO: add this method to patched namespaces
        array = xp.to_device(array, device=device, **kwargs)

    return array
