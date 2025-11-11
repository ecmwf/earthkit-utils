from .array_namespace import _get_array_name
from .array_namespace import array_namespace as array_namespace_func
from .converter import _CONVERTERS
from .converter import FromUnknownConverter
from .namespace import _CUPY_NAMESPACE
from .namespace import _DEFAULT_NAMESPACE
from .namespace import UnknownPatchedNamespace


def _get_converter(source_array_namespace):
    if isinstance(source_array_namespace, UnknownPatchedNamespace):
        return _CONVERTERS.get(_get_array_name(source_array_namespace), FromUnknownConverter)
    elif isinstance(source_array_namespace, str):
        return _CONVERTERS[source_array_namespace]
    else:
        raise ValueError(f"Unknown array backend: {source_array_namespace._earthkit_array_namespace_name}")


def convert(array, *, device=None, array_namespace=None, **kwargs):
    """
    Return a copy/view of a converted array.

    Parameters
    ----------
    array : array
        The array to convert.
    device : array namespace-specific device spec or str
        The device to which the array should be moved. For example,
        "cpu", "cuda:0", etc.
    array_namespace : str or array namespace
        The array namespace to use for the conversion. If None, the following logic
        is applied:
        - if the device is "cpu", it will use numpy
        - otherwise it will use the namespace of the array ``v``, but if that
          backend is numpy, it will use the cupy backend.
    **kwargs : forwarded to the underlying call
    """

    # TODO: dtype conversion support also?

    if array_namespace is None and device is None:
        return array

    source_xp = array_namespace_func(array)
    source_name = _get_array_name(source_xp)

    if array_namespace is None:
        if device == "cpu":
            array_namespace = _DEFAULT_NAMESPACE
            device = None
        elif source_name == "numpy":  # and device != "cpu" -> handled above
            array_namespace = _CUPY_NAMESPACE

    if array_namespace is not None:
        target_xp = array_namespace_func(array_namespace)
        converter = _get_converter(source_xp)
        converter_instance = converter(target_xp)
        target_name = _get_array_name(target_xp)
        # TODO: decide if we want to pass device here, or later.
        # Currently, do it later
        array = converter_instance.to(array, target_name)

    if device is not None:
        xp = array_namespace_func(array)
        array = xp.to_device(array, device=device, **kwargs)

    return array
