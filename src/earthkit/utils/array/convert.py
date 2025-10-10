# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from abc import ABCMeta
from abc import abstractmethod

from .array_backend import array_namespace
from .backend import _CUPY
from .backend import _JAX
from .backend import _NUMPY
from .backend import _TORCH
from .backend import get_backend
from .namespace.abstract import PatchedNamespace
from .namespace.cupy import PatchedCupyNamespace  # noqa: F401
from .namespace.numpy import PatchedNumpyNamespace  # noqa: F401
from .namespace.torch import PatchedTorchNamespace  # noqa: F401


class Converter(metaclass=ABCMeta):
    source = None
    target = None

    @abstractmethod
    def convert(self, array, **kwargs):
        pass


class DefaultConverter(Converter):
    def __init__(self, target):
        self.target = target

    def convert(self, array, **kwargs):
        return self.target.from_other(array, **kwargs)


class NumpyToOtherConverter(Converter):
    source = _NUMPY

    def convert(self, array, **kwargs):
        return self.target.from_numpy(array, **kwargs)


class OtherToNumpyConverter(Converter):
    target = _NUMPY

    def convert(self, array, **kwargs):
        return self.source.to_numpy(array, **kwargs)


class NumpyToTorchConverter(NumpyToOtherConverter):
    target = _TORCH


class NumpyToCupyConverter(NumpyToOtherConverter):
    target = _CUPY


class NumpyToJaxConverter(NumpyToOtherConverter):
    target = _JAX


class TorchToNumpyConverter(OtherToNumpyConverter):
    source = _TORCH


class CupyToNumpyConverter(OtherToNumpyConverter):
    source = _CUPY


class JaxToNumpyConverter(OtherToNumpyConverter):
    source = _JAX


class TorchToCupyConverter(Converter):
    source = _TORCH
    target = _CUPY

    def convert(self, array, **kwargs):

        import cupy
        from torch.utils.dlpack import to_dlpack

        # Convert it into a DLPack tensor.
        dx = to_dlpack(array.cuda())

        # Convert it into a CuPy array.
        return cupy.fromDlpack(dx)


class CupyToTorchConverter(Converter):
    source = _CUPY
    target = _TORCH

    def convert(self, array, **kwargs):

        from torch.utils.dlpack import from_dlpack

        return from_dlpack(array.toDlpack())


CONVERTERS = {
    (c.source.name, c.target.name): c
    for c in [
        NumpyToTorchConverter(),
        NumpyToCupyConverter(),
        NumpyToJaxConverter(),
        TorchToNumpyConverter(),
        CupyToNumpyConverter(),
        JaxToNumpyConverter(),
        TorchToCupyConverter(),
        CupyToTorchConverter(),
    ]
}

_NAMESPACES_BY_NAME = {
    "numpy": PatchedNumpyNamespace,
    "cupy": PatchedCupyNamespace,
    "torch": PatchedTorchNamespace,
}


def converter(array, target_xp, **kwargs):
    if isinstance(target_xp, PatchedNamespace):
        pass
    else:
        name = target_xp.__name__
        if "." in name:
            name = name.split(".")[-1]
        matched_xp = _NAMESPACES_BY_NAME.get(name, None)
        if matched_xp is None:
            target_xp = PatchedNamespace(target_xp)
        else:
            target_xp = matched_xp()

    source_xp = array_namespace(array)

    # if source_xp == target_xp:
    #     return array

    key = (source_xp._earthkit_array_namespace_name, target_xp._earthkit_array_namespace_name)

    if key[1] is None:
        # we don't know about the target

        # try DLPack conversion
        if hasattr(array, "__dlpack") and hasattr(target_xp, "from_dlpack"):
            return target_xp.from_dlpack(array.__dlpack__(), **kwargs)
        # hope xp.asarray works, otherwise try xp.array
        else:
            try:
                return target_xp.asarray(array, **kwargs)
            except Exception:
                try:
                    import numpy as np

                    return target_xp.asarray(np.array(array), **kwargs)
                except Exception:
                    try:
                        return target_xp.array(array, **kwargs)
                    except Exception:
                        return target_xp.array(np.array(array), **kwargs)

    else:
        c = CONVERTERS.get(key, None)
        if c is None:
            target_backend = get_backend(target_xp._earthkit_array_namespace_name)
            c = DefaultConverter(target_backend)

        return c.convert(array, **kwargs)
