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

from .backend import _CUPY
from .backend import _JAX
from .backend import _NUMPY
from .backend import _TORCH
from .backend import backend_from_array
from .backend import get_backend


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


def converter(array, target):
    if target is None:
        return None

    source_backend = backend_from_array(array)
    target_backend = get_backend(target)

    if source_backend == target_backend:
        return None

    key = (source_backend.name, target_backend.name)
    c = CONVERTERS.get(key, None)

    if c is None:
        c = DefaultConverter(target_backend)
    return c


def convert_array(array, target_backend=None, target_array_sample=None, **kwargs):
    if target_backend is not None and target_array_sample is not None:
        raise ValueError("Only one of target_backend or target_array_sample can be specified")
    if target_backend is not None:
        target = target_backend
    else:
        target = backend_from_array(target_array_sample)

    r = []
    target_is_list = True
    if not isinstance(array, (list, tuple)):
        array = [array]
        target_is_list = False

    for a in array:
        c = converter(a, target)
        if c is None:
            r.append(a)
        else:
            r.append(c.convert(a, **kwargs))

    if not target_is_list:
        return r[0]
    return r


def array_to_numpy(array):
    """Convert an array to a numpy array."""
    return backend_from_array(array).to_numpy(array)
