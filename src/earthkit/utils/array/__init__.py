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
from functools import cached_property
from functools import partial

import array_api_compat


def is_scalar(data):
    return isinstance(data, (int, float)) or data is not data


class ArrayBackend(metaclass=ABCMeta):
    """Abstract base class for array backends.

    An ArrayBackend enables using different array libraries
    (numpy, torch, cupy, jax) in a uniform way. It provides methods to
    convert between different array types, and to access the related
    array namespaces
    """

    name = None
    module_name = None

    @abstractmethod
    def _make_sample(self):
        return None

    @abstractmethod
    def match_namespace(self, xp):
        """Check if the given namespace matches this backend."""
        pass

    @cached_property
    @abstractmethod
    def namespace(self):
        """Return the patched array-api-compat namespace."""
        pass

    @cached_property
    @abstractmethod
    def raw_namespace(self):
        """Return the original module namespace."""
        pass

    @cached_property
    @abstractmethod
    def compat_namespace(self):
        """Return the array-api-compat namespace of the backend."""
        pass

    @abstractmethod
    def to_numpy(self, v):
        """Convert an array to a numpy array."""
        pass

    @abstractmethod
    def from_numpy(self, v):
        """Convert a numpy array to an array."""
        pass

    @abstractmethod
    def from_other(self, v, **kwargs):
        """Convert an array-like object to an array."""
        pass

    def dtype(self, v):
        """Return the dtype of an array."""
        if isinstance(v, str):
            d = self.compat_namespace.__array_namespace_info__().dtypes()
            return d.get(v, None)
        return None

    def to_numpy_dtype(self, dtype):
        dtype = self.dtype_to_str(dtype)
        if dtype is None:
            return None
        else:
            return _NUMPY.dtype(dtype)

    def dtype_to_str(self, dtype):
        """Convert a dtype to a numpy dtype."""
        if not isinstance(dtype, str):
            d = self.compat_namespace.__array_namespace_info__().dtypes()
            for k, v in d.items():
                if v == dtype:
                    return k
            return None
        return dtype

    @property
    @abstractmethod
    def dtypes(self):
        """Return a dictionary of dtype classes."""
        pass

    @cached_property
    def float64(self):
        """Return the float64 dtype class."""
        return self.dtypes.get("float64")

    @cached_property
    def float32(self):
        """Return the float32 dtype class."""
        return self.dtypes.get("float32")

    def to_dtype(self, dtype):
        """Return the dtype class from a string or dtype class."""
        if isinstance(dtype, str):
            return self.dtypes.get(dtype, None)
        return dtype

    def match_dtype(self, v, dtype):
        """Return True if the dtype of an array matches the specified dtype."""
        if dtype is not None:
            dtype = self.to_dtype(dtype)
            f = v.dtype == dtype if dtype is not None else False
            return f
        return True

    @abstractmethod
    def _make_device(self, device):
        pass

    def astype(self, *args, **kwargs):
        """Convert an array to a new dtype."""
        return self.namespace.astype(*args, **kwargs)

    def asarray(self, *data, **kwargs):
        """Convert data to an array.

        Parameters
        ----------
        data: tuple
            The data to convert to an array.
        kwargs: dict
            Additional keyword arguments.

        This method is a wrapper around the namespace.asarray method, which does
        not work with scalars. It ensures that scalars are converted to arrays
        with the correct dtype.
        """
        # TODO: add support for dtype
        res = [self.namespace.asarray(d, **kwargs) for d in data]
        # if "dtype" not in kwargs:
        #     dtype = res[0].dtype
        #     for i in range(1, len(res)):
        #         res[i] = self.namespace.asarray(res[i], dtype=dtype)

        r = res if len(res) > 1 else res[0]
        return r

    def allclose(self, *args, **kwargs):
        """Return True if all arrays are equal within a tolerance.

        This method is a wrapper around the namespace.asarray method. It ensures that
        scalars are converted to arrays with the correct dtype.
        """
        if is_scalar(args[0]):
            dtype = self.float64
            v = [self.asarray(a, dtype=dtype) for a in args]
        else:
            v = args
        return self.namespace.allclose(*v, **kwargs)

    def isclose(self, *args, **kwargs):
        """Return True if all arrays are equal within a tolerance.

        This method is a wrapper around the namespace.isclose method. It ensures that
        scalars are converted to arrays with the correct dtype.
        """
        if is_scalar(args[0]):
            dtype = self.float64
            v = [self.asarray(a, dtype=dtype) for a in args]
        else:
            v = args
        return self.namespace.isclose(*v, **kwargs)


class NumpyBackend(ArrayBackend):
    name = "numpy"
    module_name = "numpy"

    def _make_sample(self):
        import numpy as np

        return np.ones(2)

    def match_namespace(self, xp):
        return array_api_compat.is_numpy_namespace(xp)

    @cached_property
    def namespace(self):
        """Return the patched version of the array-api-compat numpy namespace."""
        import earthkit.utils.array.namespace.numpy as xp

        return xp

    @cached_property
    def compat_namespace(self):
        """Return the array-api-compat numpy namespace."""
        import array_api_compat.numpy as xp

        return xp

    @cached_property
    def raw_namespace(self):
        import numpy as np

        return np

    def to_numpy(self, v):
        return v

    def from_numpy(self, v):
        return v

    def from_other(self, v, **kwargs):
        import numpy as np

        if not kwargs and isinstance(v, np.ndarray):
            return v

        return np.array(v, **kwargs)

    def to_numpy_dtype(self, dtype):
        dtype = self.dtype_to_str(dtype)
        if dtype is None:
            return None
        else:
            return self.dtype(dtype)

    @cached_property
    def dtypes(self):
        import numpy

        return {"float64": numpy.float64, "float32": numpy.float32}

    def _make_device(self, device):
        return device

    def to_device(self, v, device, *args, **kwargs):
        if device != "cpu":
            raise ValueError(f"Can only specify 'cpu' as device for numpy backend, got {device}")
        b = get_backend(v)
        if b is not self:
            v = b.to_numpy(v)

        return self.namespace.asarray(v)


class TorchBackend(ArrayBackend):
    name = "torch"
    module_name = "torch"

    def _make_sample(self):
        import torch

        return torch.ones(2)

    def match_namespace(self, xp):
        return array_api_compat.is_torch_namespace(xp)

    @cached_property
    def namespace(self):
        """Return the patched version of the array-api-compat torch namespace."""
        import earthkit.utils.array.namespace.torch as xp

        return xp

    @cached_property
    def compat_namespace(self):
        """Return the array-api-compat torch namespace."""
        import array_api_compat.torch as xp

        return xp

    @cached_property
    def raw_namespace(self):
        import torch

        return torch

    def to_numpy(self, v):
        return v.cpu().numpy()

    def from_numpy(self, v):
        import torch

        return torch.from_numpy(v)

    def from_other(self, v, **kwargs):
        import torch

        return torch.tensor(v, **kwargs)

    @cached_property
    def dtypes(self):
        import torch

        return {"float64": torch.float64, "float32": torch.float32}

    def _make_device(self, device):
        return device

    def to_device(self, v, device, *args, **kwargs):
        b = get_backend(v)
        if b is not self:
            v = self.from_other(v)

        return v.to(device, *args, **kwargs)


class CupyBackend(ArrayBackend):
    name = "cupy"
    module_name = "cupy"

    def _make_sample(self):
        import cupy

        return cupy.ones(2)

    def match_namespace(self, xp):
        return array_api_compat.is_cupy_namespace(xp)

    @cached_property
    def namespace(self):
        """Return the patched version of the array-api-compat numpy namespace."""
        import earthkit.utils.array.namespace.cupy as xp

        return xp

    @cached_property
    def compat_namespace(self):
        """Return the array-api-compat cupy namespace."""
        import array_api_compat.cupy as xp

        return xp

    @cached_property
    def raw_namespace(self):
        import cupy

        return cupy

    def from_numpy(self, v):
        return self.from_other(v)

    def to_numpy(self, v):
        return v.get()

    def from_other(self, v, **kwargs):
        import cupy as cp

        return cp.array(v, **kwargs)

    @cached_property
    def dtypes(self):
        import cupy as cp

        return {"float64": cp.float64, "float32": cp.float32}

    def _make_device(self, device):
        return device

    def to_device(self, v, device, *args, **kwargs):
        b = get_backend(v)
        if b is not self:
            v = self.from_other(v)

        # CuPy uses integer devices; "cuda:1" is 1, "cuda" is 0
        if isinstance(device, str) and device.startswith("cuda"):
            _, _, idx = device.partition(":")
            dev_id = int(idx) if idx else 0
        else:
            dev_id = device

        with self.namespace.cuda.Device(dev_id):
            return self.namespace.asarray(v)


class JaxBackend(ArrayBackend):
    name = "jax"
    module_name = "jax"

    def _make_sample(self):
        import jax.numpy as jarray

        return jarray.ones(2)

    def match_namespace(self, xp):
        return array_api_compat.is_jax_namespace(xp)

    @cached_property
    def namespace(self):
        """Return the of the array-api-compat jax namespace."""
        return array_api_compat.array_namespace(self._make_sample())

    @cached_property
    def compat_namespace(self):
        return self.namespace

    @cached_property
    def raw_namespace(self):
        import jax.numpy as jarray

        return jarray

    def to_numpy(self, v):
        import numpy as np

        return np.array(v)

    def from_numpy(self, v):
        return self.from_other(v)

    def from_other(self, v, **kwargs):
        import jax.numpy as jarray

        return jarray.array(v, **kwargs)

    @cached_property
    def dtypes(self):
        import jax.numpy as jarray

        return {"float64": jarray.float64, "float32": jarray.float32}

    def _make_device(self, device):
        return device

    def to_device(self, v, device, *args, **kwargs):
        raise NotImplementedError("")


_NUMPY = NumpyBackend()
_TORCH = TorchBackend()
_JAX = JaxBackend()
_CUPY = CupyBackend()

_DEFAULT_BACKEND = _NUMPY
_BACKENDS = [_NUMPY, _TORCH, _CUPY, _JAX]
_BACKENDS_BY_NAME = {v.name: v for v in _BACKENDS}
_BACKENDS_BY_MODULE = {v.module_name: v for v in _BACKENDS}

# add pytorch name for backward compatibility
_BACKENDS_BY_NAME["pytorch"] = _TORCH


# TODO: maybe this is not necessary
def other_namespace(xp):
    """Return the patched version of an array-api-compat namespace."""
    if not hasattr(xp, "histogram2d"):
        from .compute import histogram2d

        xp.histogram2d = partial(histogram2d, xp)
    if not hasattr(xp, "polyval"):
        from .compute import polyval

        xp.polyval = partial(polyval, xp)
    if not hasattr(xp, "percentile"):
        from .compute import percentile

        xp.percentile = partial(percentile, xp)

    if not hasattr(xp, "seterr"):
        from .compute import seterr

        xp.seterr = partial(seterr, xp)

    return xp


def array_namespace(*args):
    """Return the array namespace of the arguments.

    Parameters
    ----------
    *args: tuple
        Scalar or array-like arguments.

    Returns
    -------
    xp: module
        The array-api-compat namespace of the arguments. The namespace
        returned from array_api_compat.array_namespace(*args) is patched with
        extra/modified methods. When only a scalar is passed, the numpy namespace
        is returned.

    Notes
    -----
    The array namespace is extended with the following methods when necessary:
        - polyval: evaluate a polynomial (available in numpy)
        - percentile: compute the n-th percentile of the data along the
          specified axis (available in numpy)
        - histogram2d: compute a 2D histogram (available in numpy)
    Some other methods may be reimplemented for a given namespace to ensure correct
    behaviour. E.g. sign() for torch.
    """
    arrays = [a for a in args if hasattr(a, "shape")]
    if not arrays:
        return _DEFAULT_BACKEND.namespace
    else:
        xp = array_api_compat.array_namespace(*arrays)
        for b in _BACKENDS:
            if b.match_namespace(xp):
                return b.namespace

        return xp


def array_to_numpy(array):
    """Convert an array to a numpy array."""
    return backend_from_array(array).to_numpy(array)


def backend_from_array(array, raise_exception=True):
    """Return the array backend of an array-like object."""
    xp = array_api_compat.array_namespace(array)
    for b in _BACKENDS:
        if b.match_namespace(xp):
            return b

    if raise_exception:
        raise ValueError(f"Can't find namespace for array type={type(array)}")

    return xp


def backend_from_name(name, raise_exception=True):
    r = _BACKENDS_BY_NAME.get(name, None)
    if raise_exception and r is None:
        raise ValueError(f"Unknown array backend name={name}")
    return r


def backend_from_module(module, raise_exception=True):
    import inspect

    r = None
    if inspect.ismodule(module):
        name = module.__name__
        if "." in name:
            name = name.split(".")[-1]  # get the top-level module name

        r = _BACKENDS_BY_MODULE.get(name, None)
        if raise_exception and r is None:
            raise ValueError(f"Unknown array backend module={module}")
    return r


def get_backend(data):
    if isinstance(data, ArrayBackend):
        return data
    if isinstance(data, str):
        return backend_from_name(data, raise_exception=True)

    r = backend_from_module(data, raise_exception=True)
    if r is None:
        r = backend_from_array(data)

    return r


def to_numpy_dtype(dtype, default=None):
    for b in _BACKENDS:
        v = b.to_numpy_dtype(dtype)
        if v is not None:
            return v
    return default


class Converter:
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def __call__(self, array, **kwargs):
        if self.source == _NUMPY:
            return self.target.from_numpy(array, **kwargs)
        return self.target.from_other(array, **kwargs)


def converter(array, target):
    if target is None:
        return None

    source_backend = backend_from_array(array)
    target_backend = get_backend(target)

    if source_backend == target_backend:
        return None
    return Converter(source_backend, target_backend)


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
            r.append(c(a, **kwargs))

    if not target_is_list:
        return r[0]
    return r


def to_device(v, device, array_backend, *args, **kwargs):
    """
    Return a copy/view of array located on device.

    Parameters
    ----------
    v      : array-like
        The array to be moved to the specified device.
    device   : backend-specific device spec or str
        The device to which the array should be moved. For example,
        "cpu", "cuda:0", etc.
    array_backend : str or ArrayBackend
        The backend to use for the conversion. If None, the follwing logic
        is applied:
        - if the device is "cpu", it will use the numpy backend
        - otherwise it will use the backend of the array ``v``, but if that
          backend is numpy, it will use the cupy backend.
    *args, **kwargs : forwarded to the underlying backend call
    """
    if array_backend is None:
        if device == "cpu":
            array_backend = get_backend("numpy")
        else:
            current_backend = get_backend(v)
            if current_backend.name == "numpy":
                array_backend = get_backend("cupy")
            else:
                array_backend = current_backend
    else:
        array_backend = get_backend(array_backend)

    assert array_backend is not None, "The 'backend' argument must be specified."

    return array_backend.to_device(v, device, *args, **kwargs)
