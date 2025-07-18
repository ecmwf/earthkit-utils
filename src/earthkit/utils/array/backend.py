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

import array_api_compat


def is_scalar(data):
    return isinstance(data, (int, float)) or data is not data


class ArrayBackend(metaclass=ABCMeta):
    """Abstract base class for array backends.

    An ArrayBackend enables using different array libraries
    (numpy, torch, cupy, jax) in a uniform way. It provides methods to
    convert between different array types, and to access the related
    array namespaces.
    """

    name = None
    module_name = None

    @abstractmethod
    def _make_sample(self):
        """Create a sample array for this backend."""
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

    def make_dtype(self, dtype):
        """Return the dtype of an array."""
        if isinstance(dtype, str):
            d = self.compat_namespace.__array_namespace_info__().dtypes()
            return d.get(dtype, None)
        return dtype

    def to_numpy_dtype(self, dtype):
        dtype = self.dtype_to_str(dtype)
        if dtype is None:
            return None
        else:
            return _NUMPY.make_dtype(dtype)

    def dtype_to_str(self, dtype):
        """Convert a dtype to a str."""
        if not isinstance(dtype, str):
            d = self.compat_namespace.__array_namespace_info__().dtypes()
            for k, v in d.items():
                if v == dtype:
                    return k
            return None
        return dtype

    @property
    @abstractmethod
    def _dtypes(self):
        """Return a dictionary of predefined dtype classes."""
        pass

    @cached_property
    def float64(self):
        """Return the float64 dtype class."""
        return self._dtypes.get("float64")

    @cached_property
    def float32(self):
        """Return the float32 dtype class."""
        return self._dtypes.get("float32")

    # def to_dtype(self, dtype):
    #     """Return the dtype class from a string or dtype class."""
    #     if isinstance(dtype, str):
    #         return self.dtypes.get(dtype, None)
    #     return dtype

    # def match_dtype(self, v, dtype):
    #     """Return True if the dtype of an array matches the specified dtype."""
    #     if dtype is not None:
    #         dtype = self.to_dtype(dtype)
    #         f = v.dtype == dtype if dtype is not None else False
    #         return f
    #     return True

    @abstractmethod
    def to_device(self, v, device, *args, **kwargs):
        pass

    @abstractmethod
    def has_device(self, device):
        pass

    def astype(self, *args, **kwargs):
        """Convert an array to a new dtype."""
        return self.namespace.astype(*args, **kwargs)

    def asarray(self, *data, dtype=None, **kwargs):
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
        dtype = self.make_dtype(dtype) if dtype is not None else None
        res = [self.namespace.asarray(d, dtype=dtype, **kwargs) for d in data]
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
            return self.make_dtype(dtype)

    @cached_property
    def _dtypes(self):
        import numpy

        return {"float64": numpy.float64, "float32": numpy.float32}

    def to_device(self, v, device, *args, **kwargs):
        if device != "cpu":
            raise ValueError(f"Can only specify 'cpu' as device for numpy backend, got {device}")
        b = get_backend(v)
        if b is not self:
            v = b.to_numpy(v)

        return self.namespace.asarray(v)

    def has_device(self, device):
        if isinstance(device, str):
            return device == "cpu"
        return False


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
    def _dtypes(self):
        import torch

        return {"float64": torch.float64, "float32": torch.float32}

    def to_device(self, v, device, *args, **kwargs):
        from .convert import convert_array

        v = convert_array(v, target_backend=self, **kwargs)
        return v.to(device, *args, **kwargs)

    def has_device(self, device):
        try:
            if isinstance(device, str):
                if device == "mps":
                    return self.namespace.backends.mps.is_available()
        except Exception:
            pass

        return False


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
    def _dtypes(self):
        import cupy as cp

        return {"float64": cp.float64, "float32": cp.float32}

    def to_device(self, v, device, *args, **kwargs):
        from .convert import convert_array

        v = convert_array(v, target_backend=self, **kwargs)

        # CuPy uses integer devices; "cuda:1" is 1, "cuda" is 0
        if isinstance(device, str) and device.startswith("cuda"):
            _, _, idx = device.partition(":")
            dev_id = int(idx) if idx else 0
        else:
            dev_id = device

        with self.namespace.cuda.Device(dev_id):
            return self.namespace.asarray(v)

    def has_device(self, device):
        return False


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
    def _dtypes(self):
        import jax.numpy as jarray

        return {"float64": jarray.float64, "float32": jarray.float32}

    def to_device(self, v, device, *args, **kwargs):
        raise NotImplementedError("")

    def has_device(self, device):
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
