from abc import ABCMeta
from abc import abstractmethod
from functools import cached_property


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
    def from_numpy(self, v, **kwargs):
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
            return self.make_dtype(dtype)

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
