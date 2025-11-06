class UnknownPatchedNamespace:

    def __init__(self, xp):
        self._xp = xp

    @property
    def xp(self):
        if self._xp is None:
            self._set_xp()
        return self._xp

    def _set_xp(self):
        # This method should write self.__xp to the appropriate array namespace
        raise NotImplementedError("Subclasses must implement set_xp()")

    def __getattr__(self, name):
        return getattr(self.xp, name)  # Delegate to underlying namespace

    # TODO: come up with a better way to compare namespaces
    def __eq__(self, other):
        return self._earthkit_array_namespace_name == other._earthkit_array_namespace_name

    @property
    def _earthkit_array_namespace_name(self):
        return self.xp.__name__

    def polyval(self, x, c):
        """Evaluation of a polynomial using Horner's scheme.

        If ``c`` is of length ``n + 1``, this function returns the value

        .. math:: p(x) = c_0 + c_1 * x + ... + c_n * x^n


        Parameters
        ----------
        x: array-like
            The values(s) at which to evaluate the polynomial. Its elements must
            support addition and multiplication with with themselves and with
            the elements of ``c``.
        c: array-like
            Array of coefficients ordered so that the coefficients for terms of
            degree n are contained in c[n].

        Returns
        -------
        values : array-like
            The value(s) of the polynomial at the given point(s).


        Comments
        --------
        Based on the ``numpy.polynomal.polynomial.polyval`` function.
        """
        if hasattr(self.xp, "polyval"):
            return self.xp.polyval(x, c)
        else:
            c0 = c[-1] + x * 0
            for i in range(2, len(c) + 1):
                c0 = c[-i] + c0 * x
            return c0

    def percentile(self, a, q, axis=None, **kwargs):
        """Compute percentiles by calling the quantile function."""
        if axis is None:
            axis = 0
            a = self.xp.reshape(a, -1)

        a = self.xp.sort(a, axis=axis)
        n = self.xp.shape(a)[axis]
        rank = (q / 100) * (n - 1)
        low = int(self.xp.floor(rank))
        high = int(self.xp.ceil(rank))
        weight = rank - low

        a_low = self.xp.take(a, low, axis=axis)
        a_high = self.xp.take(a, high, axis=axis)

        return (1 - weight) * a_low + weight * a_high

    def histogram2d(self, x, y, *args, **kwargs):
        """Compute a 2D histogram.

        Parameters
        ----------
        x: array-like
            An array containing the x coordinates of the points to be histogrammed.
        y: array-like
            An array containing the y coordinates of the points to be histogrammed.
        """
        if hasattr(self.xp, "histogram2d"):
            return self.xp.histogram2d(x, y, *args, **kwargs)
        else:
            # TODO: fix - this is not array-api compliant
            return self.xp.histogramdd(self.xp.stack([x, y]).T, *args, **kwargs)

    # TODO: figure out what this is for
    def seterr(self, *args, **kwargs):
        """Set how floating-point errors are handled.

        Just a placeholder for the numpy function.
        """
        return dict()

    def size(self, x):
        """Return the size of an array."""
        # array.size is part of array api spec
        # but in practice not all backends implement it yet
        # therefore we provide this function
        # in order to be able to patch it if needed
        return x.size

    def shape(self, x):
        """Return the shape of an array."""
        # array.shape is part of array api spec
        # but in practice not all backends implement it yet
        # therefore we provide this function
        # in order to be able to patch it if needed
        return x.shape

    def to_device(self, x, device, **kwargs):
        # array.to_device(device, **kwargs) is part of array api spec
        # but in practice not all backends implement it yet
        # therefore we provide this function
        # in order to be able to patch it if needed
        return x.to_device(device, **kwargs)

    def device(self, x):
        # array.device is part of array api spec
        # but in practice not all backends implement it yet
        # therefore we provide this function
        # in order to be able to patch it if needed
        return x.device

    def dtype(self, x):
        # array.device is part of array api spec
        # but in practice not all backends implement it yet
        # therefore we provide this function
        # in order to be able to patch it if needed
        return x.dtype
