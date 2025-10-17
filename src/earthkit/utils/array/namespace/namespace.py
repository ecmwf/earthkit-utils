class PatchedNamespace:

    def __init__(self, xp):
        self._xp = xp

    def __getattr__(self, name):
        return getattr(self._xp, name)  # Delegate to underlying namespace

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    @property
    def _earthkit_array_namespace_name(self):
        return None

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
        if hasattr(self._xp, "polyval"):
            return self._xp.polyval(x, c)
        else:
            c0 = c[-1] + x * 0
            for i in range(2, len(c) + 1):
                c0 = c[-i] + c0 * x
            return c0

    def percentile(self, a, q, **kwargs):
        """Compute percentiles by calling the quantile function."""
        if hasattr(self._xp, "percentile"):
            return self._xp.percentile(a, q, **kwargs)
        else:
            # TODO: fix - this is not array-api compliant
            return self._xp.quantile(a, q / 100, **kwargs)

    def histogram2d(self, x, y, *args, **kwargs):
        """Compute a 2D histogram.

        Parameters
        ----------
        x: array-like
            An array containing the x coordinates of the points to be histogrammed.
        y: array-like
            An array containing the y coordinates of the points to be histogrammed.
        """
        if hasattr(self._xp, "histogram2d"):
            return self._xp.histogram2d(x, y, *args, **kwargs)
        else:
            # TODO: fix - this is not array-api compliant
            return self._xp.histogramdd(self._xp.stack([x, y]).T, *args, **kwargs)

    # TODO: figure out what this is for
    def seterr(self, *args, **kwargs):
        """Set how floating-point errors are handled.

        Just a placeholder for the numpy function.
        """
        return dict()
