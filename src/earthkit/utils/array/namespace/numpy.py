# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from .unknown import UnknownPatchedNamespace


class PatchedNumpyNamespace(UnknownPatchedNamespace):

    def __init__(self):
        super().__init__(None)

    def _set_xp(self):
        import array_api_compat.numpy as np

        self._xp = np

    @property
    def _earthkit_array_namespace_name(self):
        return "numpy"

    def polyval(self, *args, **kwargs):
        from numpy.polynomial.polynomial import polyval

        return polyval(*args, **kwargs)

    def percentile(self, a, q, axis=None):
        return self._xp.percentile(a, q, axis=axis)

    def quantile(self, a, q, axis=None):
        return self._xp.quantile(a, q, axis=axis)

    def histogram2d(self, x, y, *, bins=10):
        return self.xp.histogram2d(x, y, bins=bins)

    def histogramdd(self, x, *, bins=10):
        return self.xp.histogramdd(x, bins=bins)

    def devices(self):
        from numpy import __array_namespace_info__

        info = __array_namespace_info__()
        return info.devices()

    def isclose(self, x, y, *, rtol=1e-5, atol=1e-8, equal_nan=False):
        return self.xp.isclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def allclose(self, x, y, *, rtol=1e-5, atol=1e-8, equal_nan=False):
        return self.xp.allclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)
