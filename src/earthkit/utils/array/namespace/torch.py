# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import array_api_compat.torch as torch

from .abstract import PatchedNamespace


class PatchedTorchNamespace(PatchedNamespace):

    def __init__(self):
        super().__init__(torch)

    def sign(self, x, *args, **kwargs):
        """Reimplement the sign function to handle NaNs.

        The problem is that torch.sign returns 0 for NaNs, but the array API
        standard requires NaNs to be propagated.
        """
        x = self._xp.asarray(x)
        r = self._xp.sign(x, *args, **kwargs)
        r = self._xp.asarray(r)
        r[self._xp.isnan(x)] = self._xp.nan
        return r
