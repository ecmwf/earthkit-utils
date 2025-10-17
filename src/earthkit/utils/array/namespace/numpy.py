# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from .namespace import PatchedNamespace


class PatchedNumpyNamespace(PatchedNamespace):

    def __init__(self):
        import array_api_compat.numpy as np

        super().__init__(np)

    @property
    def _earthkit_array_namespace_name(self):
        return "numpy"

    def polyval(self, *args, **kwargs):
        from numpy.polynomial.polynomial import polyval

        return polyval(*args, **kwargs)
