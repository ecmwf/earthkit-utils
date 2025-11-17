# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from .unknown import FromUnknownConverter


class FromJaxConverter(FromUnknownConverter):

    def __init__(self, xp_target):
        super().__init__(xp_target)

    def to_numpy(self, array, **kwargs):
        return self.xp_target.asarray(array, **kwargs)

    def to_cupy(self, array, **kwargs):
        import cupy as cp

        return cp.from_dlpack(array)

    # def to_torch(self, array, **kwargs):
    #     return

    def to_jax(self, array, **kwargs):
        # TODO: device handling?
        return array
