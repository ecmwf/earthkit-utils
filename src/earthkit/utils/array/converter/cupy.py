# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from .unknown import FromUnknownConverter


class FromCupyConverter(FromUnknownConverter):

    def __init__(self, xp_target):
        super().__init__(xp_target)

    def to_numpy(self, array, **kwargs):
        return array.get()

    def to_cupy(self, array, **kwargs):
        return array

    def to_torch(self, array, **kwargs):
        # TODO: add device handling
        from torch.utils.dlpack import from_dlpack

        return from_dlpack(array.toDlpack())

    def to_jax(self, array, **kwargs):
        # TODO: add device handling
        from jax.dlpack import from_dlpack

        return from_dlpack(array.toDlpack())
