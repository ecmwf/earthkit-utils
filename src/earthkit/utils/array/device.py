# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from .array_namespace import array_namespace
from .convert import _NAMESPACES_BY_NAME
from .convert import converter
from .namespace.cupy import PatchedCupyNamespace
from .namespace.numpy import PatchedNumpyNamespace


def to_device(v, device=None, array_backend=None, **kwargs):
    """
    Return a copy/view of array moved to device.

    Parameters
    ----------
    v : array-like
        The array to be moved to the specified device.
    device : backend-specific device spec or str
        The device to which the array should be moved. For example,
        "cpu", "cuda:0", etc.
    array_backend : str or ArrayBackend
        The backend to use for the conversion. If None, the following logic
        is applied:
        - if the device is "cpu", it will use the numpy backend
        - otherwise it will use the backend of the array ``v``, but if that
          backend is numpy, it will use the cupy backend.
    *args, **kwargs : forwarded to the underlying backend call
    """
    if array_backend is None:
        if device is None:
            return v
        else:
            current_xp = array_namespace(v)
            if device == "cpu":
                device = None
                target_xp = PatchedNumpyNamespace()
            else:
                if current_xp._earthkit_array_namespace_name == "numpy":
                    device = None
                    target_xp = PatchedCupyNamespace()
                else:
                    target_xp = current_xp
    else:
        if type(array_backend) is str:
            target_xp = _NAMESPACES_BY_NAME[array_backend]()
        else:
            target_xp = array_backend

    # TODO: if target_xp == current_xp: do something smarter

    if device is None:
        return converter(v, target_xp, **kwargs)
    else:
        return converter(v, target_xp, device=device, **kwargs)
