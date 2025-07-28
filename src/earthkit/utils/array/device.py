# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


def to_device(v, device, *args, array_backend=None, **kwargs):
    """
    Return a copy/view of array moved to device.

    Parameters
    ----------
    v      : array-like
        The array to be moved to the specified device.
    device   : backend-specific device spec or str
        The device to which the array should be moved. For example,
        "cpu", "cuda:0", etc.
    array_backend : str or ArrayBackend
        The backend to use for the conversion. If None, the follwing logic
        is applied:
        - if the device is "cpu", it will use the numpy backend
        - otherwise it will use the backend of the array ``v``, but if that
          backend is numpy, it will use the cupy backend.
    *args, **kwargs : forwarded to the underlying backend call
    """
    from .backend import get_backend

    if array_backend is None:
        if device == "cpu":
            array_backend = get_backend("numpy")
        else:
            current_backend = get_backend(v)
            if current_backend.name == "numpy":
                array_backend = get_backend("cupy")
            else:
                array_backend = current_backend
    else:
        array_backend = get_backend(array_backend)

    assert array_backend is not None, "The 'backend' argument must be specified."

    return array_backend.to_device(v, device, *args, **kwargs)
