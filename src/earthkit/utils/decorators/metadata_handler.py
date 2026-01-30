# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import typing as T
from functools import wraps


def metadata_handler(
    **kwargs
) -> T.Callable:
    """This is a placeholder decorator for handling automatic input data formatting.
    
    The real decorator is located in earthkit-data, this is only used if earthkit-data is not installed.

    Returns
    -------
    Callable
        Wrapped function.
    """
    def decorator(function: T.Callable) -> T.Callable:
        @wraps(function)
        def wrapper(*args, _auto_metadata_handler=True, **kwargs):
            return function(*args, **kwargs)
        return wrapper

    return decorator

