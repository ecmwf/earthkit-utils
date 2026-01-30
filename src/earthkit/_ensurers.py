# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

def ensure_iterable(input_item):
    """Ensure that an item is iterable."""
    if not isinstance(input_item, (tuple, list, dict)):
        return [input_item]
    return input_item


def ensure_tuple(input_item):
    """Ensure that an item is a tuple."""
    if not isinstance(input_item, tuple):
        return tuple(ensure_iterable(input_item))
    return input_item

