# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from earthkit.utils.units.array import convert_units as convert_array
from earthkit.utils.units.convert import (
    are_compatible,
    are_equal,
    convert_units,
)
from earthkit.utils.units.units import Units

# TODO: remove convert_array, it is only kept because earthkit-data and earthkit-plots
# still use it. New code should use convert_units.

__all__ = [
    "Units",
    "are_compatible",
    "are_equal",
    "convert_array",
    "convert_units",
]
