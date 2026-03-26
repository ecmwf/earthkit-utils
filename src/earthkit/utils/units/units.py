# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import re
from abc import ABCMeta, abstractmethod
from typing import Any

import pint
from pint import UnitRegistry

ureg = UnitRegistry()
Q_ = ureg.Quantity

UNITS_PATTERN_1 = re.compile(r"(?<=[a-zA-Z0-9])\s+(?=[a-zA-Z])")
UNITS_PATTERN_2 = re.compile(r"([a-zA-Z])(-?\d+)")
UNIT_STR_ALIASES = {"(0 - 1)": "percent"}


def _prepare_str(units: str = None) -> str:
    """
    Convert a unit string to a Pint-compatible string.

    For example, it converts "m s-1" to "m.s^-1".

    Parameters
    ----------
    units : str
        The unit string to convert.

    Returns
    -------
    str
        The converted unit string. When `units` is `None`, returns "dimensionless".
    """
    if units is None:
        units = "dimensionless"

    if not isinstance(units, str):
        raise ValueError(f"Unsupported type for units: {type(units)}")

    if units in UNIT_STR_ALIASES:
        units = UNIT_STR_ALIASES[units]

    # Replace spaces between unit chunks with dots (e.g. "m s-1" -> "m.s-1")
    # Only replace spaces followed by a letter to avoid turning "** 2" into "**.2"
    units = UNITS_PATTERN_1.sub(".", units)

    # Insert ^ between characters and numbers (including negative numbers)
    units = UNITS_PATTERN_2.sub(r"\1^\2", units)

    return units


class Units(metaclass=ABCMeta):
    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def __getstate__(self) -> dict:
        pass

    @abstractmethod
    def __setstate__(self, state: dict) -> None:
        pass

    @abstractmethod
    def to_pint(self) -> pint.Unit | None:
        pass

    @staticmethod
    def from_any(units):
        if isinstance(units, str) or units is None:
            units = _prepare_str(units)
            # TODO: consider the range of exceptions that we accept here.
            try:
                return PintUnits(ureg(units).units)
            except (pint.errors.UndefinedUnitError, AssertionError, AttributeError):
                return StrUnits(units)
        elif isinstance(units, pint.Unit):
            return PintUnits(units)
        elif isinstance(units, Units):
            return units

        else:
            raise ValueError(f"Unsupported type for units: {type(units)}")


class StrUnits(Units):
    def __init__(self, units: str) -> None:
        self._units = units

    def __repr__(self) -> str:
        return self._units

    def __str__(self) -> str:
        return self._units

    def __eq__(self, other) -> bool:
        other = Units.from_any(other)
        return str(other) == self._units

    def __hash__(self) -> int:
        return hash(str(self))

    def to_pint(self) -> None:
        return None

    def __getstate__(self) -> dict:
        return {"units": self._units}

    def __setstate__(self, state: dict) -> None:
        self._units = state["units"]


class PintUnits(Units):
    def __init__(self, units: pint.Unit) -> None:
        self._units = units

    def __repr__(self) -> Any:
        return self._units.__repr__()

    def __str__(self) -> str:
        return str(self._units)

    def __eq__(self, other) -> bool:
        other = Units.from_any(other)
        self_pint = self.to_pint()
        other_pint = other.to_pint()
        if self_pint is None and other_pint is None:
            return self_pint == other_pint

        return str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))

    def to_pint(self) -> pint.Unit | None:
        return self._units

    @staticmethod
    def _to_pint(units: str) -> pint.Unit:
        return ureg(units).units

    def __getstate__(self) -> dict:
        return {"units": str(self)}

    def __setstate__(self, state: dict) -> None:
        self._units = PintUnits._to_pint(state["units"])
