# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import re
from functools import cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pint


@cache
def get_registry() -> pint.UnitRegistry:
    """Return the Pint unit registry shared by earthkit, creating it on first use."""
    import pint

    return pint.UnitRegistry()


UNITS_PATTERN_1 = re.compile(r"(?<=[a-zA-Z0-9])\s+(?=[a-zA-Z])")
UNITS_PATTERN_2 = re.compile(r"([a-zA-Z])(-?\d+)")
UNIT_STR_ALIASES: dict[str, str] = {"(0 - 1)": "percent"}


def _prepare_str(units: str | None = None) -> str:
    """Convert a unit string to a Pint-compatible string.

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
    elif not isinstance(units, str):
        raise ValueError(f"Unsupported type for units: {type(units)}")
    elif units in UNIT_STR_ALIASES:
        units = UNIT_STR_ALIASES[units]

    # Replace spaces between unit chunks with dots (e.g. "m s-1" -> "m.s-1")
    # Only replace spaces followed by a letter to avoid turning "** 2" into "**.2"
    units = UNITS_PATTERN_1.sub(".", units)

    # Insert ^ between characters and numbers (including negative numbers)
    units = UNITS_PATTERN_2.sub(r"\1^\2", units)

    return units


def _parse(units: str | None) -> pint.Unit | None:
    """Parse ``units`` with Pint, returning None when it is not a unit Pint understands."""
    ureg = get_registry()
    try:
        return ureg(_prepare_str(units)).units
    # Unit strings come from arbitrary metadata, and Pint raises many different
    # exceptions (including from tokenize and arithmetic) for strings it cannot parse
    except Exception:
        return None


class Units:
    """A physical unit, backed by a Pint unit when possible.

    Strings that Pint cannot parse (e.g. ``"code table"``
    or ``"dBZ"``, common in GRIB and NetCDF metadata) are kept as opaque
    strings: they can be compared and printed but not converted, and
    :meth:`to_pint` returns ``None`` for them.

    Examples
    --------
    Create a unit from a string — equivalent notations resolve to the same unit:

    >>> from earthkit.utils.units import Units
    >>> u = Units.from_any("m/s")
    >>> str(u)
    'meter / second'
    >>> Units.from_any("m s-1") == u
    True
    >>> Units.from_any("m s^-1") == u
    True

    Units can also be compared directly with strings:

    >>> u == "m s-1"
    True

    Use :meth:`to_pint` to retrieve the underlying :class:`pint.Unit` and
    perform Pint-based operations such as unit conversion:

    >>> pint_unit = u.to_pint()
    >>> pint_unit
    Unit("meter / second")
    >>> from earthkit.utils.units.units import get_registry
    >>> ureg = get_registry()
    >>> quantity = 10 * ureg.meter / ureg.second
    >>> quantity.to(pint_unit)
    Quantity(10.0, "meter / second")

    The ``(0 - 1)`` alias is recognised as a percentage:

    >>> str(Units.from_any("(0 - 1)"))
    'percent'
    >>> Units.from_any("(0 - 1)") == "%"
    True

    Unknown unit strings are kept as-is and return ``None`` from
    :meth:`to_pint`:

    >>> u_invalid = Units.from_any("code table")
    >>> str(u_invalid)
    'code table'
    >>> u_invalid.to_pint() is None
    True
    """

    def __init__(self, units: str | pint.Unit | None = None) -> None:
        """Initialise the unit.

        Parameters
        ----------
        units : str, pint.Unit or None
            The unit. ``None`` means dimensionless.
        """
        import pint

        if isinstance(units, pint.Unit):
            self._units, self._pint = str(units), units
        elif units is None or isinstance(units, str):
            self._units, self._pint = units, _parse(units)
        else:
            raise ValueError(f"Unsupported type for units: {type(units)}")

    @staticmethod
    def from_any(units: str | pint.Unit | Units | None) -> Units:
        """Construct a :class:`Units` instance from various input types.

        Parameters
        ----------
        units : str, None, pint.Unit, or Units
            The unit. Existing :class:`Units` instances are returned unchanged.

        Returns
        -------
        Units

        Raises
        ------
        ValueError
            If *units* is of an unsupported type.
        """
        if isinstance(units, Units):
            return units
        return Units(units)

    def to_pint(self) -> pint.Unit | None:
        """Return the Pint unit, or ``None`` if Pint cannot parse the unit."""
        return self._pint

    def _key(self) -> str:
        """Return the string identifying the unit, used for equality and hashing."""
        return _prepare_str(self._units) if self._pint is None else str(self._pint)

    def __str__(self) -> str:
        """Return the Pint name of the unit, or the original string when Pint cannot parse it."""
        return self._units if self._pint is None else str(self._pint)

    def __repr__(self) -> str:
        return self._units if self._pint is None else repr(self._pint)

    def __eq__(self, other: Any) -> bool:
        """Check equality with another unit, which can be any input accepted by :meth:`from_any`."""
        try:
            other = Units.from_any(other)
        except ValueError:
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __getstate__(self) -> dict:
        return {"units": str(self)}

    def __setstate__(self, state: dict) -> None:
        self.__init__(state["units"])
