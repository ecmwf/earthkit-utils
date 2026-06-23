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

ureg: UnitRegistry = UnitRegistry()
Q_: type[pint.Quantity] = ureg.Quantity

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
    """Abstract base class representing a physical unit.

    Provides an interface for unit representations that may or may not
    be backed by a Pint unit.

    Two concrete implementations exist:

    :class:`PintUnits`
        Wraps a :class:`pint.Unit` object.  Created when the input string can
        be successfully parsed by Pint.  :meth:`to_pint` returns the
        underlying :class:`pint.Unit`, enabling further Pint-based arithmetic
        and dimensional analysis.
    :class:`StrUnits`
        Stores the unit as a plain string.  Created as a fallback when Pint
        cannot parse the input.  :meth:`to_pint` returns ``None``.

    Use :meth:`to_pint` to obtain the underlying :class:`pint.Unit` when
    interoperability with the Pint library is needed.  Callers should guard
    against ``None`` to handle unrecognised unit strings gracefully.

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
    <Unit('meter / second')>
    >>> from earthkit.utils.units.units import ureg
    >>> quantity = 10 * ureg.meter / ureg.second
    >>> quantity.to(pint_unit)
    <Quantity(10, 'meter / second')>

    The ``(0 - 1)`` alias is recognised as a percentage:

    >>> str(Units.from_any("(0 - 1)"))
    'percent'
    >>> Units.from_any("(0 - 1)") == "%"
    True

    Unknown unit strings are stored as-is and return ``None`` from
    :meth:`to_pint`:

    >>> u_invalid = Units.from_any("invalid")
    >>> str(u_invalid)
    'invalid'
    >>> u_invalid.to_pint() is None
    True
    """

    @abstractmethod
    def __repr__(self) -> str:
        """Return the string representation of the unit."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Return the string form of the unit."""
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        """Check equality with another unit."""
        pass

    @abstractmethod
    def __hash__(self) -> int:
        """Return a hash of the unit."""
        pass

    @abstractmethod
    def __getstate__(self) -> dict:
        """Return a picklable state dictionary."""
        pass

    @abstractmethod
    def __setstate__(self, state: dict) -> None:
        """Restore the unit from a state dictionary.

        Parameters
        ----------
        state : dict
            The state dictionary previously returned by ``__getstate__``.
        """
        pass

    @abstractmethod
    def to_pint(self) -> pint.Unit | None:
        """Return the underlying Pint unit, or ``None`` if not available.

        Returns
        -------
        pint.Unit or None
            The Pint unit, or ``None`` if this unit cannot be represented
            as a Pint unit.
        """
        pass

    @staticmethod
    def from_any(units):
        """Construct a :class:`Units` instance from various input types.

        Parameters
        ----------
        units : str, None, pint.Unit, or Units
            The unit to convert.  Strings and ``None`` are parsed and, where
            possible, resolved to a Pint-backed unit.  ``pint.Unit`` objects
            are wrapped directly.  Existing :class:`Units` instances are
            returned unchanged.

        Returns
        -------
        Units
            A :class:`PintUnits` or :class:`StrUnits` instance.

        Raises
        ------
        ValueError
            If *units* is of an unsupported type.
        """
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
    """A unit representation backed by a plain string.

    Used when the unit string cannot be parsed by Pint.
    """

    def __init__(self, units: str) -> None:
        """Initialise with a unit string.

        Parameters
        ----------
        units : str
            The unit string to store.
        """
        self._units = units

    def __repr__(self) -> str:
        """Return the stored unit string."""
        return self._units

    def __str__(self) -> str:
        """Return the stored unit string."""
        return self._units

    def __eq__(self, other) -> bool:
        """Check equality by comparing string representations.

        Parameters
        ----------
        other : str, None, pint.Unit, or Units
            The unit to compare against.

        Returns
        -------
        bool
            ``True`` if the string representations are equal.
        """
        other = Units.from_any(other)
        return str(other) == self._units

    def __hash__(self) -> int:
        """Return a hash based on the string representation."""
        return hash(str(self))

    def to_pint(self) -> None:
        """Return ``None`` as this unit has no Pint representation.

        Returns
        -------
        None
        """
        return None

    def __getstate__(self) -> dict:
        """Return the pickle state dictionary.

        Returns
        -------
        dict
            Dictionary with key ``'units'`` containing the unit string.
        """
        return {"units": self._units}

    def __setstate__(self, state: dict) -> None:
        """Restore the unit from a pickle state dictionary.

        Parameters
        ----------
        state : dict
            Dictionary with key ``'units'`` containing the unit string.
        """
        self._units = state["units"]


class PintUnits(Units):
    """A unit representation backed by a :class:`pint.Unit` object."""

    def __init__(self, units: pint.Unit) -> None:
        """Initialise with a Pint unit.

        Parameters
        ----------
        units : pint.Unit
            The Pint unit to wrap.
        """
        self._units = units

    def __repr__(self) -> Any:
        """Return the Pint unit's repr."""
        return self._units.__repr__()

    def __str__(self) -> str:
        """Return the string form of the Pint unit."""
        return str(self._units)

    def __eq__(self, other) -> bool:
        """Check equality against another unit.

        Parameters
        ----------
        other : str, None, pint.Unit, or Units
            The unit to compare against.

        Returns
        -------
        bool
            ``True`` if both units have the same string representation, or
            if both have no Pint representation and compare equal.
        """
        other = Units.from_any(other)
        self_pint = self.to_pint()
        other_pint = other.to_pint()
        if self_pint is None and other_pint is None:
            return self_pint == other_pint

        return str(self) == str(other)

    def __hash__(self) -> int:
        """Return a hash based on the string representation."""
        return hash(str(self))

    def to_pint(self) -> pint.Unit | None:
        """Return the underlying Pint unit.

        Returns
        -------
        pint.Unit
            The wrapped Pint unit.
        """
        return self._units

    @staticmethod
    def _to_pint(units: str) -> pint.Unit:
        """Parse a unit string into a Pint unit.

        Parameters
        ----------
        units : str
            A Pint-compatible unit string.

        Returns
        -------
        pint.Unit
            The parsed Pint unit.
        """
        return ureg(units).units

    def __getstate__(self) -> dict:
        """Return the pickle state dictionary.

        Returns
        -------
        dict
            Dictionary with key ``'units'`` containing the unit as a string.
        """
        return {"units": str(self)}

    def __setstate__(self, state: dict) -> None:
        """Restore the unit from a pickle state dictionary.

        Parameters
        ----------
        state : dict
            Dictionary with key ``'units'`` containing a unit string.
        """
        self._units = PintUnits._to_pint(state["units"])
