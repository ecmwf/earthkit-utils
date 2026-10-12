# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from itertools import combinations
from typing import Any, TypeAlias

import numpy as np

from ..convert import are_compatible
from ..units import Units, get_registry

LOG = logging.getLogger(__name__)

ArrayLike: TypeAlias = Any


def convert(data: ArrayLike, target_units, source_units=None, errors: str = "raise") -> ArrayLike:
    """Convert the units of an array.

    Parameters
    ----------
    data : array-like
        The data to convert. It is converted to a numpy array.
    target_units : str, pint.Unit, Units or list
        The units to convert to. With a list, the data is converted to the one
        unit in it compatible with ``source_units``; two compatible units in the
        list are ambiguous and always raise.
    source_units : str, pint.Unit or Units
        The units of the data.
    errors : {"raise", "ignore"}, default "raise"
        What to do when the conversion cannot be performed. With ``"ignore"``
        a warning is logged and ``data`` is returned unchanged. With ``"raise"``
        the error is raised: a ``ValueError`` for units Pint does not recognise,
        otherwise the Pint error (e.g. ``pint.DimensionalityError``).

    Returns
    -------
    numpy.ndarray
        The converted data, or ``data`` unchanged when the units are equal or
        the conversion is not possible and ``errors="ignore"``.
    """
    import pint

    if errors not in ("ignore", "raise"):
        raise ValueError(f"errors must be 'ignore' or 'raise', got {errors!r}")

    if isinstance(target_units, list):
        if any(are_compatible(a, b) for a, b in combinations(target_units, 2)):
            raise ValueError(f"Ambiguous target units, some are compatible: {target_units}")
        for units in target_units:
            try:
                return convert(data, units, source_units, errors="raise")
            except (ValueError, pint.errors.PintError):
                pass
        message = f"Cannot convert {source_units} to any of the target units {target_units}"
        if errors == "raise":
            raise ValueError(message)
        LOG.warning(message)
        return data

    source, target = Units.from_any(source_units), Units.from_any(target_units)
    if source == target:
        return data

    # Pint treats a None unit as dimensionless, which would silently give wrong results
    if source.to_pint() is None or target.to_pint() is None:
        message = f"Cannot convert between unrecognised units: {source_units} -> {target_units}"
        if errors == "raise":
            raise ValueError(message)
        LOG.warning(message)
        return data

    try:
        return get_registry().Quantity(np.asarray(data), source.to_pint()).to(target.to_pint()).magnitude
    except pint.errors.PintError as exc:
        if errors == "raise":
            raise
        LOG.warning(f"Cannot convert units: {exc}")
        return data
