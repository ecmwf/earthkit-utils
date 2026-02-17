# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
import warnings
from functools import wraps

__all__ = ["ExperimentalWarning", "experimental"]

_ENV_VAR = "EARTHKIT_EXPERIMENTAL_WARNINGS"

_DEFAULT_MESSAGE = "**Experimental API**: may change or be removed without notice."


class ExperimentalWarning(UserWarning):
    """Warning category for experimental API usage."""


def _env_warnings_enabled():
    """Return ``True`` unless the env var explicitly disables warnings."""
    value = os.environ.get(_ENV_VAR)
    if value is None:
        return True
    normalized = value.strip().lower()
    if normalized in {"", "0", "false", "no", "off"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    return True


def experimental(
    obj=None,
    *,
    message=_DEFAULT_MESSAGE,
    warn_runtime=True,
):
    """Mark a function as experimental.

    Prepends a Sphinx ``.. warning::`` directive to the function's
    docstring and, when *warn_runtime* is ``True``, emits an
    :class:`ExperimentalWarning` at runtime each time the function is
    called.

    Parameters
    ----------
    obj : callable or None
        When used as a bare decorator (``@experimental``), *obj* is the
        decorated function. When used with arguments
        (``@experimental(...)``), *obj* is ``None`` and a decorator is
        returned.
    message : str, optional
        Text inserted into the Sphinx ``.. warning::`` directive.
    warn_runtime : bool, optional
        If ``True`` (default), emit an :class:`ExperimentalWarning` on
        each call. If ``False``, only the docstring is modified and the
        original function is returned.

    Notes
    -----
    Runtime warnings can be silenced by setting
    ``EARTHKIT_EXPERIMENTAL_WARNINGS=0`` (or ``false``), or via
    ``warnings.filterwarnings("ignore", category=ExperimentalWarning)``.

    Examples
    --------
    >>> from earthkit.utils.decorators.experimental import experimental
    >>> @experimental
    ... def compute():
    ...     return 42
    ...
    """
    warning_block = f".. warning::\n   {message}\n"

    def decorate(target):
        doc = (target.__doc__ or "").strip()
        target.__doc__ = warning_block + ("\n" + doc if doc else "")

        if not warn_runtime:
            return target

        @wraps(target)
        def wrapper(*args, **kwargs):
            if _env_warnings_enabled():
                warnings.warn(
                    f"'{target.__qualname__}' is experimental and may "
                    "change or be removed without notice.",
                    category=ExperimentalWarning,
                    stacklevel=2,
                )
            return target(*args, **kwargs)

        return wrapper

    if obj is None:
        return decorate
    return decorate(obj)
