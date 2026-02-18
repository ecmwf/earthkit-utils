# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import warnings
from functools import wraps

__all__ = ["ExperimentalWarning", "experimental"]

# This prefix enables silencing via PYTHONWARNINGS message filters, e.g.
#   PYTHONWARNINGS='ignore:^earthkit experimental:'
# Category‑based filters are brittle for non‑stdlib warnings and can fail
# easily (see https://github.com/python/cpython/issues/66733).
# A short, stable prefix ensures reliable re.match‑based filtering.
# In‑code filtering remains preferred:
#   warnings.filterwarnings("ignore", category=ExperimentalWarning)
_WARNING_PREFIX = "earthkit experimental:"

_DEFAULT_DOCS_MESSAGE = "**Experimental API**: may change or be removed without notice."


class ExperimentalWarning(UserWarning):
    """Warning category for experimental API usage."""


def experimental(
    obj=None,
    *,
    msg=_DEFAULT_DOCS_MESSAGE,
    warn_runtime=True,
):
    """Mark a function as experimental.

    This signals to users that the decorated function is considered experimental
    and may change or be removed without notice.

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
    msg : str, optional
        Text inserted into the Sphinx ``.. warning::`` directive.
    warn_runtime : bool, optional
        If ``True`` (default), emit an :class:`ExperimentalWarning` on
        each call. If ``False``, only the docstring is modified and the
        original function is returned.

    Notes
    -----
    To silence runtime warnings, use any of Python's standard mechanisms::

        import warnings
        from earthkit.utils.decorators.experimental import ExperimentalWarning
        warnings.filterwarnings("ignore", category=ExperimentalWarning)

    Or via the environment variable ``PYTHONWARNINGS``::

        export PYTHONWARNINGS="ignore:earthkit experimental:"

    Examples
    --------
    >>> from earthkit.utils.decorators.experimental import experimental
    >>> @experimental
    ... def compute():
    ...     return 42
    ...
    """
    warning_block = f".. warning::\n   {msg}\n"

    def decorate(target):
        doc = (target.__doc__ or "").strip()
        target.__doc__ = warning_block + ("\n" + doc if doc else "")

        if not warn_runtime:
            return target

        @wraps(target)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{_WARNING_PREFIX} {target.__qualname__} " "may change or be removed without notice.",
                category=ExperimentalWarning,
                stacklevel=2,
            )
            return target(*args, **kwargs)

        return wrapper

    if obj is None:
        return decorate
    return decorate(obj)
