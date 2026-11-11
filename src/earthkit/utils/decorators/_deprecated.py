# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import warnings
from functools import wraps

__all__ = ["DeprecatedWarning", "deprecated"]

# This prefix enables silencing via PYTHONWARNINGS message filters, e.g.
#   PYTHONWARNINGS='ignore:earthkit deprecated:'
# Category‑based filters are brittle for non‑stdlib warnings and can fail
# easily (see https://github.com/python/cpython/issues/66733).
# A short, stable prefix ensures reliable re.match‑based filtering.
# In‑code filtering remains preferred:
#   warnings.filterwarnings("ignore", category=DeprecatedWarning)
_WARNING_PREFIX = "earthkit deprecated:"

_DEFAULT_DOCS_MESSAGE = "**Deprecated API**: will be removed in a future release."


class DeprecatedWarning(DeprecationWarning):
    """Warning category for deprecated API usage."""


def deprecated(
    obj=None,
    *,
    msg=_DEFAULT_DOCS_MESSAGE,
    replacement=None,
    removal_version=None,
    warn_runtime=True,
):
    """Mark a function as experimental.

    This signals to users that the decorated function is considered deprecated
    and will be removed in a future release.

    Prepends a Sphinx ``.. warning::`` directive to the function's
    docstring and, when *warn_runtime* is ``True``, emits an
    :class:`DeprecatedWarning` at runtime each time the function is
    called.

    Parameters
    ----------
    obj : callable or None
        When used as a bare decorator (``@deprecated``), *obj* is the
        decorated function. When used with arguments
        (``@deprecated(...)``), *obj* is ``None`` and a decorator is
        returned.
    msg : str, optional
        Text inserted into the Sphinx ``.. warning::`` directive.
        If multi-line, each line should be plain unindented text.
    replacement : str or None, optional
        Suggested replacement API.
    removal_version : str or None, optional
        Version in which the API is planned to be removed.
    warn_runtime : bool, optional
        If ``True`` (default), emit an :class:`DeprecatedWarning` on
        each call. If ``False``, only the docstring is modified and the
        original function is returned.

    Notes
    -----
    The Sphinx ``.. warning::`` directive is only rendered by documentation
    tools that import and execute the decorator during the build process.
    This works with ``sphinx.ext.autodoc`` but not with ``Sphinx AutoAPI``,
    which generates documentation from static code analysis without importing
    modules.

    To silence runtime warnings, use any of Python's standard mechanisms::

        import warnings
        from earthkit.utils.decorators import DeprecatedWarning
        warnings.filterwarnings("ignore", category=DeprecatedWarning)

    Or via the environment variable ``PYTHONWARNINGS``::

        export PYTHONWARNINGS="ignore:earthkit deprecated:"

    Examples
    --------
    >>> from earthkit.utils.decorators import deprecated
    >>> @deprecated
    ... def compute():
    ...     return 42
    ...

    """
    details = [msg]
    if replacement:
        details.append(f"Use {replacement} instead.")
    if removal_version:
        details.append(f"Planned removal in version {removal_version}.")
    msg = " ".join(details)

    # Every line of msg must be indented under the directive or reST silently drops it.
    indented = "\n".join("   " + line for line in msg.splitlines())
    warning_block = f".. warning::\n{indented}\n"

    def decorate(target):
        doc = (target.__doc__ or "").strip()
        target.__doc__ = warning_block + ("\n" + doc if doc else "")

        if not warn_runtime:
            return target

        @wraps(target)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{_WARNING_PREFIX} {target.__qualname__} may change or be removed without notice.",
                category=DeprecatedWarning,
                stacklevel=2,
            )
            return target(*args, **kwargs)

        return wrapper

    if obj is None:
        return decorate
    return decorate(obj)
