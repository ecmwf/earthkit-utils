# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Standard arguments, options and helpers shared by the commands of the ``earthkit`` command line interface.

Using these keeps the commands contributed by the earthkit packages consistent::

    import click

    from earthkit.cli.main import earthkit
    from earthkit.cli.standard_args import add_options, source_file_argument, target_file_argument


    @earthkit.command()
    @add_options([source_file_argument, target_file_argument])
    def copy(source_file, target_file): ...

The arguments and options are plain :mod:`click` decorators, so they can be reused on any number of commands.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import click

__all__ = [
    "add_options",
    "profile_option",
    "source_file_argument",
    "split_csv",
    "target_file_argument",
]


def split_csv(ctx: click.Context, param: click.Parameter, value: Iterable[str] | None) -> list[str] | None:
    """Turn a tuple of (possibly comma-separated) option values into a flat list.

    Use as the ``callback`` of an option with ``multiple=True``, so that ``-k a,b -k c`` gives ``["a", "b", "c"]``.
    """
    if not value:
        return None
    result = []
    for item in value:
        result.extend(v.strip() for v in item.split(",") if v.strip())
    return result or None


def add_options(options: Iterable[Callable[[Any], Any]]) -> Callable[[Any], Any]:
    """Apply a list of click arguments/options to a command, in the order they are listed."""
    options = list(options)

    def _decorator(func: Any) -> Any:
        for option in reversed(options):
            func = option(func)
        return func

    return _decorator


#: The file a command reads from, passed to the command as ``source_file``.
source_file_argument = click.argument("source-file", type=click.Path(exists=True, dir_okay=False))

#: The file a command writes to, passed to the command as ``target_file``.
target_file_argument = click.argument("target-file", type=click.Path(dir_okay=False, writable=True))

#: The earthkit-data Xarray engine profile used when reading the source file, passed as ``profile``.
profile_option = click.option(
    "--profile",
    default=None,
    help="Name of the earthkit Xarray engine profile used when opening GRIB data, e.g. 'mars'. "
    "Uses the earthkit-data default if not given.",
)
