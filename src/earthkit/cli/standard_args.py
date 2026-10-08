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
    from earthkit.cli.standard_args import add_options, source_options, target_options


    @earthkit.command()
    @add_options([source_options(), target_options()])
    def copy(source, target):
        target.to_target(source)

Sources and targets are given as ``[NAME:]VALUE``, split on the first ``:``. NAME is the earthkit-data source
or target, ``file`` if not given. The command gets each source as the object returned by
:func:`earthkit.data.from_source`, and each target as a :class:`Target`.

VALUE is passed to a source as its first argument, e.g. a path or a URL, or, if it is a JSON object (or list of
objects), as its ``request``. Several sources are merged into a single ``multi`` source::

    earthkit copy --source in.grib --target out.grib  # same as file:in.grib and file:out.grib
    earthkit copy --source url:https://myhost.int/file.nc --target out.nc
    earthkit copy --source 'mars:{"param": "2t", "levtype": "sfc"}' --target out.grib
    earthkit copy --source a.grib --source b.grib --target out.grib

The ``dataset`` key of a JSON request, if present, is passed to the source as its first argument, e.g. for
the CDS::

    --source 'cds:{"dataset": "reanalysis-era5-single-levels", "variable": "2m_temperature", "year": "2020"}'

VALUE is passed to a target as its first argument, e.g. a path, or, if it is a JSON object, as its keyword
arguments::

    --target 'file:{"file": "out.grib", "append": true}'
    --target 'zarr:{"xarray_to_zarr_kwargs": {"store": "out.zarr"}}'

Commands with several sources or targets name each of them, which also names their options::

    @earthkit.command()
    @add_options([source_options("source_1"), source_options("source_2"), target_options()])
    def combine(source_1, source_2, target): ...

gives the ``--source-1``, ``--source-2`` and ``--target`` options.

A command can take its sources and targets as positional arguments instead, with ``positional=True``::

    @earthkit.command()
    @add_options([source_options(positional=True), target_options(positional=True)])
    def copy(source, target): ...

gives ``earthkit copy [OPTIONS] SOURCE... TARGET``, e.g. ``earthkit copy a.grib b.grib 'zarr:{...}'``.

The arguments and options are plain :mod:`click` decorators, so they can be reused on any number of commands.
"""

from __future__ import annotations

import glob
import json
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import click

__all__ = [
    "Target",
    "add_options",
    "profile_option",
    "source_options",
    "split_csv",
    "target_options",
]

#: Default earthkit-data source and target when the value has no ``NAME:``.
DEFAULT_NAME = "file"

#: Key of a JSON request holding the first positional argument of the source, e.g. the CDS dataset.
REQUEST_DATASET_KEY = "dataset"

_NAME_PATTERN = re.compile(r"[A-Za-z][\w-]*")


def split_csv(ctx: click.Context, param: click.Parameter, value: Iterable[str] | None) -> list[str] | None:
    """Turn a tuple of (possibly comma-separated) option values into a flat list.

    Use as the ``callback`` of an option with ``multiple=True``, so that ``-k a,b -k c`` gives
    ``["a", "b", "c"]``.
    """
    if not value:
        return None
    result: list[str] = []
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


def _earthkit_data() -> Any:
    try:
        import earthkit.data as ekd
    except ImportError:
        raise click.ClickException(
            "earthkit-data is required to read and write data, install it with 'pip install earthkit-data'"
        )
    return ekd


@dataclass(frozen=True)
class Target:
    """An earthkit-data target given on the command line, written with :meth:`to_target`."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)

    def to_target(self, data: Any, **kwargs: Any) -> None:
        """Write ``data`` with :func:`earthkit.data.to_target`, with ``kwargs`` added to the target kwargs."""
        _earthkit_data().to_target(*self.args, data=data, **{**self.kwargs, **kwargs})


def _split_name(value: str) -> tuple[str, str]:
    """Split a ``[NAME:]VALUE`` string on its first ``:``, defaulting NAME to :data:`DEFAULT_NAME`.

    Anything before the first ``:`` that is not a valid name, e.g. a path containing a ``:``, is part of VALUE.
    """
    name, separator, value_ = value.partition(":")
    if separator and _NAME_PATTERN.fullmatch(name):
        return name, value_
    return DEFAULT_NAME, value


def _parse_json(value: str, allow_list: bool) -> Any:
    """Decode VALUE if it is a JSON object, or, if ``allow_list``, a list of objects, otherwise return None."""
    if not value.lstrip().startswith(("{", "[")):
        return None
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise click.BadParameter(f"Invalid JSON: {error}") from error
    if isinstance(decoded, dict):
        return decoded
    if allow_list and isinstance(decoded, list) and all(isinstance(d, dict) for d in decoded):
        return decoded
    raise click.BadParameter("Expected a JSON object or a list of objects" if allow_list else "Expected a JSON object")


def _pop_dataset(request: Any) -> tuple[list[Any], Any]:
    """Remove the ``dataset`` key from a request (or list of requests) and return it as positional arguments."""
    requests = request if isinstance(request, list) else [request]
    datasets = {r[REQUEST_DATASET_KEY] for r in requests if REQUEST_DATASET_KEY in r}
    if not datasets:
        return [], request
    if len(datasets) > 1 or any(REQUEST_DATASET_KEY not in r for r in requests):
        raise click.BadParameter(f"All requests must have the same {REQUEST_DATASET_KEY!r}")
    stripped = [{k: v for k, v in r.items() if k != REQUEST_DATASET_KEY} for r in requests]
    return [datasets.pop()], stripped if isinstance(request, list) else stripped[0]


def _check_path(path: str, exists: bool) -> None:
    """Check a source path exists (globs allowed), or a target path is not a directory."""
    if exists and not glob.glob(os.path.expanduser(path)):
        raise click.BadParameter(f"Path {path!r} does not exist.")
    if not exists and os.path.isdir(path):
        raise click.BadParameter(f"Path {path!r} is a directory.")


def _parse_source(value: str) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Turn a ``[NAME:]VALUE`` string into the arguments of :func:`earthkit.data.from_source`."""
    name, value = _split_name(value)
    request = _parse_json(value, allow_list=True)
    if request is not None:
        dataset, request = _pop_dataset(request)
        return (name, *dataset), {"request": request}
    if name == "file":
        _check_path(value, exists=True)
    return ((name, value) if value else (name,)), {}


def _open_sources(ctx: click.Context, param: click.Parameter, values: tuple[str, ...]) -> Any:
    """Open the sources with earthkit-data, merging several of them into a ``multi`` source."""
    ekd = _earthkit_data()
    sources = [ekd.from_source(*args, **kwargs) for args, kwargs in values]
    return sources[0] if len(sources) == 1 else ekd.from_source("multi", *sources)


def _parse_target(value: str) -> Target:
    """Turn a ``[NAME:]VALUE`` string into a :class:`Target`."""
    name, value = _split_name(value)
    kwargs = _parse_json(value, allow_list=False)
    if kwargs is not None:
        if name == "file" and isinstance(kwargs.get("file"), str):
            _check_path(kwargs["file"], exists=False)
        return Target((name,), kwargs)
    if name == "file":
        _check_path(value, exists=False)
    return Target((name, value) if value else (name,))


def _data_param(name: str, positional: bool, help: str, **attrs: Any) -> Callable[[Any], Any]:
    """Return the click argument or option ``name``, converting ``[NAME:]VALUE`` strings with ``attrs``."""
    if not name.isidentifier():
        raise ValueError(f"name must be a valid Python identifier, got {name!r}")
    if positional:
        # click arguments have no help, the command docstring describes them
        return click.argument(name, metavar=name.upper(), required=True, **attrs)
    return click.option("--" + name.replace("_", "-"), name, required=True, metavar="[NAME:]VALUE", help=help, **attrs)


def source_options(name: str = "source", *, positional: bool = False) -> Callable[[Any], Any]:
    """Add earthkit-data sources to a command, passed to the command as ``name``.

    Adds the repeatable ``--name`` option, with dashes in ``name``, taking a ``[NAME:]VALUE`` string, e.g.
    ``file:/path/to/file.nc``, ``url:https://myhost.int/file.nc`` or ``mars:{"param": "2t"}``. The command gets
    the object returned by :func:`earthkit.data.from_source`, with several sources merged into a ``multi``
    source. Use a different ``name`` for each source of a command with several sources, e.g. ``source_1``
    and ``source_2``.

    With ``positional=True``, adds the ``NAME...`` positional argument instead, with ``name`` in upper case,
    e.g. ``SOURCE``, taking one or more values. A command can only have one positional source, and must
    describe it in its docstring, as click arguments have no help.
    """
    return _data_param(
        name,
        positional,
        "The earthkit-data source to read, as [NAME:]VALUE, with NAME 'file' if not given. VALUE is passed to "
        "the source as its first argument, or as its request if it is a JSON object, whose "
        f"'{REQUEST_DATASET_KEY}' key, if any, is passed as the first argument. E.g. '/path/to/file.nc' "
        "(glob patterns allowed), 'url:https://myhost.int/file.nc' or "
        '\'mars:{"param": "2t", "levtype": "sfc"}\'. Can be repeated to merge several sources.',
        type=_parse_source,
        callback=_open_sources,
        **({"nargs": -1} if positional else {"multiple": True}),
    )


def target_options(name: str = "target", *, positional: bool = False) -> Callable[[Any], Any]:
    """Add an earthkit-data target to a command, passed to the command as the :class:`Target` ``name``.

    Adds the ``--name`` option, with dashes in ``name``, taking a ``[NAME:]VALUE`` string, e.g.
    ``file:/path/to/file.nc`` or ``zarr:{"xarray_to_zarr_kwargs": {"store": "out.zarr"}}``. Use a different
    ``name`` for each target of a command with several targets.

    With ``positional=True``, adds the ``NAME`` positional argument instead, with ``name`` in upper case, e.g.
    ``TARGET``. Describe it in the command docstring, as click arguments have no help.
    """
    return _data_param(
        name,
        positional,
        "The earthkit-data target to write to, as [NAME:]VALUE, with NAME 'file' if not given. VALUE is passed "
        "to the target as its first argument, or as its keyword arguments if it is a JSON object. E.g. "
        '\'/path/to/file.nc\', \'file:{"file": "out.grib", "append": true}\' or '
        '\'zarr:{"xarray_to_zarr_kwargs": {"store": "out.zarr"}}\'.',
        type=_parse_target,
    )


#: The earthkit-data Xarray engine profile used when reading the source, passed as ``profile``.
profile_option = click.option(
    "--profile",
    default=None,
    help="Name of the earthkit Xarray engine profile used when opening GRIB data, e.g. 'mars'. "
    "Uses the earthkit-data default if not given.",
)
