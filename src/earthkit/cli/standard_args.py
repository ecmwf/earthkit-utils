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
        target.to_target(source.from_source())

Sources and targets are given with a single ``--source [NAME:]VALUE`` and ``--target [NAME:]VALUE`` option,
split on the first ``:``. NAME is the earthkit-data source or target, ``file`` if not given. VALUE is
passed to a source as its first argument, e.g. a path or a URL, or, if it is a JSON object (or list of
objects), as its ``request``::

    earthkit copy --source in.grib --target out.grib  # same as file:in.grib and file:out.grib
    earthkit copy --source url:https://myhost.int/file.nc --target out.nc
    earthkit copy --source 'mars:{"param": "2t", "levtype": "sfc"}' --target out.grib

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

gives ``earthkit copy [OPTIONS] SOURCE TARGET``, e.g. ``earthkit copy in.grib 'zarr:{...}'``.

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
    "Source",
    "Target",
    "add_options",
    "profile_option",
    "source_options",
    "split_csv",
    "target_options",
]


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


def _parse_json(value: str, option: str, allow_list: bool) -> Any:
    """Decode a JSON object, or, if ``allow_list``, a list of objects."""
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise click.BadParameter(f"Invalid JSON: {error}", param_hint=f"'{option}'") from error
    if isinstance(decoded, dict):
        return decoded
    if allow_list and isinstance(decoded, list) and all(isinstance(d, dict) for d in decoded):
        return decoded
    expected = "a JSON object or a list of objects" if allow_list else "a JSON object"
    raise click.BadParameter(f"Expected {expected}", param_hint=f"'{option}'")


#: Key of a JSON request holding the first positional argument of the source, e.g. the CDS dataset.
REQUEST_DATASET_KEY = "dataset"


def _pop_dataset(request: Any, option: str) -> tuple[list[Any], Any]:
    """Remove the ``dataset`` key from a request (or list of requests) and return it as positional arguments."""
    requests = request if isinstance(request, list) else [request]
    datasets = {r[REQUEST_DATASET_KEY] for r in requests if REQUEST_DATASET_KEY in r}
    if not datasets:
        return [], request
    if len(datasets) > 1 or any(REQUEST_DATASET_KEY not in r for r in requests):
        raise click.BadParameter(f"All requests must have the same {REQUEST_DATASET_KEY!r}", param_hint=f"'{option}'")
    stripped = [{k: v for k, v in r.items() if k != REQUEST_DATASET_KEY} for r in requests]
    return [datasets.pop()], stripped if isinstance(request, list) else stripped[0]


def _import_earthkit_data() -> Any:
    try:
        import earthkit.data as ekd
    except ImportError:
        raise click.ClickException(
            "earthkit-data is required to read and write data, install it with 'pip install earthkit-data'"
        )
    return ekd


@dataclass(frozen=True)
class Source:
    """An earthkit-data source given on the command line.

    ``args`` starts with the name of the source, so the data is read with
    ``earthkit.data.from_source(*source.args, **source.kwargs)``, or simply ``source.from_source()``.
    """

    args: tuple[Any, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """The name of the earthkit-data source, e.g. 'file' or 'cds'."""
        return self.args[0]

    def from_source(self, **kwargs: Any) -> Any:
        """Read the data with :func:`earthkit.data.from_source`, with ``kwargs`` added to the source kwargs."""
        return _import_earthkit_data().from_source(*self.args, **{**self.kwargs, **kwargs})


@dataclass(frozen=True)
class Target:
    """An earthkit-data target given on the command line.

    ``args`` starts with the name of the target, so the data is written with
    ``earthkit.data.to_target(*target.args, data=data, **target.kwargs)``, or simply ``target.to_target(data)``.
    """

    args: tuple[Any, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """The name of the earthkit-data target, e.g. 'file' or 'zarr'."""
        return self.args[0]

    def to_target(self, data: Any, **kwargs: Any) -> None:
        """Write ``data`` with :func:`earthkit.data.to_target`, with ``kwargs`` added to the target kwargs."""
        _import_earthkit_data().to_target(*self.args, data=data, **{**self.kwargs, **kwargs})


#: Default earthkit-data source and target when the value has no ``NAME:``.
DEFAULT_NAME = "file"

_NAME_PATTERN = re.compile(r"[A-Za-z][\w-]*")


def _split_name(value: str) -> tuple[str, str]:
    """Split a ``[NAME:]VALUE`` string on its first ``:``, defaulting NAME to :data:`DEFAULT_NAME`.

    Anything before the first ``:`` that is not a valid name, e.g. a path containing a ``:``, is part of VALUE.
    """
    data_name, separator, payload = value.partition(":")
    if separator and _NAME_PATTERN.fullmatch(data_name):
        return data_name, payload
    return DEFAULT_NAME, value


def _is_json(value: str) -> bool:
    return value.lstrip().startswith(("{", "["))


def _parse_source(value: str, option: str) -> Source:
    """Turn a ``[NAME:]VALUE`` string into a :class:`Source`."""
    source_name, value = _split_name(value)
    if _is_json(value):
        dataset, request = _pop_dataset(_parse_json(value, option, allow_list=True), option)
        return Source((source_name, *dataset), {"request": request})
    if source_name == "file":
        _check_file("source", value, option)
    return Source((source_name, value) if value else (source_name,))


def _parse_target(value: str, option: str) -> Target:
    """Turn a ``[NAME:]VALUE`` string into a :class:`Target`."""
    target_name, value = _split_name(value)
    if _is_json(value):
        target_kwargs = _parse_json(value, option, allow_list=False)
        if target_name == "file" and isinstance(target_kwargs.get("file"), str):
            _check_file("target", target_kwargs["file"], option)
        return Target((target_name,), target_kwargs)
    if target_name == "file":
        _check_file("target", value, option)
    return Target((target_name, value) if value else (target_name,))


def _data_option(
    kind: str, name: str, positional: bool, parse: Callable[[str, str], Any], help: str
) -> Callable[[Any], Any]:
    if not name.isidentifier():
        raise ValueError(f"{kind} name must be a valid Python identifier, got {name!r}")
    if positional:
        # click arguments have no help, the command docstring describes them
        metavar = name.upper()
        return click.argument(name, metavar=metavar, callback=lambda ctx, param, value: parse(value, metavar))
    flag = "--" + name.replace("_", "-")
    return click.option(
        flag,
        name,
        required=True,
        metavar="[NAME:]VALUE",
        callback=lambda ctx, param, value: parse(value, flag),
        help=help,
    )


def source_options(name: str = "source", *, positional: bool = False) -> Callable[[Any], Any]:
    """Add an earthkit-data source to a command, passed to the command as the :class:`Source` ``name``.

    Adds the ``--name`` option, with dashes in ``name``, taking a ``[NAME:]VALUE`` string, e.g.
    ``file:/path/to/file.nc``, ``url:https://myhost.int/file.nc`` or ``mars:{"param": "2t"}``. Use a different
    ``name`` for each source of a command with several sources, e.g. ``source_1`` and ``source_2``.

    With ``positional=True``, adds the ``NAME`` positional argument instead, with ``name`` in upper case, e.g.
    ``SOURCE``. Describe it in the command docstring, as click arguments have no help.
    """
    return _data_option(
        "source",
        name,
        positional,
        _parse_source,
        "The earthkit-data source to read, as [NAME:]VALUE, with NAME 'file' if not given. "
        "VALUE is passed to the source as its first "
        "argument, or as its request if it is a JSON object, whose "
        f"'{REQUEST_DATASET_KEY}' key, if any, is passed as the first argument. E.g. '/path/to/file.nc' "
        "(glob patterns allowed), 'url:https://myhost.int/file.nc' or "
        '\'mars:{"param": "2t", "levtype": "sfc"}\'.',
    )


def target_options(name: str = "target", *, positional: bool = False) -> Callable[[Any], Any]:
    """Add an earthkit-data target to a command, passed to the command as the :class:`Target` ``name``.

    Adds the ``--name`` option, with dashes in ``name``, taking a ``[NAME:]VALUE`` string, e.g.
    ``file:/path/to/file.nc`` or ``zarr:{"xarray_to_zarr_kwargs": {"store": "out.zarr"}}``. Use a different
    ``name`` for each target of a command with several targets.

    With ``positional=True``, adds the ``NAME`` positional argument instead, with ``name`` in upper case, e.g.
    ``TARGET``. Describe it in the command docstring, as click arguments have no help.
    """
    return _data_option(
        "target",
        name,
        positional,
        _parse_target,
        "The earthkit-data target to write to, as [NAME:]VALUE, with NAME 'file' if not given. "
        "VALUE is passed to the target as its first "
        "argument, or as its keyword arguments if it is a JSON object. E.g. '/path/to/file.nc', "
        '\'file:{"file": "out.grib", "append": true}\' or '
        '\'zarr:{"xarray_to_zarr_kwargs": {"store": "out.zarr"}}\'.',
    )


def _check_file(kind: str, path: str, option: str) -> None:
    """Check the path of a source file exists (globs allowed), and the path of a target file is not a directory."""
    param_hint = f"'{option}'"
    if kind == "source":
        if not glob.glob(os.path.expanduser(path)):
            raise click.BadParameter(f"Path {path!r} does not exist.", param_hint=param_hint)
    elif os.path.isdir(path):
        raise click.BadParameter(f"Path {path!r} is a directory.", param_hint=param_hint)


#: The earthkit-data Xarray engine profile used when reading the source, passed as ``profile``.
profile_option = click.option(
    "--profile",
    default=None,
    help="Name of the earthkit Xarray engine profile used when opening GRIB data, e.g. 'mars'. "
    "Uses the earthkit-data default if not given.",
)
