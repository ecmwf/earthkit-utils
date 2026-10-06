# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The ``earthkit`` command line interface.

``earthkit`` is a single :mod:`click` entry point shared by the whole earthkit ecosystem. Any installed
earthkit package can contribute commands to it by declaring them in the ``earthkit.cli`` entry point
group of its ``pyproject.toml``::

    [project.entry-points."earthkit.cli"]
    ls = "earthkit.data.cli:ls"

The entry point name (``ls`` above) becomes the sub-command name, so the example is invoked as
``earthkit ls <file>``. The entry point value must resolve to a :class:`click.Command`. A
:class:`click.Group` is also a command, so a package can register a whole group of nested commands under a
single name (e.g. ``earthkit data ls <file>``).

Commands are discovered from the installed distributions and loaded on demand: ``earthkit ls`` imports
only the ``ls`` command, while ``earthkit --help`` imports every registered command to show its help text.
Packages should therefore keep heavy imports inside their command functions. A command that fails to
import is still listed, with a warning, so that one broken package does not take down the whole CLI.

Commands can also be attached directly to the :data:`earthkit` group in the usual click way::

    from earthkit.utils.cli import earthkit

    @earthkit.command()
    def hello():
        click.echo("hello")

but this only takes effect once the defining module has been imported, which is why packages should
prefer the entry point mechanism.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from importlib import metadata
from typing import Any, Iterable

import click

from earthkit.utils import __version__

__all__ = [
    "ENTRY_POINT_GROUP",
    "BrokenCommand",
    "EarthkitCLI",
    "discover_commands",
    "earthkit",
    "installed_earthkit_packages",
    "main",
]

#: Entry point group that earthkit packages use to register commands.
ENTRY_POINT_GROUP = "earthkit.cli"

#: Prefix identifying earthkit distributions on PyPI (``earthkit-data``, ``earthkit-plots``, ...).
_DISTRIBUTION_PREFIX = "earthkit"


def installed_earthkit_packages() -> dict[str, str]:
    """Return the installed earthkit distributions.

    Returns
    -------
    dict
        Mapping of distribution name (e.g. ``"earthkit-data"``) to installed version, sorted by name.
    """
    packages: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name and name.lower().startswith(_DISTRIBUTION_PREFIX):
            packages[name] = dist.version
    return dict(sorted(packages.items()))


def _iter_entry_points() -> Iterable[metadata.EntryPoint]:
    """Yield the raw ``earthkit.cli`` entry points of all installed distributions."""
    return metadata.entry_points(group=ENTRY_POINT_GROUP)


def discover_commands() -> dict[str, metadata.EntryPoint]:
    """Find the commands registered by installed packages.

    Returns
    -------
    dict
        Mapping of command name to the (not yet loaded) entry point providing it. If two distributions
        register the same command name, the first one found wins and a warning is emitted.
    """
    found: dict[str, metadata.EntryPoint] = {}
    for entry_point in _iter_entry_points():
        if entry_point.name in found:
            warnings.warn(
                f"Ignoring duplicate earthkit command '{entry_point.name}' from '{entry_point.value}'; "
                f"using '{found[entry_point.name].value}' instead",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        found[entry_point.name] = entry_point
    return found


def _distribution_name(entry_point: metadata.EntryPoint) -> str:
    dist = getattr(entry_point, "dist", None)
    if dist is None:
        return "<unknown>"
    return dist.metadata["Name"] or "<unknown>"


class BrokenCommand(click.Command):
    """Placeholder for a registered command that could not be loaded.

    It is listed in ``earthkit --help`` with a warning and raises a :class:`click.ClickException`
    describing the problem when invoked.
    """

    def __init__(self, name: str, entry_point: metadata.EntryPoint, error: BaseException) -> None:
        self.entry_point = entry_point
        self.error = error
        message = f"Warning: could not load '{entry_point.value}': {error!r}"
        super().__init__(name, help=message, short_help=message)

    def invoke(self, ctx: click.Context) -> Any:
        raise click.ClickException(
            f"Command '{self.name}' could not be loaded from '{self.entry_point.value}' "
            f"(provided by {_distribution_name(self.entry_point)}): {self.error!r}"
        )


class EarthkitCLI(click.Group):
    """A :class:`click.Group` whose commands are collected from installed earthkit packages.

    Commands added directly with :meth:`add_command` take precedence over commands registered through
    entry points. Entry point commands are loaded lazily, the first time they are looked up.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._plugins: dict[str, metadata.EntryPoint] | None = None

    @property
    def plugins(self) -> dict[str, metadata.EntryPoint]:
        """Commands registered through entry points, discovered on first access."""
        if self._plugins is None:
            self._plugins = discover_commands()
        return self._plugins

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted(set(self.commands) | set(self.plugins))

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        if cmd_name in self.commands:
            return self.commands[cmd_name]

        entry_point = self.plugins.get(cmd_name)
        if entry_point is None:
            return None

        command: click.Command
        try:
            loaded = entry_point.load()
        except Exception as error:
            command = BrokenCommand(cmd_name, entry_point, error)
        else:
            if isinstance(loaded, click.Command):
                command = loaded
            else:
                command = BrokenCommand(
                    cmd_name,
                    entry_point,
                    TypeError(f"expected a click.Command, got {type(loaded).__name__}"),
                )

        # Cache so the entry point is only loaded once per process.
        self.add_command(command, cmd_name)
        return command


@click.group(cls=EarthkitCLI, name="earthkit", context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-V", "--version", prog_name="earthkit-utils")
def earthkit() -> None:
    """Command line interface for the earthkit ecosystem.

    Commands are provided by the installed earthkit packages. Run 'earthkit info' to see which packages are
    installed and which commands each of them provides.
    """


@earthkit.command()
def info() -> None:
    """List the installed earthkit packages and the commands they provide."""
    commands_by_package: dict[str, list[str]] = defaultdict(list)
    for name, entry_point in discover_commands().items():
        commands_by_package[_distribution_name(entry_point)].append(name)

    packages = installed_earthkit_packages()
    for dist_name in commands_by_package:
        packages.setdefault(dist_name, "")

    if not packages:
        click.echo("No earthkit packages found.")
        return

    width = max(len(name) for name in packages)
    for dist_name, version in sorted(packages.items()):
        commands = ", ".join(sorted(commands_by_package.get(dist_name, [])))
        line = f"{dist_name:<{width}}  {version}"
        if commands:
            line += f"  (commands: {commands})"
        click.echo(line.rstrip())


def main() -> None:
    """Console script entry point for the ``earthkit`` command."""
    earthkit(prog_name="earthkit")
