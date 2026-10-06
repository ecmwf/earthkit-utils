# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The ``earthkit`` command line interface.

``earthkit`` is a single :mod:`click` group shared by the whole earthkit ecosystem. Each package listed in
:data:`EARTHKIT_PACKAGES` can contribute commands by defining a ``COMMANDS`` mapping of command name to
:class:`click.Command` in its ``earthkit.<package>.cli`` module::

    # earthkit/data/cli.py
    COMMANDS = {"ls": ls}

which makes ``earthkit ls <file>`` available once earthkit-data is installed. Packages that are not
installed, or have no ``cli`` module, are skipped. Importing ``earthkit.<package>.cli`` also imports the
package itself, so heavy imports should live inside the command functions.

Commands can also be attached directly to the :data:`earthkit` group::

    from earthkit.utils.cli import earthkit

    @earthkit.command()
    def hello():
        click.echo("hello")

but this only takes effect once the defining module has been imported.
"""

from __future__ import annotations

import importlib
import warnings
from importlib import metadata
from typing import Any, Iterable

import click

from earthkit.utils import __version__

__all__ = ["EARTHKIT_PACKAGES", "EarthkitCLI", "discover_commands", "earthkit", "load_commands", "main"]

#: The earthkit packages checked for CLI commands, in search order. Each is looked up as
#: ``earthkit.<package>.cli`` and is expected to define a ``COMMANDS`` mapping.
EARTHKIT_PACKAGES: tuple[str, ...] = (
    "data",
    "geo",
    "hydro",
    "meteo",
    "plots",
    "time",
    "climate",
    "transforms",
    "workflows",
)


def load_commands(package: str) -> dict[str, click.Command]:
    """
    Return the commands provided by ``earthkit.<package>.cli``.

    An empty mapping is returned if the package is not installed or has no ``cli`` module. Any other
    import error is propagated.
    """
    module_name = f"earthkit.{package}.cli"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return {}
    
    commands = getattr(module, "COMMANDS", {})
    for name, command in commands.items():
        if not isinstance(command, click.Command):
            raise TypeError(f"{module_name}.COMMANDS[{name!r}] is not a click.Command: {command!r}")
    return dict(commands)


def discover_commands(packages: Iterable[str] = EARTHKIT_PACKAGES) -> dict[str, click.Command]:
    """
    Collect the commands of all installed earthkit packages.

    If two packages provide the same command name, the first one in ``packages`` wins and a warning is
    emitted.
    """
    found: dict[str, click.Command] = {}
    for package in packages:
        for name, command in load_commands(package).items():
            if name in found:
                warnings.warn(f"Ignoring duplicate earthkit command '{name}' from earthkit-{package}", stacklevel=2)
                continue
            found[name] = command
    return found


class EarthkitCLI(click.Group):
    """
    A :class:`click.Group` that adds the commands of the installed earthkit packages on first use.

    Commands added directly with :meth:`add_command` take precedence over discovered ones.
    """

    def __init__(self, *args: Any, packages: Iterable[str] = EARTHKIT_PACKAGES, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.packages = tuple(packages)
        self._discovered = False

    def _discover(self) -> None:
        if not self._discovered:
            self._discovered = True
            for name, command in discover_commands(self.packages).items():
                self.commands.setdefault(name, command)

    def list_commands(self, ctx: click.Context) -> list[str]:
        self._discover()
        return super().list_commands(ctx)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        self._discover()
        return super().get_command(ctx, cmd_name)


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
    for package in ("utils", *EARTHKIT_PACKAGES):
        try:
            version = metadata.version(f"earthkit-{package}")
        except metadata.PackageNotFoundError:
            continue
        line = f"earthkit-{package:<12} {version}"
        commands = ", ".join(sorted(load_commands(package))) if package != "utils" else ""
        if commands:
            line += f"  (commands: {commands})"
        click.echo(line)


def main() -> None:
    """Console script entry point for the ``earthkit`` command."""
    earthkit(prog_name="earthkit")
