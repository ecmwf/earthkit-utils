# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The ``earthkit`` command line interface.

``earthkit`` is a single :mod:`click` group shared by the whole earthkit ecosystem. ``earthkit.cli`` is a
namespace package (it has no ``__init__.py``) that is shared by all earthkit packages: each package contributes
commands by installing a module ``earthkit/cli/<name>.py`` that registers them on the group::

    # earthkit/cli/data.py (shipped by earthkit-data)
    import click

    from earthkit.cli.main import earthkit

    @earthkit.command()
    @click.argument("filename")
    def ls(filename):
        ...

which makes ``earthkit ls <file>`` available once earthkit-data is installed. All modules of ``earthkit.cli``
are imported when the commands are needed (e.g. for ``earthkit -h``), which runs the decorators. Modules
starting with an underscore, and the modules of earthkit-utils itself (``main`` and ``standard_args``), are
ignored. Commands should use the shared arguments and options of :mod:`earthkit.cli.standard_args` where they fit.

A package with many commands can instead install a sub-package ``earthkit/cli/<name>/``, whose
``__init__.py`` imports the submodules that register the commands. Only ``earthkit.cli.<name>`` itself is
imported by the discovery, and the commands of all its submodules are attributed to ``earthkit-<name>``.

Because these modules live outside of the ``earthkit.<package>`` packages, importing them does not run the
``__init__`` of those packages. They should only import :mod:`click` and light standard library modules at
module level and import any heavy dependency (including the contributing package itself) inside the command
functions, so that ``earthkit -h`` stays fast.
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings
from importlib import metadata
from typing import Any

import click

__all__ = ["EarthkitCLI", "command_modules", "earthkit", "main"]

#: Modules of ``earthkit.cli`` that are not command modules.
_RESERVED = frozenset({"main", "standard_args"})


def command_modules() -> list[str]:
    """Return the sorted names of the ``earthkit.cli`` modules, as merged from all installed packages."""
    import earthkit.cli as namespace

    return sorted(
        info.name
        for info in pkgutil.iter_modules(list(namespace.__path__))
        if not info.name.startswith("_") and info.name not in _RESERVED
    )


class EarthkitCLI(click.Group):
    """
    A :class:`click.Group` that imports all ``earthkit.cli`` modules on first use.

    The modules register their commands on the group when they are imported. If two commands are registered
    under the same name, the first one wins and a warning is emitted.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._discovered = False

    def add_command(self, cmd: click.Command, name: str | None = None) -> None:
        key = name or cmd.name
        if key in self.commands and self.commands[key] is not cmd:
            origin = getattr(cmd.callback, "__module__", None) or "unknown module"
            warnings.warn(f"Ignoring duplicate earthkit command '{key}' from {origin}", stacklevel=2)
            return
        super().add_command(cmd, name)

    def _discover(self) -> None:
        if not self._discovered:
            self._discovered = True
            for name in command_modules():
                importlib.import_module(f"earthkit.cli.{name}")

    def list_commands(self, ctx: click.Context) -> list[str]:
        self._discover()
        return super().list_commands(ctx)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        self._discover()
        return super().get_command(ctx, cmd_name)


@click.group(cls=EarthkitCLI, name="earthkit", context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(None, "-V", "--version", package_name="earthkit-utils", prog_name="earthkit-utils")
def earthkit() -> None:
    """Command line interface for the earthkit ecosystem.

    Commands are provided by the installed earthkit packages. Run 'earthkit info' to see which packages are
    installed and which commands each of them provides.
    """


@earthkit.command()
def info() -> None:
    """List the installed earthkit packages and the commands they provide."""
    earthkit._discover()
    owners: dict[str, list[str]] = {}
    for cmd_name, command in earthkit.commands.items():
        module = getattr(command.callback, "__module__", None) or ""
        # Commands may be defined in submodules of a command package, e.g. earthkit.cli.transforms.temporal
        owners.setdefault(module.removeprefix("earthkit.cli.").split(".")[0], []).append(cmd_name)

    for name in ("utils", *command_modules()):
        try:
            version = metadata.version(f"earthkit-{name}")
        except metadata.PackageNotFoundError:
            version = "unknown"
        line = f"earthkit-{name:<12} {version}"
        commands = ", ".join(sorted(owners.get(name, []))) if name != "utils" else ""
        if commands:
            line += f"  (commands: {commands})"
        click.echo(line)


def main() -> None:
    """Console script entry point for the ``earthkit`` command."""
    earthkit(prog_name="earthkit")


if __name__ == "__main__":
    # Run the importable module, so that the command modules register on the same group that runs here.
    from earthkit.cli.main import main as _main

    _main()
