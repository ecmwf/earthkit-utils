# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import sys
import textwrap

import pytest
from click.testing import CliRunner

from earthkit.cli.main import earthkit

GOOD_MODULE = """
import click

from earthkit.cli.main import earthkit


@earthkit.command(name="zz-hello")
def hello():
    click.echo("hello")
"""

BROKEN_MODULE = """
import earthkit_missing_dependency_for_tests  # noqa: F401
"""


@pytest.fixture
def command_modules(tmp_path, monkeypatch):
    """Install command modules in the ``earthkit.cli`` namespace package, and reset the discovery around the test.

    Returns a function taking the source of each module by name.
    """
    monkeypatch.setattr(earthkit, "_discovered", False)
    monkeypatch.setattr(earthkit, "_failed", {})
    monkeypatch.setattr(earthkit, "commands", dict(earthkit.commands))
    installed = []

    def install(**modules):
        directory = tmp_path / "earthkit" / "cli"
        directory.mkdir(parents=True, exist_ok=True)
        for name, source in modules.items():
            (directory / f"{name}.py").write_text(textwrap.dedent(source))
            installed.append(f"earthkit.cli.{name}")
        monkeypatch.syspath_prepend(str(tmp_path))

    yield install

    for module in installed:
        sys.modules.pop(module, None)


def _runner():
    """Return a CliRunner capturing stderr separately from stdout, so ``result.stderr`` can be checked.

    This is the default from click 8.2, which removed the ``mix_stderr`` argument; before 8.2 stderr is mixed into
    stdout unless ``mix_stderr=False`` is given.
    """
    try:
        return CliRunner(mix_stderr=False)
    except TypeError:
        return CliRunner()


def _invoke(*args):
    return _runner().invoke(earthkit, list(args), prog_name="earthkit")


def test_broken_module_does_not_break_other_commands(command_modules):
    command_modules(zz_good=GOOD_MODULE, zz_broken=BROKEN_MODULE)

    result = _invoke("zz-hello")

    assert result.exit_code == 0, result.output
    assert result.stdout == "hello\n"
    assert result.stderr == (
        "Warning: the commands of earthkit-zz_broken could not be loaded: "
        "ModuleNotFoundError: No module named 'earthkit_missing_dependency_for_tests'\n"
    )


def test_broken_module_help(command_modules):
    command_modules(zz_good=GOOD_MODULE, zz_broken=BROKEN_MODULE)

    result = _invoke("-h")

    assert result.exit_code == 0, result.output
    assert "zz-hello" in result.stdout
    # The modules are imported once, although listing the commands looks each of them up
    assert result.stderr.count("could not be loaded") == 1


def test_broken_module_info(command_modules):
    command_modules(zz_good=GOOD_MODULE, zz_broken=BROKEN_MODULE)

    result = _invoke("info")

    assert result.exit_code == 0, result.output
    lines = {line.split()[0]: line for line in result.stdout.splitlines()}
    assert lines["earthkit-zz_good"].endswith("(commands: zz-hello)")
    assert lines["earthkit-zz_broken"].endswith(
        "(failed to load: ModuleNotFoundError: No module named 'earthkit_missing_dependency_for_tests')"
    )


def test_unknown_command_names_broken_packages(command_modules):
    command_modules(zz_good=GOOD_MODULE, zz_broken=BROKEN_MODULE)

    result = _invoke("zz-missing")

    assert result.exit_code == 2
    assert (
        "No such command 'zz-missing'. It may be provided by earthkit-zz_broken, whose commands could not be "
        "loaded." in result.stderr
    )


def test_unknown_command_without_broken_modules(command_modules):
    command_modules(zz_good=GOOD_MODULE)

    result = _invoke("zz-missing")

    assert result.exit_code == 2
    assert "No such command 'zz-missing'." in result.stderr
    assert "could not be loaded" not in result.stderr


def test_known_command_error_is_unchanged(command_modules):
    command_modules(zz_good=GOOD_MODULE, zz_broken=BROKEN_MODULE)

    result = _invoke("zz-hello", "--bogus")

    assert result.exit_code == 2
    assert "No such option" in result.stderr and "--bogus" in result.stderr
    assert "It may be provided by" not in result.stderr
