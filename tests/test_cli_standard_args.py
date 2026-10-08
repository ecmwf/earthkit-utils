# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import click
import pytest
from click.testing import CliRunner

from earthkit.cli.main import command_modules
from earthkit.cli.standard_args import (
    add_options,
    profile_option,
    source_file_argument,
    split_csv,
    target_file_argument,
)


@click.command()
@add_options([source_file_argument, target_file_argument, profile_option])
@click.option("-k", "--keys", multiple=True, callback=split_csv)
def _command(source_file, target_file, profile, keys):
    click.echo(repr((source_file, target_file, profile, keys)))


@pytest.mark.parametrize(
    "value, expected",
    (
        ((), None),
        (("a",), ["a"]),
        (("a, b", "c"), ["a", "b", "c"]),
        ((",",), None),
    ),
)
def test_split_csv(value, expected):
    assert split_csv(None, None, value) == expected


def test_standard_args_usage():
    result = CliRunner().invoke(_command, ["--help"])
    assert result.exit_code == 0
    assert "[OPTIONS] SOURCE_FILE TARGET_FILE" in result.output
    assert "--profile" in result.output


def test_standard_args_values(tmp_path):
    source = tmp_path / "in.grib"
    source.touch()
    target = tmp_path / "out.nc"
    result = CliRunner().invoke(_command, [str(source), str(target), "--profile", "mars", "-k", "a,b", "-k", "c"])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == repr((str(source), str(target), "mars", ["a", "b", "c"]))


def test_standard_args_missing_source(tmp_path):
    result = CliRunner().invoke(_command, [str(tmp_path / "missing.grib"), str(tmp_path / "out.nc")])
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_standard_args_is_not_a_command_module():
    assert "standard_args" not in command_modules()
