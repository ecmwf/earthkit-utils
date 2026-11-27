# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import sys
from typing import Any, NamedTuple

import click
import pytest
from click.testing import CliRunner

from earthkit.cli import standard_args
from earthkit.cli.main import command_modules
from earthkit.cli.standard_args import (
    SOURCE_HELP,
    TARGET_HELP,
    Target,
    _earthkit_data,
    add_options,
    index_option,
    parse_index,
    profile_option,
    source_options,
    split_csv,
    target_options,
)


class Call(NamedTuple):
    """The arguments of a call to the fake earthkit-data."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any]


def call(*args, **kwargs):
    return Call(args, kwargs)


class FakeEarthkitData:
    """Stands for earthkit.data: ``from_source`` returns its arguments, ``to_target`` records them."""

    def __init__(self):
        self.written = []

    def from_source(self, *args, **kwargs):
        return Call(args, kwargs)

    def to_target(self, *args, **kwargs):
        self.written.append(Call(args, kwargs))


@pytest.fixture(autouse=True)
def ekd(monkeypatch):
    """Replace earthkit-data with a fake where standard_args gets it, so no test reads or writes data."""
    fake = FakeEarthkitData()
    monkeypatch.setattr(standard_args, "_earthkit_data", lambda: fake)
    return fake


@pytest.fixture(autouse=True)
def in_tmp_path(tmp_path, monkeypatch):
    """Run each test in a directory holding the source files used in the tests, so they have relative paths."""
    monkeypatch.chdir(tmp_path)
    for name in ("in.grib", "a:b.grib", "2020-01-01T00:00.grib", ":in.grib"):
        (tmp_path / name).touch()


def standard():
    return [source_options(), target_options()]


def named():
    return [source_options("source_1"), source_options("source_2"), target_options()]


def positional():
    return [source_options("source_1", positional=True), source_options("source_2"), target_options(positional=True)]


def run(*args, options=standard):
    """Invoke a command made of the arguments and options returned by ``options()`` on ``args``.

    Returns the click result and the keyword arguments the command was called with.
    """
    received = {}

    @click.command()
    @add_options(options())
    def command(**kwargs):
        received.update(kwargs)

    return CliRunner().invoke(command, list(args)), received


def received_by(*args, options=standard):
    """Return the keyword arguments of the command invoked on ``args``, which must succeed."""
    result, received = run(*args, options=options)
    assert result.exit_code == 0, result.output
    return received


def help_text(options=standard):
    return run("--help", options=options)[0].output


# ---------------------------------------------------------------------------- helpers


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


@pytest.mark.parametrize("value, expected", (("3", 3), ("-2", -2), ("0,2", [0, 2]), ("1:5:2", slice(1, 5, 2))))
def test_parse_index(value, expected):
    assert parse_index(value) == expected


def test_parse_index_invalid():
    with pytest.raises(click.BadParameter):
        parse_index("x")


@pytest.mark.parametrize(
    "args, expected",
    (
        ([], None),
        (["-p", "mars"], ["mars"]),
        (["--profile", "earthkit", "-p", "/my/profile.yaml"], ["earthkit", "/my/profile.yaml"]),
    ),
)
def test_profile_option(args, expected):
    assert received_by(*args, options=lambda: [profile_option]) == {"profile": expected}


@pytest.mark.parametrize(
    "args, expected",
    (
        ([], None),
        (["-i", "3"], 3),
        (["-i", "-1"], -1),
        (["-i", "4,5,8"], [4, 5, 8]),
        (["-i", "1:7"], slice(1, 7)),
        (["-i", "::2"], slice(None, None, 2)),
        (["--index", "2:"], slice(2, None)),
    ),
)
def test_index_option(args, expected):
    assert received_by(*args, options=lambda: [index_option]) == {"index": expected}


@pytest.mark.parametrize("value", ("a", "1,a", "1:2:3:4", "1.5"))
def test_index_option_invalid(value):
    result, _ = run("--index", value, options=lambda: [index_option])
    assert result.exit_code == 2
    assert "Invalid value for '-i' / '--index'" in result.output


# ---------------------------------------------------------------------------- usage


def test_usage():
    usage = help_text()
    assert "[OPTIONS]\n" in usage
    assert "-s, --source [NAME:]VALUE" in usage
    assert "-t, --target [NAME:]VALUE" in usage
    # click may wrap the help at hyphens, e.g. "earthkit-" / "data", so compare without whitespace
    for text in (SOURCE_HELP, TARGET_HELP):
        assert "".join(text.split()) in "".join(usage.split())


def test_usage_named():
    usage = help_text(named)
    for option in ("--source-1 [NAME:]VALUE", "--source-2 [NAME:]VALUE", "-t, --target [NAME:]VALUE"):
        assert option in usage
    # only the default source and target have short flags
    assert "-s," not in usage


def test_usage_positional():
    usage = help_text(positional)
    assert "[OPTIONS] SOURCE_1 TARGET\n" in usage
    assert "--source-2 [NAME:]VALUE" in usage
    assert "--source-1" not in usage and "--target " not in usage


@pytest.mark.parametrize("name", ("", "source-1", "1source"))
@pytest.mark.parametrize("is_positional", (False, True))
def test_invalid_name(name, is_positional):
    with pytest.raises(ValueError):
        source_options(name, positional=is_positional)


# ---------------------------------------------------------------------------- sources and targets


@pytest.mark.parametrize(
    "value, expected",
    (
        ("in.grib", call("file", "in.grib")),
        ("file:in.grib", call("file", "in.grib")),
        ("file:*.grib", call("file", "*.grib")),
        # text before the first ':' that is not a valid name belongs to the path
        ("./a:b.grib", call("file", "./a:b.grib")),
        ("2020-01-01T00:00.grib", call("file", "2020-01-01T00:00.grib")),
        (":in.grib", call("file", ":in.grib")),
        # only the first ':' separates the name from the value
        ("url:https://myhost.int/file.nc", call("url", "https://myhost.int/file.nc")),
        ("dummy:", call("dummy")),
        ('mars:{"param": "2t", "levtype": "sfc"}', call("mars", request={"param": "2t", "levtype": "sfc"})),
        ('cds:{"dataset": "era5", "year": "2020"}', call("cds", "era5", request={"year": "2020"})),
        (
            'cds:[{"dataset": "era5", "year": "2020"}, {"dataset": "era5", "year": "2021"}]',
            call("cds", "era5", request=[{"year": "2020"}, {"year": "2021"}]),
        ),
    ),
)
def test_source(value, expected):
    assert received_by("--source", value, "--target", "out.nc")["source"] == expected


@pytest.mark.parametrize(
    "value, expected",
    (
        ("out.nc", Target(("file", "out.nc"))),
        ("file:out.nc", Target(("file", "out.nc"))),
        ('{"file": "out.grib", "append": true}', Target(("file",), {"file": "out.grib", "append": True})),
        ('file:{"file": "out.grib", "append": true}', Target(("file",), {"file": "out.grib", "append": True})),
        (
            'zarr:{"xarray_to_zarr_kwargs": {"store": "out.zarr"}}',
            Target(("zarr",), {"xarray_to_zarr_kwargs": {"store": "out.zarr"}}),
        ),
        ("dummy:", Target(("dummy",))),
    ),
)
def test_target(value, expected):
    assert received_by("--source", "in.grib", "--target", value)["target"] == expected


def test_short_flags():
    assert received_by("-s", "in.grib", "-t", "out.nc") == {
        "source": call("file", "in.grib"),
        "target": Target(("file", "out.nc")),
    }


def test_sources_are_merged():
    received = received_by("--source", "in.grib", "--source", "url:https://myhost.int/file.nc", "--target", "o.nc")
    assert received["source"] == call("multi", call("file", "in.grib"), call("url", "https://myhost.int/file.nc"))


def test_named():
    received = received_by(
        "--source-1", "in.grib", "--source-2", 'cds:{"dataset": "era5", "year": "2020"}', "-t", "out.nc", options=named
    )
    assert received == {
        "source_1": call("file", "in.grib"),
        "source_2": call("cds", "era5", request={"year": "2020"}),
        "target": Target(("file", "out.nc")),
    }


def test_positional():
    received = received_by("in.grib", "in.grib", "zarr:{}", "--source-2", "in.grib", options=positional)
    assert received == {
        "source_1": call("multi", call("file", "in.grib"), call("file", "in.grib")),
        "source_2": call("file", "in.grib"),
        "target": Target(("zarr",), {}),
    }


def test_stdio():
    @click.command()
    @add_options([source_options(positional=True), target_options(positional=True)])
    def command(source, target):
        # stdin and stdout are only replaced by the runner while the command runs
        assert source == call("stream", sys.stdin.buffer)
        assert target == Target(("file", sys.stdout.buffer))

    result = CliRunner().invoke(command, ["-", "-"])
    assert result.exit_code == 0, result.output + repr(result.exception)


@pytest.mark.parametrize(
    "options, args, message",
    (
        (standard, ["--target", "out.nc"], "Missing option '-s' / '--source'"),
        (standard, ["--source", "in.grib"], "Missing option '-t' / '--target'"),
        (standard, ["--source", "missing.grib", "--target", "out.nc"], "does not exist"),
        (standard, ["--source", "file:missing.grib", "--target", "out.nc"], "does not exist"),
        (standard, ["--source", "in.grib", "--target", "."], "is a directory"),
        (standard, ["--source", "in.grib", "--target", "file:."], "is a directory"),
        (standard, ["--source", "in.grib", "--target", 'file:{"file": "."}'], "is a directory"),
        (standard, ["--source-file", "in.grib", "--target", "out.nc"], "No such option"),
        (standard, ["--source", "in.grib", "--target-file", "out.nc"], "No such option"),
        (standard, ["--source", 'mars:{"param": ', "--target", "out.nc"], "Invalid JSON"),
        (standard, ["--source", "mars:[1, 2]", "--target", "out.nc"], "Expected a JSON object or a list of objects"),
        (standard, ["--source", "in.grib", "--target", 'zarr:[{"a": 1}]'], "Expected a JSON object"),
        (standard, ["--source", 'cds:[{"dataset": "a"}, {"dataset": "b"}]', "-t", "out.nc"], "same 'dataset'"),
        (standard, ["--source", 'cds:[{"dataset": "a"}, {"year": "2020"}]', "-t", "out.nc"], "same 'dataset'"),
        (positional, ["--source-2", "in.grib"], "Missing argument 'SOURCE_1'"),
        # a single value is taken by TARGET
        (positional, ["in.grib", "--source-2", "in.grib"], "Missing argument 'SOURCE_1'"),
        (positional, ["missing.grib", "out.nc", "--source-2", "in.grib"], "Invalid value for 'SOURCE_1'"),
    ),
)
def test_invalid(options, args, message):
    result, received = run(*args, options=options)
    assert result.exit_code == 2, result.output
    assert message in result.output
    assert received == {}


# ---------------------------------------------------------------------------- earthkit-data


def test_target_to_target(ekd):
    Target(("file", "out.grib"), {"append": False}).to_target("data", append=True)
    assert ekd.written == [call("file", "out.grib", data="data", append=True)]


# _earthkit_data is the function imported above, not the fake the ekd fixture puts in standard_args


def test_earthkit_data():
    assert _earthkit_data() is pytest.importorskip("earthkit.data")


def test_missing_earthkit_data(monkeypatch):
    monkeypatch.setitem(sys.modules, "earthkit.data", None)
    with pytest.raises(click.ClickException, match="earthkit-data is required"):
        _earthkit_data()


def test_standard_args_is_not_a_command_module():
    assert "standard_args" not in command_modules()
