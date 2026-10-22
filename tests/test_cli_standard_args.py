# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import json
import sys
import types

import click
import pytest
from click.testing import CliRunner

from earthkit.cli.main import command_modules
from earthkit.cli.standard_args import (
    SOURCE_HELP,
    TARGET_HELP,
    Target,
    add_options,
    index_option,
    profile_option,
    source_options,
    split_csv,
    target_options,
)


@pytest.fixture(autouse=True)
def earthkit_data(monkeypatch):
    """Stub earthkit.data, with from_source returning its arguments and to_target recording its calls."""
    module = types.ModuleType("earthkit.data")
    module.from_source = lambda *args, **kwargs: [list(args), kwargs]
    module.to_target = lambda *args, **kwargs: module.written.append((args, kwargs))
    module.written = []
    monkeypatch.setitem(sys.modules, "earthkit.data", module)
    return module


def _resolved_json(*data):
    return json.dumps([[list(d.args), d.kwargs] if isinstance(d, Target) else d for d in data])


@click.command()
@add_options([source_options(), target_options(), profile_option])
@click.option("--keys", multiple=True, callback=split_csv)
def _command(source, target, profile, keys):
    assert isinstance(target, Target)
    click.echo(json.dumps({"data": json.loads(_resolved_json(source, target)), "profile": profile, "keys": keys}))


@click.command()
@add_options([source_options("source_1"), source_options("source_2"), target_options()])
def _two_sources(source_1, source_2, target):
    click.echo(_resolved_json(source_1, source_2, target))


def _invoke(command, *args):
    return CliRunner().invoke(command, [str(a) for a in args])


def _resolved(*args, command=_command):
    result = _invoke(command, *args)
    assert result.exit_code == 0, result.output + repr(result.exception)
    resolved = json.loads(result.output)
    return resolved["data"] if command is _command else resolved


@pytest.fixture
def source_file(tmp_path):
    path = tmp_path / "in.grib"
    path.touch()
    return path


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
    result = _invoke(_command, "--help")
    assert result.exit_code == 0
    assert "[OPTIONS]\n" in result.output
    for option in ("--source [NAME:]VALUE", "--target [NAME:]VALUE", "--profile"):
        assert option in result.output
    for removed in ("--source-file", "--source-request", "--target-file", "--target-options"):
        assert removed not in result.output


@pytest.mark.parametrize(
    "args, expected",
    (
        ([], None),
        (["-p", "mars"], ["mars"]),
        (["--profile", "earthkit", "-p", "/my/profile.yaml"], ["earthkit", "/my/profile.yaml"]),
    ),
)
def test_profile_option(args, expected):
    @click.command()
    @profile_option
    def command(profile):
        click.echo(json.dumps(profile))

    result = _invoke(command, *args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == expected


@pytest.mark.parametrize(
    "value, expected",
    (
        (None, None),
        ("3", 3),
        ("-1", -1),
        ("4,5,8", [4, 5, 8]),
        ("1:7", slice(1, 7)),
        ("::2", slice(None, None, 2)),
        ("2:", slice(2, None)),
    ),
)
def test_index_option(value, expected):
    @click.command()
    @index_option
    def command(index):
        click.echo(repr(index))

    result = _invoke(command, *(["-i", value] if value is not None else []))
    assert result.exit_code == 0, result.output
    assert result.output.strip() == repr(expected)


@pytest.mark.parametrize("value", ("a", "1,a", "1:2:3:4", "1.5"))
def test_index_option_invalid(value):
    @click.command()
    @index_option
    def command(index): ...

    result = _invoke(command, "--index", value)
    assert result.exit_code == 2
    assert "Invalid value for '-i' / '--index'" in result.output


def test_standard_args_stdio(earthkit_data):
    @click.command()
    @add_options([source_options(positional=True), target_options(positional=True)])
    def command(source, target):
        assert source[0] == ["stream", sys.stdin.buffer]
        assert target.args == ("file", sys.stdout.buffer)
        target.to_target("data")

    result = CliRunner().invoke(command, ["-", "-"], input=b"GRIB")
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert earthkit_data.written[0][1] == {"data": "data"}


def test_standard_args_help_text():
    usage = " ".join(_invoke(_command, "--help").output.split())
    for text in (SOURCE_HELP, TARGET_HELP):
        assert " ".join(text.split()) in usage


def test_standard_args_files(source_file, tmp_path):
    target = tmp_path / "out.nc"
    result = _invoke(
        _command, "--source", f"file:{source_file}", "--target", f"file:{target}", "--profile", "mars", "--keys", "a,b"
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "data": [[["file", str(source_file)], {}], [["file", str(target)], {}]],
        "profile": ["mars"],
        "keys": ["a", "b"],
    }


def _resolved_source(value, tmp_path):
    return _resolved("--source", value, "--target", f"file:{tmp_path / 'out.nc'}")[0]


def _resolved_target(value, source_file):
    return _resolved("--source", f"file:{source_file}", "--target", value)[1]


def test_standard_args_source_glob(source_file, tmp_path):
    assert _resolved_source(f"file:{tmp_path / '*.grib'}", tmp_path) == [["file", str(tmp_path / "*.grib")], {}]


@pytest.mark.parametrize("prefix", ("", "file:"))
def test_standard_args_default_file(source_file, tmp_path, prefix):
    target = tmp_path / "out.nc"
    resolved = _resolved("--source", f"{prefix}{source_file}", "--target", f"{prefix}{target}")
    assert resolved == [[["file", str(source_file)], {}], [["file", str(target)], {}]]


@pytest.mark.parametrize("name", ("a:b.grib", "2020-01-01T00:00.grib", ":in.grib"))
def test_standard_args_default_file_with_colon(tmp_path, name):
    # Text before the first ':' that is not a valid name, e.g. '/tmp/.../a', belongs to the path
    path = tmp_path / name
    path.touch()
    assert _resolved_source(str(path), tmp_path) == [["file", str(path)], {}]


def test_standard_args_default_file_json_target(source_file, tmp_path):
    options = {"file": str(tmp_path / "out.grib"), "append": True}
    assert _resolved_target(json.dumps(options), source_file) == [["file"], options]


def test_standard_args_url_source(tmp_path):
    # Only the first ':' separates the name from the value
    resolved = _resolved_source("url:https://myhost.int/file.nc", tmp_path)
    assert resolved == [["url", "https://myhost.int/file.nc"], {}]


def test_standard_args_name_only(source_file, tmp_path):
    assert _resolved_source("dummy:", tmp_path) == [["dummy"], {}]
    assert _resolved_target("dummy:", source_file) == [["dummy"], {}]


def test_standard_args_request_with_dataset(tmp_path):
    request = {"variable": ["2m_temperature"], "year": "2020"}
    resolved = _resolved_source("cds:" + json.dumps({"dataset": "reanalysis-era5-single-levels", **request}), tmp_path)
    assert resolved == [["cds", "reanalysis-era5-single-levels"], {"request": request}]


def test_standard_args_request_without_dataset(tmp_path):
    request = {"param": "2t", "levtype": "sfc"}
    assert _resolved_source("mars:" + json.dumps(request), tmp_path) == [["mars"], {"request": request}]


def test_standard_args_request_list_with_dataset(tmp_path):
    requests = [{"dataset": "d", "year": "2020"}, {"dataset": "d", "year": "2021"}]
    resolved = _resolved_source("cds:" + json.dumps(requests), tmp_path)
    assert resolved == [["cds", "d"], {"request": [{"year": "2020"}, {"year": "2021"}]}]


def test_standard_args_file_target_with_options(source_file, tmp_path):
    options = {"file": str(tmp_path / "out.grib"), "append": True}
    assert _resolved_target("file:" + json.dumps(options), source_file) == [["file"], options]


def test_standard_args_other_target(source_file, tmp_path):
    options = {"xarray_to_zarr_kwargs": {"store": str(tmp_path / "out.zarr")}}
    assert _resolved_target("zarr:" + json.dumps(options), source_file) == [["zarr"], options]


def test_standard_args_two_sources(source_file, tmp_path):
    usage = _invoke(_two_sources, "--help").output
    for option in ("--source-1 [NAME:]VALUE", "--source-2 [NAME:]VALUE", "--target [NAME:]VALUE"):
        assert option in usage

    request = {"dataset": "reanalysis-era5-single-levels", "year": "2020"}
    resolved = _resolved(
        "--source-1",
        f"file:{source_file}",
        "--source-2",
        "cds:" + json.dumps(request),
        "--target",
        f"file:{tmp_path / 'out.nc'}",
        command=_two_sources,
    )
    assert resolved == [
        [["file", str(source_file)], {}],
        [["cds", "reanalysis-era5-single-levels"], {"request": {"year": "2020"}}],
        [["file", str(tmp_path / "out.nc")], {}],
    ]


@pytest.mark.parametrize(
    "args, message",
    (
        (["--target", "file:{tmp}/out.nc"], "Missing option '--source'"),
        (["--source", "file:{src}"], "Missing option '--target'"),
        (["--source", "{tmp}/missing.grib", "--target", "{tmp}/o.nc"], "does not exist"),
        (["--source", "{src}", "--target", "{tmp}"], "is a directory"),
        (["--source", "file:{tmp}/missing.grib", "--target", "file:{tmp}/o.nc"], "does not exist"),
        (["--source", "file:{src}", "--target", "file:{tmp}"], "is a directory"),
        (["--source", "file:{src}", "--target", 'file:{"file": "{tmp}"}'], "is a directory"),
        (["--source-file", "{src}", "--target", "file:{tmp}/o.nc"], "No such option"),
        (["--source", "file:{src}", "--target-file", "{tmp}/o.nc"], "No such option"),
        (["--source", 'mars:{"param": ', "--target", "file:{tmp}/o.nc"], "Invalid JSON"),
        (["--source", "mars:[1, 2]", "--target", "file:{tmp}/o.nc"], "Expected a JSON object or a list of objects"),
        (["--source", "file:{src}", "--target", 'zarr:[{"a": 1}]'], "Expected a JSON object"),
        (["--source", 'cds:[{"dataset": "a"}, {"dataset": "b"}]', "--target", "file:{tmp}/o.nc"], "same 'dataset'"),
        (["--source", 'cds:[{"dataset": "a"}, {"year": "2020"}]', "--target", "file:{tmp}/o.nc"], "same 'dataset'"),
    ),
)
def test_standard_args_invalid(source_file, tmp_path, args, message):
    args = [a.replace("{tmp}", str(tmp_path)).replace("{src}", str(source_file)) for a in args]
    result = _invoke(_command, *args)
    assert result.exit_code == 2, result.output
    assert message in result.output


@click.command()
@add_options([source_options("source_1", positional=True), source_options("source_2"), target_options(positional=True)])
def _positional(source_1, source_2, target):
    click.echo(_resolved_json(source_1, source_2, target))


def test_standard_args_positional(source_file, tmp_path):
    usage = _invoke(_positional, "--help").output
    assert "[OPTIONS] SOURCE_1 TARGET\n" in usage
    assert "--source-2 [NAME:]VALUE" in usage
    assert "--source-1" not in usage and "--target " not in usage

    request = {"dataset": "d", "year": "2020"}
    resolved = _resolved(
        source_file,
        "zarr:{}",
        "--source-2",
        "cds:" + json.dumps(request),
        command=_positional,
    )
    assert resolved == [
        [["file", str(source_file)], {}],
        [["cds", "d"], {"request": {"year": "2020"}}],
        [["zarr"], {}],
    ]


def test_standard_args_positional_merges_sources(source_file):
    resolved = _resolved(source_file, source_file, "out.nc", "--source-2", source_file, command=_positional)
    assert resolved[0] == [["multi", [["file", str(source_file)], {}], [["file", str(source_file)], {}]], {}]
    assert resolved[2] == [["file", "out.nc"], {}]


@pytest.mark.parametrize(
    "args, message",
    (
        (["--source-2", "{src}"], "Missing argument 'SOURCE_1'"),
        (["{src}", "--source-2", "{src}"], "Missing argument 'SOURCE_1'"),
        (["{tmp}/missing.grib", "{tmp}/o.nc", "--source-2", "{src}"], "Invalid value for 'SOURCE_1'"),
    ),
)
def test_standard_args_positional_invalid(source_file, tmp_path, args, message):
    args = [a.replace("{tmp}", str(tmp_path)).replace("{src}", str(source_file)) for a in args]
    result = _invoke(_positional, *args)
    assert result.exit_code == 2, result.output
    assert message in result.output


def test_standard_args_merges_sources(source_file):
    resolved = _resolved("--source", source_file, "--source", "url:https://myhost.int/file.nc", "--target", "o.nc")
    assert resolved[0] == [
        ["multi", [["file", str(source_file)], {}], [["url", "https://myhost.int/file.nc"], {}]],
        {},
    ]


def test_target_calls_earthkit_data(earthkit_data):
    Target(("file", "out.grib"), {"append": False}).to_target("data", append=True)
    assert earthkit_data.written == [(("file", "out.grib"), {"data": "data", "append": True})]


def test_missing_earthkit_data(monkeypatch, source_file):
    monkeypatch.setitem(sys.modules, "earthkit.data", None)
    result = _invoke(_command, "--source", source_file, "--target", "o.nc")
    assert result.exit_code == 1
    assert "earthkit-data is required" in result.output


@pytest.mark.parametrize("name", ("", "source-1", "1source"))
@pytest.mark.parametrize("positional", (False, True))
def test_source_options_invalid_name(name, positional):
    with pytest.raises(ValueError):
        source_options(name, positional=positional)


def test_standard_args_is_not_a_command_module():
    assert "standard_args" not in command_modules()
