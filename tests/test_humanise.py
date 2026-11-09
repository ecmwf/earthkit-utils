#!/usr/bin/env python3

# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

# The tests run against ``earthkit.utils.humanise``. During the move they were
# validated against the original implementations in earthkit-data and earthkit-geo.
humanise = pytest.importorskip("earthkit.utils.humanise")


@pytest.mark.parametrize(
    "value,expected",
    [
        ("30s", 30),
        ("1m", 60),
        (" 5 m ", 300),
        ("2h", 7200),
        ("1d", 86400),
        ("1w", 604800),
        ("10", 10),
        (5, 5),
        # only the first letter of the unit counts
        ("5minutes", 300),
    ],
)
def test_as_seconds(value, expected):
    assert humanise.as_seconds(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1K", 1024),
        ("1k", 1024),
        ("2M", 2 * 1024**2),
        ("1g", 1024**3),
        ("1T", 1024**4),
        ("1P", 1024**5),
        ("100G", 100 * 1024**3),
        ("3", 3),
        (1024, 1024),
        # only the first letter of the unit counts
        ("1KB", 1024),
        ("1KiB", 1024),
    ],
)
def test_as_bytes(value, expected):
    assert humanise.as_bytes(value) == expected


@pytest.mark.parametrize("value,expected", [("90%", 90), ("90", 90), ("5 %", 5), ("0%", 0)])
def test_as_percent(value, expected):
    assert humanise.as_percent(value) == expected


@pytest.mark.parametrize("func", ["as_seconds", "as_bytes", "as_percent"])
def test_as_number_none_ok(func):
    assert getattr(humanise, func)(None, none_ok=True) is None


@pytest.mark.parametrize(
    "func,value,message",
    [
        ("as_seconds", "1.5s", "invalid number/unit 1.5s"),
        ("as_seconds", "abc", "invalid number/unit abc"),
        ("as_seconds", "", "invalid number/unit "),
        ("as_seconds", "5 s s", "invalid number/unit 5 s s"),
        ("as_seconds", None, "invalid number/unit None"),
        ("as_bytes", "x", "invalid number/unit x"),
        ("as_percent", None, "invalid number/unit None"),
        ("as_seconds", "5x", "invalid unit 'x', valid values are s, m, h, d, w"),
        ("as_bytes", "1E", "invalid unit 'E', valid values are K, k, M, m, G, g, T, t, P, p"),
        ("as_percent", "90m", "invalid unit 'm', valid values are %"),
    ],
)
def test_as_number_invalid(func, value, message):
    # without a name the message is prefixed with "None"
    with pytest.raises(ValueError) as exc:
        getattr(humanise, func)(value)
    assert str(exc.value) == f"None: {message}"

    with pytest.raises(ValueError) as exc:
        getattr(humanise, func)(value, name="opt")
    assert str(exc.value) == f"opt: {message}"


def test_as_number_custom_units():
    assert humanise.as_number("3k", "n", {"k": 1000}, False) == 3000


@pytest.mark.parametrize(
    "items,kwargs,expected",
    [
        ([], {}, "??"),
        (["a"], {}, "a"),
        (["a", "b"], {}, "a and b"),
        (["a", "b", "c"], {}, "a, b and c"),
        (["a", "b", "c", "d"], {}, "a, b, c and d"),
        (["a", "b"], {"conjunction": "or"}, "a or b"),
        (["a", "b", "c"], {"conjunction": "or"}, "a, b or c"),
    ],
)
def test_list_to_human(items, kwargs, expected):
    assert humanise.list_to_human(items, **kwargs) == expected


@pytest.mark.parametrize("value,expected", [(5, "5"), ("a", "a"), (None, "None")])
def test_as_str(value, expected):
    assert humanise.as_str(value) == expected


def test_as_str_none_ok():
    assert humanise.as_str(None, none_ok=True) is None


@pytest.mark.parametrize("value,expected", [("5", 5), (5, 5), (5.0, 5), (" 5 ", 5), ('"5"', 5), ("'5'", 5)])
def test_as_int(value, expected):
    assert humanise.as_int(value) == expected


def test_as_int_none_ok():
    assert humanise.as_int(None, none_ok=True) is None


@pytest.mark.parametrize("value", ["A", "5.5", ""])
def test_as_int_invalid(value):
    with pytest.raises(ValueError):
        humanise.as_int(value)


def test_as_int_none():
    with pytest.raises(TypeError):
        humanise.as_int(None)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("a,b", ["a", "b"]),
        (' "a,b" ', ["a", "b"]),
        ("a", ["a"]),
        ("", [""]),
        (("a", "b"), ["a", "b"]),
        (["a", "b"], ["a", "b"]),
    ],
)
def test_as_list(value, expected):
    assert humanise.as_list(value) == expected


def test_as_list_none_ok():
    assert humanise.as_list(None, none_ok=True) == []


def test_as_list_none():
    with pytest.raises(TypeError):
        humanise.as_list(None)
