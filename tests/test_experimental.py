# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import warnings

import pytest

from earthkit.utils.decorators.experimental import ExperimentalWarning
from earthkit.utils.decorators.experimental import experimental

_DEFAULT_MESSAGE = "**Experimental API**: may change or be removed without notice."


def test_decorator_callable_bare():
    @experimental
    def add(a, b):
        return a + b

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        assert add(1, 2) == 3


def test_decorator_callable_parens():
    @experimental()
    def add(a, b):
        return a + b

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        assert add(1, 2) == 3


@pytest.mark.parametrize(
    "doc",
    [None, "My function."],
)
def test_docstring_injection(doc):
    def func():
        pass

    func.__doc__ = doc
    decorated = experimental(func)

    assert decorated.__doc__ is not None
    assert ".. warning::" in decorated.__doc__
    assert _DEFAULT_MESSAGE in decorated.__doc__
    if doc:
        assert doc in decorated.__doc__


def test_custom_message():
    @experimental(message="Beta feature.")
    def func():
        pass

    assert "Beta feature." in func.__doc__
    assert _DEFAULT_MESSAGE not in func.__doc__


def test_runtime_warning_emitted():
    @experimental
    def func():
        return 42

    with pytest.warns(ExperimentalWarning, match="func"):
        assert func() == 42


@pytest.mark.parametrize("env_value", ["", "0", "false", "False", "FALSE", "no", "off", " 0 "])
def test_env_var_silences_warning(monkeypatch, env_value):
    monkeypatch.setenv("EARTHKIT_EXPERIMENTAL_WARNINGS", env_value)

    @experimental
    def func():
        return 1

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        func()

    experimental_warnings = [w for w in record if issubclass(w.category, ExperimentalWarning)]
    assert len(experimental_warnings) == 0


def test_env_var_default_warns(monkeypatch):
    monkeypatch.delenv("EARTHKIT_EXPERIMENTAL_WARNINGS", raising=False)

    @experimental
    def func():
        return 1

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        func()

    experimental_warnings = [w for w in record if issubclass(w.category, ExperimentalWarning)]
    assert len(experimental_warnings) == 1


def test_warn_runtime_false_returns_original_bare():
    def original():
        """Original doc."""

    decorated = experimental(original, warn_runtime=False)

    assert decorated is original
    assert ".. warning::" in decorated.__doc__
    assert "Original doc." in decorated.__doc__

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        decorated()

    experimental_warnings = [w for w in record if issubclass(w.category, ExperimentalWarning)]
    assert len(experimental_warnings) == 0


def test_warn_runtime_false_returns_original_parens():
    @experimental(warn_runtime=False)
    def original():
        """Original doc."""

    assert ".. warning::" in original.__doc__
    assert "Original doc." in original.__doc__

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        original()

    experimental_warnings = [w for w in record if issubclass(w.category, ExperimentalWarning)]
    assert len(experimental_warnings) == 0


def test_warn_once_same_call_site(monkeypatch):
    monkeypatch.delenv("EARTHKIT_EXPERIMENTAL_WARNINGS", raising=False)

    @experimental
    def func():
        return 1

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("default", ExperimentalWarning)
        for _ in range(2):
            func()

    experimental_warnings = [w for w in record if issubclass(w.category, ExperimentalWarning)]
    assert len(experimental_warnings) == 1


def test_warn_twice_two_call_sites(monkeypatch):
    """Two call sites -> two warnings (line number is part of the key)."""
    monkeypatch.delenv("EARTHKIT_EXPERIMENTAL_WARNINGS", raising=False)

    @experimental
    def func():
        return 1

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("default", ExperimentalWarning)
        func()
        func()

    experimental_warnings = [w for w in record if issubclass(w.category, ExperimentalWarning)]
    assert len(experimental_warnings) == 2


def test_metadata_preserved():
    @experimental
    def my_function():
        """My docstring."""

    assert my_function.__name__ == "my_function"
    assert "my_function" in my_function.__qualname__
    assert my_function.__module__ == __name__


def test_args_kwargs_passthrough():
    @experimental
    def func(a, b, *, c=10):
        return a, b, c

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        assert func(1, 2, c=3) == (1, 2, 3)
        assert func(1, 2) == (1, 2, 10)
