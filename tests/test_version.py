import earthkit.utils


def test_version() -> None:
    assert earthkit.utils.__version__ != "999"
