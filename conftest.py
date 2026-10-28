import importlib.util

import pytest


def _missing_modules(modules: tuple[str, ...]) -> list[str]:
    """Return the modules that are not importable."""
    return [module for module in modules if importlib.util.find_spec(module) is None]


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add CLI option to force optional-dependency tests to run."""
    parser.addoption(
        "--run-optional",
        action="store_true",
        default=False,
        help="Run tests even if optional dependencies are missing.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Auto-add dependency names as plain markers from requires(...)."""
    for item in items:
        marker = item.get_closest_marker("requires")
        if marker is None:
            continue

        if not marker.args:
            raise pytest.UsageError(
                '@pytest.mark.requires needs at least one module name, e.g. @pytest.mark.requires("torch")'
            )

        for arg in marker.args:
            item.add_marker(str(arg))


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip tests with missing optional dependencies unless forced to run."""
    marker = item.get_closest_marker("requires")
    if marker is None:
        return

    if item.config.getoption("--run-optional"):
        return

    modules = tuple(str(arg) for arg in marker.args)
    missing = _missing_modules(modules)

    if missing:
        pytest.skip("Missing optional dependenc" + ("y" if len(missing) == 1 else "ies") + f": {', '.join(missing)}")
