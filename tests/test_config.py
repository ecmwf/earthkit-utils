# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Behaviour of the user-facing config object.

The test bodies are taken from the existing earthkit-data and earthkit-geo test suites
(``tests/core/test_config.py`` and ``tests/utils/test_config.py``). They only use the
public config API as documented for those packages: ``get``, ``set``, ``reset``,
``temporary``, ``save_as``, ``autosave`` and the environment variables.

``ConfigTests`` holds the tests and is not collected itself. Each subclass selects the
implementation under test by providing the ``config`` fixture and the ``env_prefix`` of its
environment variables:

- ``TestUtilsConfig``: ``earthkit.utils.config``, the implementation to be shared. The only
  part of the file that depends on how a config is created. It also has the tests for what is
  specific to the shared implementation (creating configs, several configs in one process,
  custom validators), which the existing implementations cannot run.
- ``TestDataConfig`` and ``TestGeoConfig``: the global config object of earthkit-data and
  earthkit-geo. They show that the tests describe the behaviour of the existing
  implementations. Remove them when data and geo use ``earthkit.utils.config``.

A subclass is skipped when its implementation is not available.
"""

import os
import stat

import pytest
import yaml


def read_config_yaml(path):
    try:
        with open(path) as f:
            s = yaml.load(f, Loader=yaml.SafeLoader)
            if not isinstance(s, dict):
                s = {}
            return s
    except Exception:
        return {}


class ConfigTests:
    env_prefix = None

    @pytest.fixture
    def config(self, tmp_path):
        raise NotImplementedError

    @pytest.fixture(autouse=True)
    def _restore_autosave(self, config):
        # make sure a failing test cannot leak the autosave setting
        autosave = config.autosave
        yield
        config.autosave = autosave

    @pytest.mark.parametrize(
        "param,default_value,new_value",
        [
            ("url-download-timeout", 30, 5),
        ],
    )
    def test_configs_params_set_reset(self, config, param, default_value, new_value):
        ori_value = config.get(param)

        with config.temporary():
            config.reset()

            assert config.get(param) == default_value
            config.set(param, new_value)
            assert config.get(param) == new_value
            config.reset()
            assert config.get(param) == default_value

        assert config.get(param) == ori_value

    def test_config_invalid(self, config):
        # invalid param
        with pytest.raises(KeyError):
            config.get("_invalid_")

        with pytest.raises(KeyError):
            config.set("_invalid_", 1)

        # invalid value
        with pytest.raises(ValueError):
            config.set("url-download-timeout", "A")

    @pytest.mark.parametrize(
        "param,set_value,stored_value,raise_error",
        [
            ("check-out-of-date-urls", True, True, None),
            ("check-out-of-date-urls", 1, True, None),
            ("check-out-of-date-urls", 1.0, True, None),
            ("check-out-of-date-urls", False, False, None),
            ("check-out-of-date-urls", 0, False, None),
            ("check-out-of-date-urls", 0.0, False, None),
            ("check-out-of-date-urls", "true", True, None),
            ("check-out-of-date-urls", "false", True, None),
            ("check-out-of-date-urls", "1", True, None),
            ("check-out-of-date-urls", "0", True, None),
        ],
    )
    def test_config_set_bool(self, config, param, set_value, stored_value, raise_error):
        with config.temporary():
            if raise_error is None:
                config.set(param, set_value)
                assert config.get(param) == stored_value
            else:
                with pytest.raises(raise_error):
                    config.set(param, set_value)

    @pytest.mark.parametrize(
        "param,set_value,stored_value,raise_error",
        [
            ("url-download-timeout", "A", "A", ValueError),
            ("url-download-timeout", 30, 30, None),
            ("url-download-timeout", "30", 30, None),
            ("url-download-timeout", "2m", 120, None),
            ("url-download-timeout", "10h", 36000, None),
            ("url-download-timeout", "7d", 7 * 24 * 3600, None),
            ("url-download-timeout", "1x", None, ValueError),
            ("url-download-timeout", "1M", 60, ValueError),
        ],
    )
    def test_config_set_numbers(self, config, param, set_value, stored_value, raise_error):
        with config.temporary():
            if raise_error is None:
                config.set(param, set_value)
                assert config.get(param) == stored_value
            else:
                with pytest.raises(raise_error):
                    config.set(param, set_value)

    def test_config_set_cache_numbers(self, config):
        with config.temporary():
            data = [
                ("maximum-cache-size", "1", 1, None),
                ("maximum-cache-size", "1k", 1024, None),
                ("maximum-cache-size", "1kb", 1024, None),
                ("maximum-cache-size", "1k", 1024, None),
                ("maximum-cache-size", "1kb", 1024, None),
                ("maximum-cache-size", "1K", 1024, None),
                ("maximum-cache-size", "1M", 1024 * 1024, None),
                ("maximum-cache-size", "1G", 1024 * 1024 * 1024, None),
                ("maximum-cache-size", "1T", 1024 * 1024 * 1024 * 1024, None),
                ("maximum-cache-size", "1P", 1024 * 1024 * 1024 * 1024 * 1024, None),
                ("maximum-cache-size", None, None, None),
                ("maximum-cache-size", "-1", None, ValueError),
                ("maximum-cache-disk-usage", "2%", 2, None),
                ("maximum-cache-disk-usage", None, None, None),
                ("maximum-cache-disk-usage", "-2%", None, ValueError),
                ("maximum-cache-disk-usage", "102%", 102, None),
                ("maximum-cache-disk-usage", "0%", 0, None),
            ]

            for param, set_value, stored_value, raise_error in data:
                if raise_error is None:
                    config.set(param, set_value)
                    if stored_value is not None:
                        assert config.get(param) == stored_value
                    else:
                        assert config.get(param) is None
                else:
                    with pytest.raises(raise_error):
                        config.set(param, set_value)

    def test_config_set_multi(self, config):
        with config.temporary():
            config.set("url-download-timeout", 7)
            assert config.get("url-download-timeout") == 7

            config.set({"url-download-timeout": 2, "check-out-of-date-urls": False})
            assert config.get("url-download-timeout") == 2
            assert not config.get("check-out-of-date-urls")

            config.set(url_download_timeout=11, check_out_of_date_urls=False)
            assert config.get("url-download-timeout") == 11
            assert not config.get("check-out-of-date-urls")

            with pytest.raises(KeyError):
                config.set({"url-download-timeout": 2, "-invalid-": 21})

            with pytest.raises(KeyError):
                config.set(url_download_timeout=3, __invalid__=11)

    def test_config_temporary_single(self, config):
        with config.temporary("url-download-timeout", 7):
            assert config.get("url-download-timeout") == 7

        with config.temporary({"url-download-timeout": 7}):
            assert config.get("url-download-timeout") == 7

        with config.temporary(url_download_timeout=7):
            assert config.get("url-download-timeout") == 7

    def test_config_temporary_multi(self, config):
        with config.temporary({"url-download-timeout": 2, "check-out-of-date-urls": False}):
            assert config.get("url-download-timeout") == 2
            assert not config.get("check-out-of-date-urls")

        with config.temporary(url_download_timeout=3, check_out_of_date_urls=False):
            assert config.get("url-download-timeout") == 3
            assert not config.get("check-out-of-date-urls")

    def test_config_temporary_nested(self, config):
        with config.temporary("url-download-timeout", 7):
            assert config.get("url-download-timeout") == 7
            with config.temporary("url-download-timeout", 10):
                assert config.get("url-download-timeout") == 10
            assert config.get("url-download-timeout") == 7

    def test_config_temporary_autosave_1(self, config, tmp_path):
        config_file = str(tmp_path / "temporary.yaml")

        with config.temporary(config_yaml=config_file):
            # now config should contain the default values
            # we ensure that the configs are saved into the file
            config.save_as(config_file)

            key = "url-download-timeout"

            config.autosave = False

            # when a key has a default value, it is not saved into the config file
            s = read_config_yaml(config_file)
            assert key not in s

            v = config.get(key)
            config.set(key, v + 10)
            assert config.get(key) == v + 10

            # the config file should be the same
            s = read_config_yaml(config_file)
            assert key not in s

    def test_config_temporary_autosave_2(self, config, tmp_path):
        config_file = str(tmp_path / "temporary.yaml")

        with config.temporary(config_yaml=config_file):
            # now config should contain the default values
            # we ensure that the config is saved into the file
            config.save_as(config_file)

            key = "url-download-timeout"

            config.autosave = True

            # when a key has a default value, it is not saved into the config file
            s = read_config_yaml(config_file)
            assert key not in s

            v = config.get(key)
            v_new = v + 10
            config.set(key, v_new)
            assert config.get(key) == v_new

            # the file changed
            s = read_config_yaml(config_file)
            assert s[key] == v_new

            config.autosave = False
            config.set(key, v)
            assert config.get(key) == v
            s = read_config_yaml(config_file)
            assert s[key] == v_new

    @pytest.mark.parametrize("value,error", [("10000", None), (10000, None), ("1b", ValueError), ("A", ValueError)])
    def test_config_env(self, config, monkeypatch, value, error):
        monkeypatch.setenv(self.env_prefix + "URL_DOWNLOAD_TIMEOUT", str(value))

        if error is None:
            v = config.get("url-download-timeout")
            assert v == 10000
        else:
            with pytest.raises(error):
                config.get("url-download-timeout")

    # ------------------------------------------------------------------
    # Additional tests for behaviour that is documented or relied upon by
    # the packages but not covered by the original test suites.
    # ------------------------------------------------------------------

    @pytest.fixture
    def changes(self, config):
        calls = []

        def callback():
            calls.append(1)

        config.on_change(callback)
        yield calls
        # there is no public way to unregister a callback
        config._callbacks.remove(callback)

    def test_config_reset_single(self, config):
        with config.temporary():
            config.set({"url-download-timeout": 5, "check-out-of-date-urls": False})

            config.reset("url-download-timeout")
            assert config.get("url-download-timeout") == 30
            assert not config.get("check-out-of-date-urls")

            with pytest.raises(KeyError):
                config.reset("_invalid_")

    @pytest.mark.parametrize("autosave", [True, False])
    def test_config_env_warning(self, config, monkeypatch, autosave):
        env_name = self.env_prefix + "URL_DOWNLOAD_TIMEOUT"
        monkeypatch.setenv(env_name, "5")
        config.autosave = autosave

        msg = (
            f"Config option 'url-download-timeout' is also set by environment variable '{env_name}'."
            "The environment variable takes precedence and its value is returned when calling get()."
        )

        with config.temporary():
            with pytest.warns(UserWarning) as record:
                config.set("url-download-timeout", 10)
            assert [str(w.message) for w in record] == [
                msg + (" Still, the value set here will be saved to the config file." if autosave else "")
            ]

            with pytest.warns(UserWarning) as record:
                config.reset("url-download-timeout")
            assert [str(w.message) for w in record] == [
                msg + (" Still, the value reset here will be saved to the config file." if autosave else "")
            ]

    def test_config_env_precedence(self, config, monkeypatch):
        # the example in the documentation
        env_name = self.env_prefix + "URL_DOWNLOAD_TIMEOUT"

        with config.temporary():
            config.reset()
            assert config.get("url-download-timeout") == 30
            assert "url-download-timeout" not in config.env()

            monkeypatch.setenv(env_name, "5")
            assert config.get("url-download-timeout") == 5
            assert config.env()["url-download-timeout"] == (env_name, "5")

            with pytest.warns(UserWarning):
                config.set("url-download-timeout", 10)
            assert config.get("url-download-timeout") == 5

            monkeypatch.delenv(env_name)
            assert config.get("url-download-timeout") == 10
            assert "url-download-timeout" not in config.env()

    def test_config_repr(self, config, monkeypatch):
        env_name = self.env_prefix + "URL_DOWNLOAD_TIMEOUT"

        with config.temporary():
            config.reset()
            assert "url-download-timeout: (30s, 30s)" in repr(config).splitlines()

            config.set("url-download-timeout", "1m")
            assert "url-download-timeout: (1m, 30s)" in repr(config).splitlines()

            monkeypatch.setenv(env_name, "5")
            assert f"url-download-timeout: ({env_name}=5, 1m, 30s)" in repr(config).splitlines()

    def test_config_repr_html(self, config, monkeypatch):
        env_name = self.env_prefix + "URL_DOWNLOAD_TIMEOUT"

        with config.temporary():
            config.reset()
            html = config._repr_html_()
            assert html.startswith("<style")
            assert "<table class='ek'>" in html
            assert "<tr><th>Name</th><th>Value</th><th>Default</th></tr>" in html
            assert "<tr><td>url-download-timeout</td><td>'30s'</td><td>'30s'</td></tr>" in html

            monkeypatch.setenv(env_name, "5")
            assert (
                f"<tr><td>url-download-timeout</td><td>{env_name}='5'<br>('30s')</td><td>'30s'</td></tr>"
                in config._repr_html_()
            )

    def test_config_on_change(self, config, changes):
        def count():
            n = len(changes)
            changes.clear()
            return n

        with config.temporary():
            # entering the context
            assert count() == 1

            config.set("url-download-timeout", 7)
            assert count() == 1

            config.set({"url-download-timeout": 8})
            assert count() == 1

            # NOTE: the callbacks are called twice when using keyword arguments
            config.set(url_download_timeout=9)
            assert count() == 2

            config.reset("url-download-timeout")
            assert count() == 1

            config.reset()
            assert count() == 1

            # invalid changes are not notified
            with pytest.raises(KeyError):
                config.set("_invalid_", 1)
            with pytest.raises(ValueError):
                config.set("url-download-timeout", "A")
            assert count() == 0

        # leaving the context
        assert count() == 1

        # setting the options when creating the temporary config does not notify
        with config.temporary("url-download-timeout", 7):
            assert count() == 1
        assert count() == 1

    def test_config_save_as_round_trip(self, config, tmp_path):
        path = tmp_path / "saved.yaml"
        names = ["url-download-timeout", "check-out-of-date-urls", "maximum-cache-size", "maximum-cache-disk-usage"]

        with config.temporary():
            config.reset()
            config.set({"url-download-timeout": "1m", "check-out-of-date-urls": False, "maximum-cache-size": "2G"})
            expected = {k: config.get(k) for k in names}
            config.save_as(str(path))

        saved = read_config_yaml(path)
        # the version is stored in the file but it is not a config option
        saved.pop("version")

        with config.temporary():
            config.reset()
            config.set(saved)
            assert {k: config.get(k) for k in names} == expected


class TestUtilsConfig(ConfigTests):
    env_prefix = "EARTHKIT_TEST_"

    @pytest.fixture
    def module(self):
        return pytest.importorskip("earthkit.utils.config")

    @staticmethod
    def make_options(module):
        _ = module.ConfigOption

        # Same option definitions as the packages use (a subset of those defined by both
        # earthkit-data and earthkit-geo).
        return {
            "url-download-timeout": _(
                "30s",
                "Timeout when downloading from an url.",
                getter="_as_seconds",
            ),
            "check-out-of-date-urls": _(
                True,
                "Perform a HTTP request to check if the remote version of a cache file has changed",
            ),
            "maximum-cache-size": _(
                None,
                "Maximum disk space used by the cache (e.g.: 100G or 2T).",
                getter="_as_bytes",
                none_ok=True,
            ),
            "maximum-cache-disk-usage": _(
                None,
                "Maximum disk usage as a percentage of the full disk capacity (e.g.: 90%).",
                getter="_as_percent",
                none_ok=True,
            ),
        }

    @pytest.fixture
    def config(self, module, tmp_path):
        # NOTE: tentative signature of the creation API
        return module.Config.create(
            "test",
            self.make_options(module),
            version="1.0.0",
            config_file=str(tmp_path / "config.yaml"),
            env_prefix=self.env_prefix,
        )

    # ------------------------------------------------------------------
    # Tests for the parts that are specific to the shared implementation:
    # creating configs, and several configs living in one process. The
    # existing implementations cannot run these, since they only provide
    # a single global config object.
    # ------------------------------------------------------------------

    def test_create_writes_default_file(self, module, tmp_path):
        path = tmp_path / "new" / "config.yaml"
        module.Config.create(
            "createtest", self.make_options(module), version="1.0.0", config_file=str(path), env_prefix="EARTHKIT_X_"
        )

        # missing directories are created
        assert path.is_file()
        assert "url-download-timeout" not in read_config_yaml(path)

    def test_create_loads_file(self, module, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("url-download-timeout: 1m\nversion: 1.0.0\n")

        config = module.Config.create(
            "createtest", self.make_options(module), version="1.0.0", config_file=str(path), env_prefix="EARTHKIT_X_"
        )
        assert config.get("url-download-timeout") == 60

    def test_create_config_file_from_env(self, module, tmp_path, monkeypatch):
        path = tmp_path / "from-env" / "config.yaml"
        monkeypatch.setenv("EARTHKIT_CREATETEST_CONFIG_FILE", str(path))

        config = module.Config.create("createtest", self.make_options(module), version="1.0.0")
        assert path.is_file()

        config.set("url-download-timeout", 7)
        assert read_config_yaml(path)["url-download-timeout"] == 7

    def test_create_default_location(self, module, tmp_path, monkeypatch):
        monkeypatch.delenv("EARTHKIT_CREATETEST_CONFIG_FILE", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))

        module.Config.create("createtest", self.make_options(module), version="1.0.0")

        path = tmp_path / ".config" / "earthkit" / "createtest" / "config.yaml"
        assert path.is_file()
        if os.name == "posix":
            assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700

    def test_create_env_prefix_from_name(self, module, tmp_path, monkeypatch):
        env_name = "EARTHKIT_CREATETEST_URL_DOWNLOAD_TIMEOUT"
        monkeypatch.setenv(env_name, "5")

        config = module.Config.create(
            "createtest", self.make_options(module), version="1.0.0", config_file=str(tmp_path / "config.yaml")
        )
        assert config.get("url-download-timeout") == 5
        assert config.env()["url-download-timeout"] == (env_name, "5")

    def test_create_explicit_overrides(self, module, tmp_path, monkeypatch):
        monkeypatch.setenv("EARTHKIT_CREATETEST_CONFIG_FILE", str(tmp_path / "from-env.yaml"))
        monkeypatch.setenv("EARTHKIT_CREATETEST_URL_DOWNLOAD_TIMEOUT", "5")
        path = tmp_path / "explicit.yaml"

        config = module.Config.create(
            "createtest",
            self.make_options(module),
            version="1.0.0",
            config_file=str(path),
            env_prefix="EARTHKIT_OTHER_",
        )

        # neither the file nor the environment variable derived from the name is used
        config.set("url-download-timeout", 7)
        assert config.get("url-download-timeout") == 7
        assert read_config_yaml(path)["url-download-timeout"] == 7
        assert not (tmp_path / "from-env.yaml").exists()

    def test_config_autosave_file(self, config, tmp_path):
        path = tmp_path / "config.yaml"
        assert path.is_file()

        config.set("url-download-timeout", 7)
        assert read_config_yaml(path)["url-download-timeout"] == 7

        # changes in a temporary config are not saved into the file of the config
        with config.temporary():
            config.set("url-download-timeout", 9)
        assert read_config_yaml(path)["url-download-timeout"] == 7

        config.autosave = False
        config.set("url-download-timeout", 11)
        assert read_config_yaml(path)["url-download-timeout"] == 7

    def test_config_instances_independent(self, module, config, tmp_path):
        other = module.Config.create(
            "other",
            self.make_options(module),
            version="1.0.0",
            config_file=str(tmp_path / "other.yaml"),
            env_prefix="EARTHKIT_OTHER_",
        )
        calls = []
        other.on_change(lambda: calls.append(1))

        assert config.autosave
        assert other.autosave

        # autosave is not shared between configs
        config.autosave = False
        assert other.autosave

        # nor are the values and the callbacks
        config.set("url-download-timeout", 7)
        assert other.get("url-download-timeout") == 30
        assert calls == []

        other.set("url-download-timeout", 9)
        assert config.get("url-download-timeout") == 7
        assert len(calls) == 1

    def test_config_autosave_shared_with_temporary(self, config):
        config.autosave = False
        with config.temporary():
            assert not config.autosave
            config.autosave = True
        assert config.autosave

    def test_config_custom_validator(self, module, tmp_path):
        class Even(module.Validator):
            def check(self, value):
                return value % 2 == 0

            def explain(self):
                return "Valid when even."

        options = {"even": module.ConfigOption(2, "An even number. {validator}", validator=Even())}
        config = module.Config.create(
            "validatortest", options, version="1.0.0", config_file=str(tmp_path / "config.yaml")
        )

        with config.temporary():
            config.set("even", 4)
            assert config.get("even") == 4

            with pytest.raises(ValueError, match=r"Config option 'even' cannot be set to 3\. Valid when even\."):
                config.set("even", 3)
            assert config.get("even") == 4

        assert options["even"].docs_description == "An even number. Valid when even."

    def test_config_list_validator(self, module, tmp_path):
        options = {
            "policy": module.ConfigOption("off", "Policy. {validator}", validator=module.ListValidator(["off", "user"]))
        }
        config = module.Config.create(
            "validatortest", options, version="1.0.0", config_file=str(tmp_path / "config.yaml")
        )

        with config.temporary():
            config.set("policy", "user")
            assert config.get("policy") == "user"

            with pytest.raises(ValueError, match="Config option 'policy' cannot be set to bad"):
                config.set("policy", "bad")
            assert config.get("policy") == "user"


class TestDataConfig(ConfigTests):
    env_prefix = "EARTHKIT_DATA_"

    @pytest.fixture
    def config(self, tmp_path):
        # the global config of the package, which uses the package's own options
        return pytest.importorskip("earthkit.data").config


class TestGeoConfig(ConfigTests):
    env_prefix = "EARTHKIT_GEO_"

    @pytest.fixture
    def config(self, tmp_path):
        # the global config of the package, which uses the package's own options
        return pytest.importorskip("earthkit.geo").config
