# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Configuration shared by the earthkit packages.

A package describes its options with :class:`ConfigOption` objects and creates its
config with :meth:`Config.create`::

    config = Config.create("data", {"cache-policy": ConfigOption("off", "...")}, version=__version__)

The values come from (highest precedence first) the ``EARTHKIT_<NAME>_<OPTION>`` environment
variables, the values set in the session and the config file, which by default is
``~/.config/earthkit/<name>/config.yaml``.

``yaml`` and ``filelock`` are only imported when a config is created or saved.
"""

import contextlib
import logging
import os
import warnings
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, ContextManager, Iterator

from earthkit.utils.html import css
from earthkit.utils.humanise import list_to_human

LOG = logging.getLogger(__name__)


class Validator(metaclass=ABCMeta):
    """Check the values of a config option."""

    @abstractmethod
    def check(self, value):
        pass

    @abstractmethod
    def explain(self):
        pass


class ListValidator(Validator):
    """Check if a value is in a list of valid values."""

    def __init__(self, values):
        self.values = values

    def check(self, value):
        return value in self.values

    def explain(self):
        return f"Valid values: {list_to_human(self.values)}."


class ConfigOption:
    """Definition of a config option.

    Parameters
    ----------
    default
        The default value.
    description: str
        Description of the option. ``{validator}`` is replaced by the explanation of the validator.
    getter: callable, optional
        Function ``getter(value, name=None, none_ok=False)`` that converts a value, e.g. ``"30s"``,
        to the one returned by ``Config.get()``. It is also used to check the value when it is set,
        so it must raise ``ValueError`` for invalid values. See :mod:`earthkit.utils.humanise`.
    none_ok: bool
        Whether the value can be ``None``. It is passed to the getter.
    kind: type, optional
        Type of the value. Defaults to the type of ``default``.
    docs_default
        Default value shown in the documentation. Defaults to ``default``.
    validator: Validator, optional
        Checks the value when it is set.
    """

    def __init__(
        self,
        default: Any,
        description: str,
        getter: Callable[..., Any] | None = None,
        none_ok: bool = False,
        kind: type | None = None,
        docs_default: Any = None,
        validator: Validator | None = None,
    ):
        if getter is not None and not callable(getter):
            raise TypeError(f"The getter of a config option must be a function, not {getter!r}")

        self.default = default
        self.description = description
        self.getter = getter
        self.none_ok = none_ok
        self.kind = kind if kind is not None else type(default)
        self.docs_default = docs_default if docs_default is not None else self.default
        self.validator = validator

    def save(self, name, value, f):
        import yaml

        for n in self.docs_description.split("\n"):
            print(f"# {n.strip()}", file=f)
        print(file=f)
        comment = yaml.dump({name: self.default}, default_flow_style=False)
        for n in comment.split("\n"):
            if n:
                print(f"# {n}", file=f)
        if value != self.default:
            print(file=f)
            yaml.dump({name: value}, f, default_flow_style=False)

    @property
    def docs_description(self):
        d = self.description
        t = ""
        if self.validator:
            t = self.validator.explain()

        return d.replace("{validator}", t)

    def validate(self, name, value):
        if self.validator is not None and not self.validator.check(value):
            raise ValueError(f"Config option '{name}' cannot be set to {value}. {self.validator.explain()}")


class _Layer:
    """The values of a config, and the file they are saved to."""

    def __init__(self, values, file):
        self.values = values
        self.file = file


def _default_config_file(name, version):
    # NOTE: the file does not depend on the version yet. The version is an argument so that
    # the location can become version specific without changing the callers.
    return os.path.join(os.path.expanduser("~/.config/earthkit"), name, "config.yaml")


def _default_env_prefix(name):
    return f"EARTHKIT_{name.upper().replace('-', '_')}_"


def _value_of_kind(name, klass, args, kwargs):
    """Return the value of an option without a getter from the arguments of set()."""
    if klass in (bool, int, float, str):
        # TODO: Proper exceptions
        assert len(args) == 1
        assert len(kwargs) == 0
        value = args[0]
        if value is not None and klass is bool:
            try:
                value = klass(value)
            except Exception as e:
                raise ValueError(f"Invalid value for config option '{name}': {value}") from e
        return value

    if klass is list:
        assert len(args) > 0
        assert len(kwargs) == 0
        if len(args) == 1 and isinstance(args[0], list):
            return args[0]
        return list(args)

    if klass is dict:
        assert len(args) <= 1
        if len(args) == 0:
            assert len(kwargs) > 0
            return kwargs
        assert len(kwargs) == 0
        return args[0]

    # any other type takes a single value
    assert len(args) == 1
    assert len(kwargs) == 0
    return args[0]


class Config:
    """The configuration of an earthkit package.

    Use :meth:`create` to get the config of a package, which is read from (and saved to)
    the config file. Creating a ``Config`` directly gives a config with the default values
    and without a file, unless ``config_file`` is passed.

    Parameters
    ----------
    name: str
        Name of the package without the ``earthkit-`` prefix, e.g. ``"data"``.
    options: dict
        Maps the name of each option to its :class:`ConfigOption`.
    version: str
        Version of the package. It is stored in the config file.
    config_file: str, optional
        File the config is saved to. No file is used when ``None``.
    env_prefix: str, optional
        Prefix of the environment variables that override the options.
        Defaults to ``EARTHKIT_<NAME>_``.

    Attributes
    ----------
    autosave: bool
        Save the config file when an option changes. Also applies to the temporary configs.
    """

    def __init__(
        self,
        name: str,
        options: dict[str, ConfigOption],
        *,
        version: str,
        config_file: str | None = None,
        env_prefix: str | None = None,
    ):
        self._name = name
        self._options = dict(options)
        self._version = version
        self._env_prefix = env_prefix if env_prefix is not None else _default_env_prefix(name)
        self._defaults = {k: v.default for k, v in self._options.items()}
        # The first layer holds the values of the config. Each active temporary config adds
        # a layer on top, which is where all the changes go until the temporary config ends.
        self._layers = [_Layer(dict(self._defaults), config_file)]
        self._callbacks: list[Callable[[], None]] = []
        self.autosave = True

    @classmethod
    def create(
        cls,
        name: str,
        options: dict[str, ConfigOption],
        *,
        version: str,
        config_file: str | None = None,
        env_prefix: str | None = None,
    ) -> "Config":
        """Create the config of a package, using its config file.

        The config file is ``config_file`` when it is given, otherwise the file set in the
        ``EARTHKIT_<NAME>_CONFIG_FILE`` environment variable, otherwise
        ``~/.config/earthkit/<name>/config.yaml``. The file is created when it is missing.
        """
        if env_prefix is None:
            env_prefix = _default_env_prefix(name)

        if not config_file:
            config_file = os.environ.get(env_prefix + "CONFIG_FILE")
        if not config_file:
            config_file = _default_config_file(name, version)
        # so that the file does not depend on the working directory
        config_file = os.path.abspath(config_file)

        config = cls(name, options, version=version, config_file=config_file, env_prefix=env_prefix)
        config._init_file()
        if config._load():
            config._save()
        return config

    @property
    def _values(self):
        return self._layers[-1].values

    @_values.setter
    def _values(self, values):
        self._layers[-1].values = values

    @property
    def _label(self):
        return f"earthkit-{self._name}"

    # ------------------------------------------------------------------
    # file
    # ------------------------------------------------------------------

    def _init_file(self):
        config_file = self._layers[0].file
        config_dir = os.path.dirname(config_file)

        try:
            if not os.path.exists(config_dir):
                ori_mask = os.umask(0o077)
                try:
                    os.makedirs(config_dir, 0o700)
                finally:
                    os.umask(ori_mask)
        except Exception:
            LOG.error(
                f"Cannot create {self._label} config directory={config_dir}",
                exc_info=True,
            )

        if os.path.exists(config_dir):
            try:
                if not os.path.exists(config_file):
                    self._write(config_file, self._defaults)
            except Exception:
                LOG.error(
                    f"Cannot save settings to {self._label} config file={config_file}",
                    exc_info=True,
                )

    def _load(self):
        """Read the config file. Return True if the file has to be saved again."""
        import yaml

        layer = self._layers[0]
        save = False
        try:
            with open(layer.file) as f:
                s = yaml.load(f, Loader=yaml.SafeLoader)
                if not isinstance(s, dict):
                    s = {}

                layer.values.update(s)

            # NOTE: this is a string comparison, which gives the wrong result for versions
            # like 1.10.0 and 1.9.0. A file without a version fails here, which is logged
            # as an error. When the file is saved again, the options that are not defined are
            # dropped. These behaviours are kept as they were in earthkit-data and earthkit-geo.
            if layer.values.get("version") < self._version:
                save = True

        except Exception:
            LOG.error(
                "Cannot load %s config (%s), reverting to defaults",
                self._label,
                layer.file,
                exc_info=True,
            )

        return save

    def _write(self, path, values):
        import yaml

        LOG.debug("Saving config")
        from filelock import FileLock

        lock = path + ".lock"
        with FileLock(lock):
            with open(path, "w") as f:
                print("# This file is automatically generated", file=f)
                print(file=f)

                for k, v in sorted(values.items()):
                    h = self._options.get(k)
                    if h:
                        print(file=f)
                        print("#", "-" * 76, file=f)
                        h.save(k, v, f)

                print(file=f)
                print("#", "-" * 76, file=f)
                print(f"# Version of {self._label}", file=f)
                print(file=f)
                yaml.dump({"version": self._version}, f, default_flow_style=False)
                print(file=f)
        try:
            os.unlink(lock)
        except OSError:
            pass

    def _save(self, layer=None):
        layer = layer if layer is not None else self._layers[-1]
        if layer.file is None:
            return

        try:
            self._write(layer.file, layer.values)
        except Exception:
            LOG.error(
                "Cannot save %s config (%s)",
                self._label,
                layer.file,
                exc_info=True,
            )

    def save_as(self, path: str) -> None:
        """Save the current values into the file at ``path``."""
        try:
            self._write(path, self._values)
        except Exception:
            LOG.error(
                f"Cannot save {self._label} config ({path})",
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # access
    # ------------------------------------------------------------------

    def get(self, name: str, default: Any = None) -> Any:
        """Get the value of a configuration option.

        Parameters
        ----------
        name: str
            The name of the configuration option.
        default: optional
            Not used, every option always has a value. Only kept so that existing calls work.

        Returns
        -------
        The value of the option. The value of the environment variable is used when it is set.
        """
        if name not in self._options:
            raise KeyError("No config name '%s'" % (name,))

        config_item = self._options[name]

        st, value, _ = self._env(name)
        if not st:
            value = self._values[name]

        if config_item.getter is None:
            return value
        return config_item.getter(value, name=name, none_ok=config_item.none_ok)

    def set(self, *args: Any, **kwargs: Any) -> None:
        """Set the value of a configuration option.

        Either all the options are set or, when one of them is invalid, none.

        Parameters
        ----------
        *args
            Either the name and the value of an option, or a dict mapping names to values.
        **kwargs
            Options as keyword arguments, with ``_`` instead of ``-`` in the names.
        """
        self._values.update(self._prepare_all(args, kwargs))
        self._changed()
        if len(args) == 0:
            # NOTE: with keyword arguments the changes are notified twice
            self._changed()

    def _prepare_all(self, args, kwargs):
        """Return the (name, value) pairs of the arguments of set(). Nothing is changed."""
        if len(args) == 0:
            assert len(kwargs) > 0
            items = [(k.replace("_", "-"), (v,), {}) for k, v in kwargs.items()]
        elif len(args) == 1 and isinstance(args[0], dict):
            items = [(k, (v,), {}) for k, v in args[0].items()]
        else:
            items = [(args[0], args[1:], kwargs)]

        return [(name, self._prepare(name, *a, **kw)) for name, a, kw in items]

    def _prepare(self, name: str, *args, **kwargs):
        """Check and convert the value of an option to what is stored."""
        if name not in self._options:
            raise KeyError("No config option '%s'" % (name,))

        self._warn_if_env(name, "set")

        config_item = self._options[name]

        if config_item.getter is not None:
            assert len(args) == 1
            assert len(kwargs) == 0
            value = args[0]
            # Check if value is properly formatted for getter
            config_item.getter(value, name=name, none_ok=config_item.none_ok)
        else:
            klass = config_item.kind
            value = _value_of_kind(name, klass, args, kwargs)
            if not isinstance(value, klass):
                raise TypeError("Config option '%s' must be of type '%s'" % (name, klass))

        config_item.validate(name, value)

        LOG.debug(f"set {name}={value} layers={len(self._layers)}")
        return value

    def reset(self, name: str | None = None) -> None:
        """Reset config to default values.

        Parameters
        ----------
        name: str, optional
            The name of the config option to reset to default. If `None` is passed, all config
            options are reset to their default values. Defaults to None.
        """
        if name is None:
            self._values = dict(self._defaults)
        else:
            if name not in self._defaults:
                raise KeyError("No config option name '%s'" % (name,))

            self._warn_if_env(name, "reset")
            self._values[name] = self._defaults[name]
        self._changed()

    def dump(self) -> Iterator[tuple[str, Any, ConfigOption | None]]:
        """Return an iterator over the sorted (name, value, option) of the current values.

        The option is ``None`` for values without a definition, such as ``version``.
        """
        values = self._values
        return ((k, v, self._options.get(k)) for k, v in sorted(values.items()))

    def _rows(self):
        """Return (name, value, default, env) for each value, with env being (variable, value) or None."""
        values = self._values
        env = self._env_vars(values)
        rows = []
        for k, v in sorted(values.items()):
            config = self._options.get(k, None)
            default = config.default if config else ""
            rows.append((k, v, default, env.get(k)))
        return rows

    def __repr__(self):
        r = []
        for k, v, default, env in self._rows():
            if env:
                r.append(f"{k}: ({env[0]}={env[1]}, {v}, {default})")
            else:
                r.append(f"{k}: ({v}, {default})")
        return "\n".join(r)

    def _repr_html_(self):
        html = [css("table")]
        html.append("<table class='ek'>")

        columns = ["Name", "Value", "Default"]
        html.append("<tr><th>%s</th></tr>" % ("</th><th>".join(columns)))

        for k, v, default, env in self._rows():
            if env:
                html.append("<tr><td>%s</td><td>%s=%r<br>(%r)</td><td>%r</td></tr>" % (k, env[0], env[1], v, default))
            else:
                html.append("<tr><td>%s</td><td>%r</td><td>%r</td></tr>" % (k, v, default))

        html.append("</table>")
        return "".join(html)

    # ------------------------------------------------------------------
    # changes
    # ------------------------------------------------------------------

    def _changed(self):
        if self.autosave:
            self._save()
        self._notify()

    def _notify(self):
        for cb in self._callbacks:
            cb()

    def on_change(self, callback: Callable[[], None]) -> None:
        """Call ``callback()`` when an option changes or a temporary config starts or ends."""
        self._callbacks.append(callback)

    def temporary(self, *args: Any, config_yaml: str | None = None, **kwargs: Any) -> ContextManager[None]:
        """Return a context manager that applies the options temporarily.

        Without options, the current values are used. Changes made in the context are not
        saved into the config file of the config, but into ``config_yaml`` when it is given.
        """
        layer = _Layer(dict(self._values), config_yaml)
        if len(args) > 0 or len(kwargs) > 0:
            layer.values.update(self._prepare_all(args, kwargs))
            # the callbacks are not notified until the layer is active
            if self.autosave:
                self._save(layer)
        return self._activate(layer)

    @contextlib.contextmanager
    def _activate(self, layer):
        self._layers.append(layer)
        try:
            self._notify()
            yield None
        finally:
            # not pop(): the contexts do not have to end in the order they started
            self._layers.remove(layer)
            self._notify()

    # ------------------------------------------------------------------
    # environment variables
    # ------------------------------------------------------------------

    def _env_var_name(self, name):
        return self._env_prefix + name.upper().replace("-", "_")

    def _has_env(self, name):
        return self._env_var_name(name) in os.environ

    def _env(self, name):
        name = self._env_var_name(name)
        if name in os.environ:
            return (True, os.environ.get(name), name)
        return (False, None, None)

    def _env_vars(self, values):
        r = {}
        for k in sorted(values.keys()):
            st, value, name = self._env(k)
            if st:
                r[k] = (name, value)
        return r

    def env(self) -> dict[str, tuple[str, str]]:
        """Return the options overridden by an environment variable as ``{name: (variable, value)}``."""
        return self._env_vars(self._layers[0].values)

    def _warn_if_env(self, name, action):
        if self._has_env(name):
            msg = (
                f"Config option '{name}' is also set by environment variable '{self._env_var_name(name)}'."
                "The environment variable takes precedence and its value is returned when calling get()."
            )
            if self.autosave:
                msg += f" Still, the value {action} here will be saved to the config file."
            warnings.warn(msg)
