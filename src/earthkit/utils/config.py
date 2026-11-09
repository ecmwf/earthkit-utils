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
import copy
import functools
import logging
import os
import warnings
from abc import ABCMeta, abstractmethod
from typing import Callable

from earthkit.utils.html import css
from earthkit.utils.humanise import list_to_human

LOG = logging.getLogger(__name__)

NONE = object()


class Validator(metaclass=ABCMeta):
    """Check the values of a config option."""

    # The validator classes used by ``make()`` for the type of a value
    _validators = {}

    @abstractmethod
    def check(self, value):
        pass

    @abstractmethod
    def explain(self):
        pass

    @staticmethod
    def make(value):
        v = Validator._validators.get(type(value), None)
        if v is not None:
            return v(value)
        else:
            raise TypeError(f"Cannot create Validator for type={type(value)}")


class ValueValidator(Validator):
    def __init__(self, value):
        self.value = value

    def check(self, value):
        return value == self.value

    def explain(self):
        return f"Valid when = {self.value}."


class ListValidator(Validator):
    """Check if a value is in a list of valid values."""

    def __init__(self, values):
        self.values = values

    def check(self, value):
        return value in self.values

    def explain(self):
        return f"Valid values: {list_to_human(self.values)}."


Validator._validators.update({bool: ValueValidator, list: ListValidator})


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
        default,
        description,
        getter=None,
        none_ok=False,
        kind=None,
        docs_default=None,
        validator=None,
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


def forward(func):
    """Run the method on the innermost active temporary config, if there is one."""

    @functools.wraps(func)
    def wrapped(self, *args, **kwargs):
        # only the root config has a stack, the temporary configs are applied directly
        if self._stack:
            return func(self._stack[-1], *args, **kwargs)
        return func(self, *args, **kwargs)

    return wrapped


class _Shared:
    """State shared by a config and its temporary configs."""

    def __init__(self):
        self.autosave = True


def _default_config_file(name, version):
    # NOTE: the file does not depend on the version yet. The version is an argument so that
    # the location can become version specific without changing the callers.
    return os.path.join(os.path.expanduser("~/.config/earthkit"), name, "config.yaml")


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
    """

    def __init__(self, name, options, *, version, config_file=None, env_prefix=None):
        self._name = name
        self._options = dict(options)
        self._version = version
        self._config_file = config_file
        self._env_prefix = env_prefix if env_prefix is not None else _default_env_prefix(name)
        self._defaults = {k: v.default for k, v in self._options.items()}
        self._config = dict(self._defaults)
        self._callbacks = []
        self._stack = []
        self._root = self
        self._shared = _Shared()

    @classmethod
    def create(cls, name, options, *, version, config_file=None, env_prefix=None):
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

        config = cls(name, options, version=version, config_file=config_file, env_prefix=env_prefix)
        config._init_file()
        if config._load():
            config._save()
        return config

    # ------------------------------------------------------------------
    # file
    # ------------------------------------------------------------------

    @property
    def _label(self):
        return f"earthkit-{self._name}"

    def _init_file(self):
        config_file = self._config_file

        try:
            config_dir = os.path.dirname(config_file)
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

        save = False
        try:
            with open(self._config_file) as f:
                s = yaml.load(f, Loader=yaml.SafeLoader)
                if not isinstance(s, dict):
                    s = {}

                self._config.update(s)

            # NOTE: this is a string comparison, which gives the wrong result for versions
            # like 1.10.0 and 1.9.0. A file without a version fails here, which is logged
            # as an error. When the file is saved again, the options that are not defined are
            # dropped. These behaviours are kept as they were in earthkit-data and earthkit-geo.
            if self._config.get("version") < self._version:
                save = True

        except Exception:
            LOG.error(
                "Cannot load %s config (%s), reverting to defaults",
                self._label,
                self._config_file,
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

    def _save(self):
        if self._config_file is None:
            return

        try:
            self._write(self._config_file, self._config)
        except Exception:
            LOG.error(
                "Cannot save %s config (%s)",
                self._label,
                self._config_file,
                exc_info=True,
            )

    @forward
    def save_as(self, path):
        try:
            self._write(path, self._config)
        except Exception:
            LOG.error(
                f"Cannot save {self._label} config ({path})",
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # access
    # ------------------------------------------------------------------

    @forward
    def get(self, name: str, default=NONE):
        """Get the value of a configuration option.

        Parameters
        ----------
        name: str
            The name of the configuration option.
        default: optional
            Value returned when the option has no value. Defaults to NONE, which raises an error.

        Returns
        -------
        The value of the option. The value of the environment variable is used when it is set.
        """
        if name not in self._options:
            raise KeyError("No config name '%s'" % (name,))

        config_item = self._options[name]

        st, value, _ = self._env(name)
        if not st:
            if default is NONE:
                value = self._config[name]
            else:
                value = self._config.get(name, default)

        if config_item.getter is None:
            return value
        return config_item.getter(value, name=name, none_ok=config_item.none_ok)

    @forward
    def set(self, *args, **kwargs):
        """Set the value of a configuration option.

        Parameters
        ----------
        *args
            Either the name and the value of an option, or a dict mapping names to values.
        **kwargs
            Options as keyword arguments, with ``_`` instead of ``-`` in the names.
        """
        if len(args) == 0:
            assert len(kwargs) > 0
            for k, v in kwargs.items():
                self._set(k.replace("_", "-"), v)
            self._changed()
        else:
            if len(args) == 1 and isinstance(args[0], dict):
                for k, v in args[0].items():
                    self._set(k, v)
            else:
                self._set(*args, **kwargs)
        # NOTE: with keyword arguments the changes are notified twice
        self._changed()

    def _set(self, name: str, *args, **kwargs):
        if name not in self._options:
            raise KeyError("No config option '%s'" % (name,))

        self._warn_if_env(name, "set")

        config_item = self._options[name]

        klass = config_item.kind

        if klass in (bool, int, float, str):
            # TODO: Proper exceptions
            assert len(args) == 1
            assert len(kwargs) == 0
            value = args[0]
            if value is not None:
                if klass is bool:
                    try:
                        value = klass(value)
                    except Exception as e:
                        raise ValueError(f"Invalid value for config option '{name}': {value}") from e

        if klass is list:
            assert len(args) > 0
            assert len(kwargs) == 0
            value = list(args)
            if len(args) == 1 and isinstance(args[0], list):
                value = args[0]

        if klass is dict:
            assert len(args) <= 1
            if len(args) == 0:
                assert len(kwargs) > 0
                value = kwargs

            if len(args) == 1:
                assert len(kwargs) == 0
                value = args[0]

        if config_item.getter is not None:
            assert len(args) == 1
            assert len(kwargs) == 0
            value = args[0]
            # Check if value is properly formatted for getter
            config_item.getter(value, name=name, none_ok=config_item.none_ok)
        else:
            if not isinstance(value, klass):
                raise TypeError("Config option '%s' must be of type '%s'" % (name, klass))

        config_item.validate(name, value)
        self._config[name] = value

        LOG.debug(f"_set {name}={value} stack_size={len(self._root._stack)}")

    @forward
    def reset(self, name: str = None):
        """Reset config to default values.

        Parameters
        ----------
        name: str, optional
            The name of the config option to reset to default. If `None` is passed, all config
            options are reset to their default values. Defaults to None.
        """
        if name is None:
            self._config = dict(**self._defaults)
        else:
            if name not in self._defaults:
                raise KeyError("No config option name '%s'" % (name,))

            self._config.pop(name, None)
            self._warn_if_env(name, "reset")
            self._config[name] = self._defaults[name]
        self._changed()

    @forward
    def dump(self):
        for k, v in sorted(self._config.items()):
            yield ((k, v, self._options.get(k)))

    @forward
    def __repr__(self):
        r = []
        env = self.env()
        for k, v in sorted(self._config.items()):
            config = self._options.get(k, None)
            default = config.default if config else ""
            if k in env:
                r.append(f"{k}: ({env[k][0]}={env[k][1]}, {v}, {default})")
            else:
                r.append(f"{k}: ({v}, {default})")
        return "\n".join(r)

    @forward
    def _repr_html_(self):
        html = [css("table")]
        html.append("<table class='ek'>")

        env = self.env()
        columns = ["Name", "Value", "Default"]
        html.append("<tr><th>%s</th></tr>" % ("</th><th>".join(columns)))

        for k, v in sorted(self._config.items()):
            config = self._options.get(k, None)
            default = config.default if config else ""
            if k in env:
                html.append(
                    "<tr><td>%s</td><td>%s=%r<br>(%r)</td><td>%r</td></tr>" % (k, env[k][0], env[k][1], v, default)
                )
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

    def on_change(self, callback: Callable[[], None]):
        self._callbacks.append(callback)

    @property
    def autosave(self):
        """Save the config file when an option changes (shared with the temporary configs)."""
        return self._shared.autosave

    @autosave.setter
    def autosave(self, v):
        self._shared.autosave = v

    @forward
    def temporary(self, *args, config_yaml=None, **kwargs):
        """Return a context manager that applies the options temporarily.

        Without options, the current values are used. Changes made in the context are not
        saved into the config file of the config, but into ``config_yaml`` when it is given.
        """
        tmp = copy.copy(self)
        tmp._config = dict(self._config)
        tmp._config_file = config_yaml
        tmp._stack = []
        # until the tmp object is at the top of the stack we do not want
        # notify the observers
        if len(args) > 0 or len(kwargs) > 0:
            # tmp does not have any callbacks so it will not broadcast the changes
            tmp._callbacks = []
            tmp.set(*args, **kwargs)
        tmp._callbacks = self._callbacks
        return self._root._push(tmp)

    @contextlib.contextmanager
    def _push(self, tmp):
        self._stack.append(tmp)
        self._notify()
        try:
            yield None
        finally:
            self._stack.pop()
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

    def env(self):
        r = {}
        for k in sorted(self._config.keys()):
            st, value, name = self._env(k)
            if st:
                r[k] = (name, value)
        return r

    def _warn_if_env(self, name, action):
        if self._has_env(name):
            msg = (
                f"Config option '{name}' is also set by environment variable '{self._env_var_name(name)}'."
                "The environment variable takes precedence and its value is returned when calling get()."
            )
            if self.autosave:
                msg += f" Still, the value {action} here will be saved to the config file."
            warnings.warn(msg)


def _default_env_prefix(name):
    return f"EARTHKIT_{name.upper().replace('-', '_')}_"
