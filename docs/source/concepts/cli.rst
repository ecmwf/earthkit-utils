Command line interface standards
================================

earthkit-utils provides the ``earthkit`` command line tool, which is shared by the whole earthkit ecosystem.
earthkit-utils itself only provides the ``earthkit info`` command: the other commands are contributed by the
installed earthkit packages, e.g. ``earthkit ls`` by earthkit-data and ``earthkit regrid`` by earthkit-geo.

This page describes the standards that the commands of all earthkit packages follow, so that they behave
consistently: how packages contribute commands, the common arguments of :mod:`earthkit.cli.standard_args` that
read and write data, the selection syntax, and the conventions for writing commands. See
:doc:`../tutorials/cli-commands` for a step-by-step guide to writing commands, and the `earthkit documentation
<https://earthkit.readthedocs.io/en/latest/tutorials/cli.html>`_ for the commands of all the earthkit packages.

The ``earthkit`` command
------------------------

Installing earthkit-utils installs the ``earthkit`` console script, which runs the `click <https://click.palletsprojects.com>`_ group
:data:`earthkit.cli.main.earthkit`. Every command is registered on this group:

.. code:: bash

   earthkit -h               # list the commands
   earthkit COMMAND -h       # help of a command
   earthkit info             # installed earthkit packages, their versions and their commands
   earthkit -V               # version of earthkit-utils

The ``earthkit.cli`` namespace package
--------------------------------------

``earthkit.cli`` is a *namespace package*: it has no ``__init__.py``, so that every earthkit package can install
modules into it. earthkit-utils installs ``earthkit/cli/main.py``, which defines the ``earthkit`` group, and
``earthkit/cli/standard_args.py``, and each package that contributes commands installs a module named after the
package:

.. code:: text

   earthkit/cli/
   ├── main.py             # earthkit-utils: the earthkit group and the info command
   ├── standard_args.py    # earthkit-utils: shared arguments and options
   ├── data.py             # earthkit-data: ls, convert, sel, isel, order_by
   ├── plots.py            # earthkit-plots: plot
   ├── geo/                # earthkit-geo, as a sub-package
   │   ├── __init__.py     #   imports the submodules below
   │   └── grids.py        #   regrid
   └── transforms/         # earthkit-transforms, as a sub-package
       ├── __init__.py
       ├── _tools.py       #   private helpers, ignored by the discovery
       └── temporal.py     #   daily-agg, monthly-agg, yearly-agg

A command module registers its commands on the shared group when it is imported:

.. code:: python

   # earthkit/cli/mypackage.py, shipped by earthkit-mypackage
   import click

   from earthkit.cli.main import earthkit


   @earthkit.command()
   @click.argument("path")
   def describe(path):
       """Describe the data in PATH."""
       import earthkit.mypackage as ekm

       click.echo(ekm.describe(path))

which makes ``earthkit describe`` available once earthkit-mypackage is installed. The name of the module is the
name of the package without its ``earthkit-`` prefix: ``earthkit info`` uses it to show the version of the
package and the commands it provides.

A package with many commands can install a sub-package ``earthkit/cli/<name>/`` instead, whose ``__init__.py``
imports the modules that define the commands, usually mirroring the submodules of the package, e.g.
``earthkit.cli.geo.grids`` for ``earthkit.geo.grids``. Only ``earthkit.cli.<name>`` itself is imported by
the discovery, and the commands of all its submodules are attributed to ``earthkit-<name>``.

Discovery of the commands
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``earthkit`` group, an :class:`earthkit.cli.main.EarthkitCLI`, imports the command modules the first time
the commands are needed, e.g. to list them for ``earthkit -h`` or to run one of them. It imports every module
of ``earthkit.cli`` returned by :func:`earthkit.cli.main.command_modules`, in alphabetical order, except:

- ``main`` and ``standard_args``, the modules of earthkit-utils,
- modules whose name starts with an underscore, which can hold private helpers.

Command names are shared by all the packages. If two modules register a command with the same name, the first
one imported wins, and a warning is emitted for the other one.

A module that fails to import, e.g. because of a missing dependency, is skipped, so that the commands of the
other packages keep working. A warning naming the package and the error is printed on stderr, ``earthkit info``
shows the error next to the package, and the error for an unknown command names the packages whose commands
could not be loaded:

.. code:: console

   $ earthkit info
   Warning: the commands of earthkit-mypackage could not be loaded: ModuleNotFoundError: No module named 'scipy'
   earthkit-utils        1.1.0
   earthkit-mypackage    0.1.0  (failed to load: ModuleNotFoundError: No module named 'scipy')

Keeping the command line fast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As every command module is imported to list or run any command, the command modules live outside of the
``earthkit.<package>`` packages, so that importing them does not run the ``__init__.py`` of the packages. Only
import `click <https://click.palletsprojects.com>`_, ``earthkit.cli`` and light standard library modules at
module level, and import anything else, including the package itself, inside the command functions. Otherwise
``earthkit -h`` becomes slow, and a missing optional dependency of the package would make all its commands
unavailable, rather than only the command that needs it.

Packaging
~~~~~~~~~

The ``earthkit.cli`` modules must be included in the distribution of the package, which depends on
earthkit-utils, e.g. with setuptools:

.. code:: toml

   [project]
   dependencies = ["earthkit-data", "earthkit-utils"]

   [tool.setuptools.packages.find]
   include = ["earthkit.mypackage", "earthkit.cli"]
   where = ["src/"]

No entry point is needed: installing the module into ``earthkit/cli/`` is enough for the command to be found.

Common arguments
----------------

:mod:`earthkit.cli.standard_args` defines the arguments and options shared by the earthkit commands, so that
all commands read and write data in the same way. They are plain `click <https://click.palletsprojects.com>`_
decorators, which the packages import and reuse on any number of commands.

.. list-table::
   :header-rows: 1
   :widths: 18 8 24 22 28

   * - Argument
     - Short flag
     - Defined by
     - Value
     - Used by
   * - ``SOURCE`` (positional), ``--source`` (option)
     - ``-s``
     - :func:`~earthkit.cli.standard_args.source_options`
     - ``[NAME:]VALUE``, repeatable, see `Sources`_. The command gets the object returned by
       :func:`earthkit.data.from_source() <earthkit-data:from_source>`.
     - ``ls``, ``convert``, ``sel``, ``isel``, ``order_by``, ``daily-agg``, ``monthly-agg``, ``yearly-agg``,
       ``regrid``, ``plot``
   * - ``TARGET`` (positional), ``--target`` (option)
     - ``-t``
     - :func:`~earthkit.cli.standard_args.target_options`
     - ``[NAME:]VALUE``, see `Targets`_. The command gets a :class:`~earthkit.cli.standard_args.Target`.
     - ``convert``, ``sel``, ``isel``, ``order_by``, ``daily-agg``, ``monthly-agg``, ``yearly-agg``,
       ``regrid``
   * - ``--profile``
     - ``-p``
     - :data:`~earthkit.cli.standard_args.profile_option`
     - Profile name or YAML file, repeatable, see `Profiles and indexes`_. The command gets a list, or ``None``.
     - ``convert``, ``daily-agg``, ``monthly-agg``, ``yearly-agg``
   * - ``--index``
     - ``-i``
     - :data:`~earthkit.cli.standard_args.index_option`
     - Integer, comma-separated list or slice, see `Profiles and indexes`_. The command gets an ``int``, a list
       of ints, a ``slice``, or ``None``.
     - ``plot``

Short flags
~~~~~~~~~~~

Only the common arguments have short flags, so that a short flag means the same in every command. All the other
options of the commands are long-only, apart from ``-h``/``--help``, which every command has, and the
``-V``/``--version`` option of ``earthkit`` itself. Sources and targets with a name other than the default
``source`` and ``target``, e.g. ``--source-1``, have no short flag either.

Helpers
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Helper
     - Description
   * - :func:`~earthkit.cli.standard_args.add_options`
     - Applies a list of arguments and options to a command, in the order they are listed.
   * - :func:`~earthkit.cli.standard_args.split_csv`
     - Callback of a repeatable option taking comma-separated values, turning ``--keys a,b --keys c`` into
       ``["a", "b", "c"]``.
   * - :data:`~earthkit.cli.standard_args.SOURCE_HELP`, :data:`~earthkit.cli.standard_args.TARGET_HELP`
     - The shared descriptions of a source and of a target. Click arguments have no help, so commands with a
       positional ``SOURCE`` or ``TARGET`` build their help from these, e.g. ``f"SOURCE: {SOURCE_HELP}"``.
   * - :func:`~earthkit.cli.standard_args.parse_index`
     - Parses an index, as ``--index`` does, for commands taking indexes in other options.

A typical command reads a source and writes to a target:

.. code:: python

   import click

   from earthkit.cli.main import earthkit
   from earthkit.cli.standard_args import SOURCE_HELP, TARGET_HELP, add_options, source_options, target_options


   @earthkit.command(
       help=f"""Compute the time mean of SOURCE and write it to TARGET.

   SOURCE: {SOURCE_HELP}

   TARGET: {TARGET_HELP}
   """
   )
   @add_options([source_options(positional=True), target_options(positional=True)])
   def time_mean(source, target):
       target.to_target(source.to_xarray().mean("time"))

.. code:: bash

   earthkit time-mean input.grib output.nc

Options and positional arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, :func:`~earthkit.cli.standard_args.source_options` and
:func:`~earthkit.cli.standard_args.target_options` add required options, named after their ``name`` argument,
with dashes instead of underscores. The default ``source`` and ``target`` also have the short flags ``-s`` and
``-t``. A command with several sources or targets gives each of them a different name:

.. code:: python

   @earthkit.command()
   @add_options([source_options("source_1"), source_options("source_2"), target_options()])
   def combine(source_1, source_2, target): ...

.. code:: bash

   earthkit combine --source-1 a.grib --source-2 b.grib -t out.grib

With ``positional=True``, they add positional arguments instead, named after ``name`` in upper case, e.g.
``SOURCE`` and ``TARGET``:

.. code:: bash

   earthkit copy a.grib b.grib out.grib    # SOURCE... TARGET

A positional source takes one or more values, so a command can only have one. Click arguments have no help of
their own, so describe them in the help of the command with
:data:`~earthkit.cli.standard_args.SOURCE_HELP` and :data:`~earthkit.cli.standard_args.TARGET_HELP`.

Sources
~~~~~~~

A source is given as ``[NAME:]VALUE``, split on the first ``:``. ``NAME`` is the name of an earthkit-data
source, ``file`` if not given, and ``VALUE`` gives its arguments:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Value
     - ``earthkit.data.from_source`` call
   * - ``input.grib``
     - ``from_source("file", "input.grib")``
   * - ``'data/*.nc'``
     - ``from_source("file", "data/*.nc")``, a glob pattern
   * - ``url:https://myhost.int/file.nc``
     - ``from_source("url", "https://myhost.int/file.nc")``
   * - ``'mars:{"param": "2t", "levtype": "sfc"}'``
     - ``from_source("mars", request={"param": "2t", "levtype": "sfc"})``
   * - ``'cds:{"dataset": "reanalysis-era5-single-levels", "variable": "2t"}'``
     - ``from_source("cds", "reanalysis-era5-single-levels", request={"variable": "2t"})``
   * - ``'mars:[{"param": "2t"}, {"param": "msl"}]'``
     - ``from_source("mars", request=[{"param": "2t"}, {"param": "msl"}])``
   * - ``-``
     - ``from_source("stream", sys.stdin.buffer)``, GRIB data read from stdin
   * - ``a.grib b.grib``
     - ``from_source("multi", from_source("file", "a.grib"), from_source("file", "b.grib"))``

- A JSON ``VALUE``, either an object or a list of objects, is passed as the ``request`` of the source. Its
  ``dataset`` key, if any, is passed as the first argument instead, and must be the same in all the requests of
  a list.
- Any other ``VALUE`` is passed as the first argument of the source. ``NAME:`` alone calls the source without
  arguments.
- The paths of ``file`` sources, or the glob patterns, must match existing files.
- Several sources, given as several values of the positional argument or by repeating the option, are merged
  into a single ``multi`` source.

``NAME`` must start with a letter and contain only letters, digits, ``_`` and ``-``. Anything else before the
first ``:`` is part of ``VALUE``, e.g. ``./data:2020.nc`` is a file. Prefix paths starting with a Windows drive
letter with ``file:``, e.g. ``file:C:\data\input.grib``, as ``C`` would otherwise be taken as the name of the
source.

The sources are opened while the command line is parsed, before the command runs. Invalid values, e.g. a
missing file or invalid JSON, are reported as usage errors.

Targets
~~~~~~~

A target is also given as ``[NAME:]VALUE``, with ``NAME`` the name of an earthkit-data target, ``file`` if not
given:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Value
     - ``earthkit.data.to_target`` call
   * - ``output.grib``
     - ``to_target("file", "output.grib", data=...)``
   * - ``'file:{"file": "out.grib", "append": true}'``
     - ``to_target("file", file="out.grib", append=True, data=...)``
   * - ``'zarr:{"xarray_to_zarr_kwargs": {"store": "out.zarr"}}'``
     - ``to_target("zarr", xarray_to_zarr_kwargs={"store": "out.zarr"}, data=...)``
   * - ``-``
     - ``to_target("file", sys.stdout.buffer, data=...)``, written to stdout

A JSON ``VALUE`` must be an object, passed as the keyword arguments of the target. The path of a ``file``
target must not be a directory.

The command gets a :class:`~earthkit.cli.standard_args.Target`, and writes data to it with
:meth:`~earthkit.cli.standard_args.Target.to_target`. Keyword arguments given to ``to_target`` are added to
those of the target, and take precedence over them, e.g. earthkit-data writes NetCDF data to stdout with:

.. code:: python

   target.to_target(dataset, encoder="netcdf")

Profiles and indexes
~~~~~~~~~~~~~~~~~~~~

:data:`~earthkit.cli.standard_args.profile_option` adds ``--profile``/``-p``, which can be repeated. Each value
is the name of an earthkit-data built-in profile, e.g. ``earthkit`` or ``mars``, or the path of a YAML file
holding a profile, and the command passes the list to ``to_xarray(profile=...)``, which layers the profiles in
order.

:data:`~earthkit.cli.standard_args.index_option` adds ``--index``/``-i``, which selects items by position. Its
value is parsed by :func:`~earthkit.cli.standard_args.parse_index`:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Value
     - Index
   * - ``3``
     - ``3``
   * - ``4,5,8``
     - ``[4, 5, 8]``
   * - ``1:7``, ``::2``, ``-3:``
     - ``slice(1, 7)``, ``slice(None, None, 2)``, ``slice(-3, None)``: slices exclude their stop, as in Python

Invalid values are reported as usage errors.

Selection syntax
----------------

Commands select data either by value or by position, with slices that follow the convention of the
corresponding Python methods:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Selection
     - Commands
     - Slices
   * - By value
     - ``earthkit sel``
     - Include their stop, as ``FieldList.sel`` and ``xarray.Dataset.sel``.
   * - By position
     - ``earthkit isel``, ``--index``
     - Exclude their stop, as in Python indexing, see `Profiles and indexes`_.

Selecting by value
~~~~~~~~~~~~~~~~~~

The conditions of ``earthkit sel`` are given as ``--KEY VALUE`` or ``--KEY=VALUE``, where ``KEY`` is an
earthkit-data metadata key for GRIB and BUFR data, or a coordinate name for NetCDF data. ``VALUE`` is a scalar, a
comma-separated list or a slice ``START:STOP[:STEP]``:

.. code:: bash

   earthkit sel in.grib out.grib --parameter.variable t                   # scalar
   earthkit sel in.grib out.grib --parameter.variable t,u                 # list
   earthkit sel in.grib out.grib --vertical.level 300:500                 # 300, 400 and 500
   earthkit sel in.grib out.grib --vertical.level 300:700:200             # 300, 500 and 700
   earthkit sel in.grib out.grib --vertical.level 700:                    # 700 and above
   earthkit sel in.grib out.grib --vertical.level :500                    # 500 and below
   earthkit sel in.grib out.grib --time.valid_datetime '2020-12-21T12:00:2020-12-21T18:00'   # date-time slice
   earthkit sel in.grib out.grib --vertical.level 300:500,850             # integer slice in a list

   # Quoting decides the type of the bounds
   earthkit sel in.nc out.nc --year 2000:2001          # slice(2000, 2001)
   earthkit sel in.nc out.nc --year '"2000":"2001"'    # slice("2000", "2001")

- Slices include their stop. The bounds are numbers, ISO date-times or quoted strings, of the same type, and
  either bound can be left out.
- A step needs integer bounds, and selects the values from ``START`` to ``STOP`` by ``STEP``.
- An integer slice in a list stands for every integer from its start to its stop, e.g. ``300:500,850`` selects
  all the integer values from 300 to 500, and 850.
- A single ISO date-time, e.g. ``2020-12-21T21:00:00``, is a value, not a slice.
- Scalars and list items are compared after conversion to the type of the metadata, so ``01400`` keeps its
  leading zero for string keys.
- Conditions on different keys are combined with AND, and each key can only be given once.

Commands of other packages that select data by value follow the same syntax.

Writing commands
----------------

Commands contributed by the earthkit packages follow these conventions:

- Use the common arguments of :mod:`earthkit.cli.standard_args` where they fit, so that all commands read and
  write data in the same way.
- Only the common arguments have short flags: make the other options long-only, see `Short flags`_.
- Keep the command modules light, and import heavy modules inside the commands, see
  `Keeping the command line fast`_.
- In a sub-package of commands, mirror the structure of the package the commands wrap, e.g.
  ``earthkit.cli.transforms.temporal`` for the commands wrapping ``earthkit.transforms.temporal``.
- Select data with the `Selection syntax`_: slices include their stop when selecting by value, and exclude it
  when selecting by position.
- Describe positional ``SOURCE`` and ``TARGET`` arguments in the help of the command with
  :data:`~earthkit.cli.standard_args.SOURCE_HELP` and :data:`~earthkit.cli.standard_args.TARGET_HELP`, and end
  the help with examples, in a paragraph starting with click's ``\b`` marker so that they are not rewrapped.
- Leave options that wrap the arguments of a function without a default (``None``), and only pass on the options
  that are given, so that the defaults of the function apply rather than duplicating them in the command.
- Report invalid input with :class:`click.BadParameter`, :class:`click.UsageError` or
  :class:`click.ClickException`, which click prints as an error message rather than a traceback.
- Test the commands with :class:`click.testing.CliRunner`, invoking the shared group after importing the command
  module:

  .. code:: python

     from click.testing import CliRunner

     import earthkit.cli.mypackage  # noqa: F401, registers the commands
     from earthkit.cli.main import earthkit


     def test_describe(tmp_path):
         result = CliRunner().invoke(earthkit, ["describe", "input.grib"])
         assert result.exit_code == 0, result.output
