Installation and Getting Started
================================

Installing from PyPI
--------------------

Install the latest release with pip:

.. code:: bash

   pip install earthkit-utils

Import and use
--------------

A minimal import that confirms the package is available:

.. code:: python

   import earthkit.utils as eku

   print(eku.constants.radian)  # Example usage of a constant

The library is organised into the ``array``, ``decorators``, and ``units`` subpackages, with additional
module-level constants under ``earthkit.utils.constants``.
