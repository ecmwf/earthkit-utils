Units
=====

The ``earthkit.utils.units`` subpackage provides unit conversion for NumPy
arrays, xarray DataArrays, and xarray Datasets.

**Built on Pint**

All unit parsing and arithmetic is delegated to `Pint
<https://pint.readthedocs.io>`_, a Python units library that understands a
wide vocabulary of physical units and can perform dimensionally-correct
conversions (including offset units such as degrees Celsius and Kelvin).
Internally, every unit string is resolved into a ``pint.Unit`` object via a
shared ``UnitRegistry``; the actual value conversion is then carried out by
Pint's ``Quantity.to()`` method.

**UDUNITS-style notation**

Many scientific datasets (particularly those following CF conventions) express
units in `UDUNITS <https://www.unidata.ucar.edu/software/udunits/>`_ notation,
where exponents are written without a caret (``m s-1`` instead of ``m/s``) and
components are separated by spaces. earthkit-utils normalises these strings
into Pint-compatible form before parsing, so both styles are accepted
interchangeably:

.. code:: python

   from earthkit.utils.units import convert_units, are_equal

   are_equal("m s-1", "m/s")   # True
   are_equal("kg m-2", "kg/m²")  # True

**Graceful degradation**

If a unit string is not recognised by Pint, earthkit-utils stores it as an
opaque ``StrUnits`` object rather than raising an error. Conversion between two
such unrecognised units is skipped and the original data is returned unchanged,
with a warning logged. This ensures that pipelines handling heterogeneous or
non-standard metadata remain robust.
