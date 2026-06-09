Frequently asked questions
==========================

How do I convert my xarray units from Celsius to Kelvin?
---------------------------------------------------------

Use :func:`earthkit.utils.units.convert_units` with ``target_units="K"``. If your
``DataArray`` already has a ``units`` attribute set to ``"degC"``, the source units
are detected automatically:

.. code:: python

   import xarray as xr
   from earthkit.utils.units import convert_units

   temperature = xr.DataArray([0.0, 20.0, 100.0], dims="x", attrs={"units": "degC"})
   temperature_K = convert_units(temperature, target_units="K")
   print(temperature_K.values)         # [273.15 293.15 373.15]
   print(temperature_K.attrs["units"]) # K

For a full worked example — including Dataset conversion and explicit source-unit
overrides — see the how-to guide:
:doc:`how-tos/convert-xarray-from-celsius-to-kelvin`.
