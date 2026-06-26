Version 1.0 Updates
/////////////////////////


Version 1.0.0
=============

This is the first public release of **earthkit-utils** — a library of small,
shared utility components for the EarthKit ecosystem.

- **Units** (``earthkit.utils.units``) — unit conversion for NumPy arrays,
  xarray DataArrays, and xarray Datasets, built on Pint with support for
  UDUNITS/CF-convention notation.
- **Array utilities** (``earthkit.utils.array``) — backend-agnostic array
  helpers supporting NumPy, CuPy, PyTorch, and JAX via ``array-api-compat``,
  including namespace detection and cross-backend array conversion.
- **Decorators** (``earthkit.utils.decorators``) — reusable decorators for
  input-type dispatching, argument coercion, xarray ufunc wrapping, experimental
  API marking, and thread-safe cached properties.
- **Constants** (``earthkit.utils.constants``) — numeric conversion factors for
  degrees and radians.
