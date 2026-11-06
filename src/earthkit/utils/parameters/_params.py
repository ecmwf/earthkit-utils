# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""
Define earthkit parameter metadata.

Used for the various meteorological parameters earthkit-meteo computes. This metadata includes:

- "variable": define the Field.parameter.variable
- "units": define the Field.parameter.units
- "paramId": define the ecCodes GRIB parameter ID. If does not apply, set to None.
- "edition": define the ecCodes GRIB edition. If does not apply, set to None.
- "long_name": the long name for the parameter
- "standard_name": the CF standard name for the parameter. If does not apply, set to None.
- "cf_units": the CF canonical units for "standard_name". If does not apply, set to None.

The keys of the dictionary are the names of the parameters used only internally by
various earthkit-meteo functions.
"""

_PARAMS = {
    "10m_wind_direction": {
        "variable": "10wdir",
        "units": "degrees",
        "paramId": 260260,
        "edition": ("2",),
        "long_name": "10 metre wind direction",
        "standard_name": "wind_from_direction",
        "cf_units": "degree",
    },
    "10m_wind_speed": {
        "variable": "10si",
        "units": "m/s",
        "paramId": 207,
        "edition": (
            "1",
            "2",
        ),
        "long_name": "10 metre wind speed",
        "standard_name": "wind_speed",
        "cf_units": "m s-1",
    },
    "100m_wind_speed": {
        "variable": "100si",
        "units": "m/s",
        "paramId": 228249,
        "edition": (
            "1",
            "2",
        ),
        "long_name": "100 metre wind speed",
        "standard_name": "wind_speed",
        "cf_units": "m s-1",
    },
    "200m_wind_speed": {
        "variable": "200si",
        "units": "m/s",
        "paramId": 228241,
        "edition": (
            "1",
            "2",
        ),
        "long_name": "200 metre wind speed",
        "standard_name": "wind_speed",
        "cf_units": "m s-1",
    },
    "2m_dewpoint": {
        "variable": "2d",
        "units": "K",
        "paramId": 168,
        "edition": (
            "1",
            "2",
        ),
        "long_name": "2 metre dewpoint",
        "standard_name": "dew_point_temperature",
        "cf_units": "K",
    },
    "2m_relative_humidity": {
        "variable": "2r",
        "units": "%",
        "paramId": 260242,
        "edition": (
            "1",
            "2",
        ),
        "long_name": "2 metre relative humidity",
        "standard_name": "relative_humidity",
        "cf_units": "1",
    },
    "2m_specific_humidity": {
        "variable": "2sh",
        "units": "kg/kg",
        "paramId": 174096,
        "edition": (
            "1",
            "2",
        ),
        "long_name": "2 metre specific humidity",
        "standard_name": "specific_humidity",
        "cf_units": "1",
    },
    "2m_temperature": {
        "variable": "2t",
        "units": "K",
        "paramId": 167,
        "edition": (
            "1",
            "2",
        ),
        "long_name": "2 metre temperature",
        "standard_name": "air_temperature",
        "cf_units": "K",
    },
    "coriolis": {
        "variable": "fc",
        "units": "1/s",
        "long_name": "Coriolis parameter",
        "standard_name": "coriolis_parameter",
        "cf_units": "s-1",
    },
    "cos_solar_zenith_angle": {
        "variable": "cossza",
        "units": "1",
        "paramId": 214001,
        "edition": (
            "1",
            "2",
        ),
        "long_name": "Cosine of solar zenith angle",
        "standard_name": None,
        "cf_units": None,
    },
    "cos_solar_zenith_angle_integrated": {
        "variable": "cossza_integrated",
        "units": "1",
        "long_name": "Cosine of solar zenith angle integrated",
        "standard_name": None,
        "cf_units": None,
    },
    "dewpoint": {
        "variable": "td",
        "units": "K",
        "long_name": "Dewpoint",
        "standard_name": "dew_point_temperature",
        "cf_units": "K",
    },
    "eqpt": {
        "variable": "eqpt",
        "units": "K",
        "paramId": 4,
        "edition": ("1", "2"),
        "long_name": "Equivalent potential temperature",
        "standard_name": "air_equivalent_potential_temperature",
        "cf_units": "K",
    },
    "geometric_height_above_ground": {
        "variable": "h",
        "units": "m",
        "paramId": 3008,
        "edition": ("1", "2"),
        "long_name": "Geometric height above ground",
        "standard_name": "height",
        "cf_units": "m",
    },
    "geometric_height_above_sea": {
        "variable": "h",
        "units": "m",
        "long_name": "Geometric height above sea level",
        "standard_name": "altitude",
        "cf_units": "m",
    },
    "geometric_vertical_velocity": {
        "variable": "wz",
        "units": "m/s",
        "paramId": 260238,
        "edition": ("2",),
        "long_name": "Geometric vertical velocity",
        "standard_name": "upward_air_velocity",
        "cf_units": "m s-1",
    },
    "geopotential": {
        "variable": "z",
        "units": "m2/s2",
        "paramId": 129,
        "edition": ("1", "2"),
        "long_name": "Geopotential",
        "standard_name": "geopotential",
        "cf_units": "m2 s-2",
    },
    "geopotential_height": {
        "variable": "gh",
        "units": "gpm",
        "paramId": 156,
        "edition": ("1", "2"),
        "long_name": "Geopotential height",
        "standard_name": "geopotential_height",
        "cf_units": "m",
    },
    "hybrid_alpha": {
        "variable": "hybrid_alpha",
        "units": "1",
        "long_name": "Hybrid alpha",
        "standard_name": None,
        "cf_units": None,
    },
    "hybrid_delta": {
        "variable": "hybrid_delta",
        "units": "1",
        "long_name": "Hybrid delta",
        "standard_name": None,
        "cf_units": None,
    },
    "lcl_pressure": {
        "variable": "p_lcl",
        "units": "Pa",
        "long_name": "Pressure at Lifting condensation level (LCL)",
        "standard_name": None,
        "cf_units": None,
    },
    "lcl_temperature": {
        "variable": "t_lcl",
        "units": "K",
        "long_name": "Temperature at Lifting condensation level (LCL)",
        "standard_name": None,
        "cf_units": None,
    },
    "mixing_ratio": {
        "variable": "mass_mixrat",
        "units": "kg/kg",
        "paramId": 402000,
        "edition": ("2",),
        "long_name": "Mixing ratio",
        "standard_name": "humidity_mixing_ratio",
        "cf_units": "1",
    },
    "potential_temperature": {
        "variable": "pt",
        "units": "K",
        "paramId": 3,
        "edition": ("1", "2"),
        "long_name": "Potential temperature",
        "standard_name": "air_potential_temperature",
        "cf_units": "K",
    },
    "pressure": {
        "variable": "pres",
        "units": "Pa",
        "paramId": 54,
        "edition": ("1", "2"),
        "long_name": "Pressure",
        "standard_name": "air_pressure",
        "cf_units": "Pa",
    },
    "pressure_full_level": {
        "variable": "pres",
        "units": "Pa",
        "long_name": "Pressure at full level",
        "standard_name": "air_pressure",
        "cf_units": "Pa",
    },
    "pressure_half_level": {
        "variable": "pres_half",
        "units": "Pa",
        "long_name": "Pressure at half level",
        "standard_name": "air_pressure",
        "cf_units": "Pa",
    },
    "pressure_on_dry_adiabat": {
        "variable": "pres_dry_adiabat",
        "units": "Pa",
        "long_name": "Pressure on dry adiabat",
        "standard_name": None,
        "cf_units": "Pa",
    },
    "pressure_on_moist_adiabat": {
        "variable": "pres_moist_adiabat",
        "units": "Pa",
        "long_name": "Pressure on moist adiabat",
        "standard_name": None,
        "cf_units": "Pa",
    },
    "relative_geopotential_thickness": {
        "variable": "rel_geopot_thick",
        "units": "m2/s2",
        "long_name": "Relative geopotential thickness",
        "standard_name": None,
        "cf_units": None,
    },
    "relative_humidity": {
        "variable": "r",
        "units": "%",
        "paramId": 157,
        "edition": ("1", "2"),
        "long_name": "Relative humidity",
        "standard_name": "relative_humidity",
        "cf_units": "1",
    },
    "saturation_mixing_ratio": {
        "variable": "ws",
        "units": "kg/kg",
        "long_name": "Saturation mixing ratio",
        "standard_name": None,
        "cf_units": None,
    },
    "saturation_mixing_ratio_slope": {
        "variable": "ws_slope",
        "units": "kg/kg/K",
        "long_name": "Derivative of saturation mixing ratio with respect to temperature",
        "standard_name": None,
        "cf_units": None,
    },
    "saturation_specific_humidity": {
        "variable": "sqw",
        "units": "kg/kg",
        "long_name": "Saturation specific humidity",
        "standard_name": None,
        "cf_units": None,
    },
    "saturation_specific_humidity_slope": {
        "variable": "sqw_slope",
        "units": "kg/kg/K",
        "long_name": "Derivative of saturation specific humidity with respect to temperature",
        "standard_name": None,
        "cf_units": None,
    },
    "saturation_vapour_pressure": {
        "variable": "swvp",
        "units": "Pa",
        "paramId": 261021,
        "edition": ("2"),
        "long_name": "Saturation vapour pressure",
        "standard_name": None,
        "cf_units": None,
    },
    "saturation_vapour_pressure_slope": {
        "variable": "swvp_slope",
        "units": "Pa/K",
        "long_name": "Derivative of saturation vapour pressure with respect to temperature",
        "standard_name": None,
        "cf_units": None,
    },
    "sept": {
        "variable": "sept",
        "units": "K",
        "paramId": 5,
        "edition": ("1", "2"),
        "long_name": "Pseudo-equivalent potential temperature",
        "standard_name": "air_pseudo_equivalent_potential_temperature",
        "cf_units": "K",
    },
    "specific_gas_constant": {
        "variable": "R",
        "units": "J/kg/K",
        "long_name": "Specific gas constant of moist air",
        "standard_name": None,
        "cf_units": None,
    },
    "specific_humidity": {
        "variable": "q",
        "units": "kg/kg",
        "paramId": 133,
        "edition": ("1", "2"),
        "long_name": "Specific humidity",
        "standard_name": "specific_humidity",
        "cf_units": "1",
    },
    "temperature": {
        "variable": "t",
        "units": "K",
        "paramId": 130,
        "edition": ("1", "2"),
        "long_name": "Temperature",
        "standard_name": "air_temperature",
        "cf_units": "K",
    },
    "temperature_on_dry_adiabat": {
        "variable": "t_dry_adiabat",
        "units": "K",
        "long_name": "Temperature on dry adiabat",
        "standard_name": None,
        "cf_units": "K",
    },
    "temperature_on_moist_adiabat": {
        "variable": "t_moist_adiabat",
        "units": "K",
        "long_name": "Temperature on moist adiabat",
        "standard_name": None,
        "cf_units": "K",
    },
    "toa_incident_solar_radiation": {
        "variable": "toa_incident_solar_radiation",
        "units": "W/m2",
        "long_name": "Top of atmosphere incident solar radiation",
        "standard_name": "toa_incoming_shortwave_flux",
        "cf_units": "W m-2",
    },
    "vapour_pressure": {
        "variable": "vapp",
        "units": "Pa",
        "paramId": 260008,
        "edition": ("2"),
        "long_name": "Vapour pressure",
        "standard_name": "water_vapor_partial_pressure_in_air",
        "cf_units": "Pa",
    },
    "virtual_potential_temperature": {
        "variable": "vptmp",
        "units": "K",
        "paramId": 3012,
        "edition": ("2"),
        "long_name": "Virtual potential temperature",
        "standard_name": None,
        "cf_units": None,
    },
    "virtual_temperature": {
        "variable": "vtmp",
        "units": "K",
        "paramId": 300012,
        "edition": ("1", "2"),
        "long_name": "Virtual temperature",
        "standard_name": "virtual_temperature",
        "cf_units": "K",
    },
    "wet_bulb_potential_temperature": {
        "variable": "wbpt",
        "units": "K",
        "paramId": 261022,
        "edition": ("2"),
        "long_name": "Wet bulb potential temperature",
        "standard_name": "wet_bulb_potential_temperature",
        "cf_units": "K",
    },
    "wet_bulb_temperature": {
        "variable": "wbgt",
        "units": "K",
        "paramId": 261014,
        "edition": ("2"),
        "long_name": "Wet bulb temperature",
        "standard_name": "wet_bulb_temperature",
        "cf_units": "K",
    },
    "wind_direction": {
        "variable": "wdir",
        "units": "degrees",
        "paramId": 3031,
        "edition": ("1", "2"),
        "long_name": "Wind direction",
        "standard_name": "wind_from_direction",
        "cf_units": "degree",
    },
    "wind_speed": {
        "variable": "ws",
        "units": "m/s",
        "paramId": 10,
        "edition": ("1", "2"),
        "long_name": "Wind speed",
        "standard_name": "wind_speed",
        "cf_units": "m s-1",
    },
}


class _ParamDB:
    def __init__(self):
        self.params = _PARAMS

    def get(self, name):
        return self.params[name]

    def field_parameter_metadata(self, name):
        param_item = self.get(name)
        return {
            "parameter.variable": param_item["variable"],
            "parameter.units": param_item["units"],
            "parameter.standard_name": param_item["standard_name"],
        }

    def xarray_variable_and_metadata(self, name):
        param_item = self.get(name)
        return param_item["variable"], {
            "long_name": param_item["long_name"],
            "units": param_item["cf_units"] if param_item["cf_units"] is not None else param_item["units"],
            "standard_name": param_item["standard_name"],
        }


PARAMS = _ParamDB()
