# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import numpy as np
import pint
import pytest

from earthkit.utils.units import _pintify
from earthkit.utils.units import are_equal
from earthkit.utils.units import convert


class TestPintify:
    """Tests for the _pintify function."""

    def test_none_input(self):
        """Test that None is converted to dimensionless."""
        result = _pintify(None)
        assert str(result) == "dimensionless"

    def test_alias_conversion(self):
        """Test that unit aliases are converted correctly."""
        result = _pintify("(0 - 1)")
        assert str(result) == "percent"

    def test_exponent_insertion(self):
        """Test that exponents are properly inserted."""
        result = _pintify("m s-1")
        assert "^" in str(result) or "/" in str(result)

    def test_negative_exponent(self):
        """Test handling of negative exponents."""
        result = _pintify("kg m-2 s-1")
        expected = _pintify("kg.m^-2.s^-1")
        assert result == expected

    def test_positive_exponent(self):
        """Test handling of positive exponents."""
        result = _pintify("m2")
        assert "^" in str(result) or "**" in str(result)

    def test_undefined_unit(self):
        """Test that undefined units return the string unchanged."""
        undefined_unit = "undefined_unit_xyz"
        result = _pintify(undefined_unit)
        assert result == undefined_unit

    def test_valid_unit_parsing(self):
        """Test that valid units are parsed correctly."""
        result = _pintify("m")
        assert str(result) == "meter"

    def test_complex_unit(self):
        """Test parsing of complex units."""
        result = _pintify("kg m2 s-2")
        assert result is not None


class TestAreEqual:
    """Tests for the are_equal function."""

    def test_identical_units(self):
        """Test that identical units are equal."""
        assert are_equal("m", "m") is True

    def test_equivalent_units(self):
        """Test that equivalent units are recognized."""
        assert are_equal("m", "meter") is True

    def test_different_units(self):
        """Test that different units are not equal."""
        assert are_equal("m", "s") is False

    def test_compound_units_equal(self):
        """Test equality of compound units."""
        assert are_equal("m s-1", "m.s^-1") is True

    def test_compound_units_different(self):
        """Test inequality of different compound units."""
        assert are_equal("m s-1", "m s-2") is False

    def test_dimensionless_units(self):
        """Test comparison of dimensionless units."""
        assert are_equal(None, "dimensionless") is True

    def test_alias_equality(self):
        """Test that aliases are recognized as equal."""
        assert are_equal("(0 - 1)", "percent") is True

    def test_temperature_units(self):
        """Test temperature unit comparisons."""
        assert are_equal("K", "kelvin") is True
        assert are_equal("K", "degC") is False

    def test_velocity_units(self):
        """Test velocity unit comparisons."""
        assert are_equal("m s-1", "m/s") is True

    def test_acceleration_units(self):
        """Test acceleration unit comparisons."""
        assert are_equal("m s-2", "m.s^-2") is True


class TestConvert:
    """Tests for the convert function."""

    def test_simple_conversion(self):
        """Test simple unit conversion."""
        data = np.array([1.0, 2.0, 3.0])
        result = convert(data, "m", "km")
        expected = np.array([0.001, 0.002, 0.003])
        np.testing.assert_array_almost_equal(result, expected)

    def test_temperature_conversion(self):
        """Test temperature conversion."""
        data = np.array([0.0, 100.0])
        result = convert(data, "degC", "K")
        expected = np.array([273.15, 373.15])
        np.testing.assert_array_almost_equal(result, expected)

    def test_velocity_conversion(self):
        """Test velocity conversion."""
        data = np.array([1.0])
        result = convert(data, "m s-1", "km h-1")
        expected = np.array([3.6])
        np.testing.assert_array_almost_equal(result, expected)

    def test_pressure_conversion(self):
        """Test pressure conversion."""
        data = np.array([1.0])
        result = convert(data, "Pa", "hPa")
        expected = np.array([0.01])
        np.testing.assert_array_almost_equal(result, expected)

    def test_no_conversion_needed(self):
        """Test conversion between identical units."""
        data = np.array([1.0, 2.0, 3.0])
        result = convert(data, "m", "m")
        np.testing.assert_array_equal(result, data)

    def test_compound_unit_conversion(self):
        """Test conversion of compound units."""
        data = np.array([1.0])
        result = convert(data, "kg m-2 s-1", "kg.m^-2.s^-1")
        np.testing.assert_array_almost_equal(result, data)

    def test_scalar_conversion(self):
        """Test conversion with scalar input."""
        data = 10.0
        result = convert(data, "m", "cm")
        assert np.isclose(result, 1000.0)

    def test_multidimensional_array(self):
        """Test conversion with multidimensional arrays."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = convert(data, "m", "km")
        expected = np.array([[0.001, 0.002], [0.003, 0.004]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_zero_values(self):
        """Test conversion with zero values."""
        data = np.array([0.0, 0.0])
        result = convert(data, "m", "km")
        expected = np.array([0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_negative_values(self):
        """Test conversion with negative values."""
        data = np.array([-10.0, -20.0])
        result = convert(data, "degC", "K")
        expected = np.array([263.15, 253.15])
        np.testing.assert_array_almost_equal(result, expected)

    def test_incompatible_units(self):
        """Test that incompatible unit conversion raises an error."""
        data = np.array([1.0])
        with pytest.raises(pint.errors.DimensionalityError):
            convert(data, "m", "s")

    def test_large_values(self):
        """Test conversion with large values."""
        data = np.array([1e6, 1e9])
        result = convert(data, "m", "km")
        expected = np.array([1e3, 1e6])
        np.testing.assert_array_almost_equal(result, expected)

    def test_small_values(self):
        """Test conversion with small values."""
        data = np.array([1e-6, 1e-9])
        result = convert(data, "m", "mm")
        expected = np.array([1e-3, 1e-6])
        np.testing.assert_array_almost_equal(result, expected)

    def test_alias_conversion(self):
        """Test conversion using aliased units."""
        data = np.array([50.0])
        result = convert(data, "(0 - 1)", "percent")
        np.testing.assert_array_almost_equal(result, data)
