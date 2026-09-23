"""Unit tests for tools/calculator.py and tools/units.py -- pure,
deterministic functions, testable with no model or server dependency
(unlike the LLM-driven tool-selection path, which is verified live/manually
per this project's established methodology)."""

import pytest
from tools.calculator import CalculatorError, calculate
from tools.units import UnitError, convert_unit


def test_calculate_basic_arithmetic():
    assert calculate("2 + 2") == 4
    assert calculate("10 - 3") == 7
    assert calculate("4 * 5") == 20
    assert calculate("10 / 4") == 2.5
    assert calculate("10 // 4") == 2
    assert calculate("10 % 3") == 1
    assert calculate("2 ** 10") == 1024


def test_calculate_operator_precedence_and_parens():
    assert calculate("2 + 3 * 4") == 14
    assert calculate("(2 + 3) * 4") == 20


def test_calculate_unary_minus():
    assert calculate("-5 + 3") == -2


def test_calculate_division_by_zero_raises_clean_error():
    with pytest.raises(CalculatorError):
        calculate("1 / 0")


def test_calculate_rejects_non_arithmetic():
    """No code-injection surface: names, calls, attribute access all rejected."""
    for bad in ["__import__('os')", "open('x')", "x + 1", "[1,2,3]", "1; 2"]:
        with pytest.raises(CalculatorError):
            calculate(bad)


def test_calculate_rejects_malformed_expression():
    with pytest.raises(CalculatorError):
        calculate("2 +")


def test_convert_pressure_bar_to_psi():
    # 1 bar = 14.5037738 psi (standard conversion)
    result = convert_unit(1, "bar", "psi")
    assert result == pytest.approx(14.5037738, rel=1e-4)


def test_convert_volume_barrel_to_liters():
    # 1 petroleum barrel = 158.987294928 liters
    result = convert_unit(1, "bbl", "l")
    assert result == pytest.approx(158.987294928, rel=1e-6)


def test_convert_temperature_celsius_to_fahrenheit():
    assert convert_unit(0, "c", "f") == pytest.approx(32.0)
    assert convert_unit(100, "c", "f") == pytest.approx(212.0)


def test_convert_temperature_celsius_to_kelvin():
    assert convert_unit(0, "c", "k") == pytest.approx(273.15)


def test_convert_same_unit_is_identity():
    assert convert_unit(42, "bar", "bar") == 42


def test_convert_case_insensitive_and_whitespace_tolerant():
    assert convert_unit(1, " BAR ", " Psi ") == pytest.approx(14.5037738, rel=1e-4)


def test_convert_mismatched_families_raises():
    with pytest.raises(UnitError):
        convert_unit(1, "bar", "kg")


def test_convert_mismatched_temperature_and_other_family_raises():
    with pytest.raises(UnitError):
        convert_unit(1, "c", "bar")


def test_convert_unrecognized_unit_raises():
    with pytest.raises(UnitError):
        convert_unit(1, "flibbertigibbet", "bar")
