"""Unit conversion -- deterministic, hand-verified conversion factors, not
an LLM guessing at them. Covers the unit families most relevant to
refinery/industrial engineering work: pressure, temperature, volume
(including the petroleum barrel, "bbl" -- the standard unit refineries
actually measure crude and product volumes in), length, and mass.
"""

from __future__ import annotations

from typing import Optional


class UnitError(Exception):
    pass


# Each table maps a unit's short code -> its value in ONE base unit for that
# family. Converting is then just (value * from_factor) / to_factor.
_PRESSURE_TO_PA = {
    "pa": 1.0, "kpa": 1_000.0, "mpa": 1_000_000.0,
    "bar": 100_000.0, "psi": 6_894.757293168, "atm": 101_325.0,
}
_LENGTH_TO_M = {
    "mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1_000.0,
    "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1_609.344,
}
_MASS_TO_KG = {
    "mg": 1e-6, "g": 0.001, "kg": 1.0, "ton": 1_000.0, "tonne": 1_000.0,
    "lb": 0.45359237, "oz": 0.028349523125,
}
# "bbl" is the petroleum barrel (42 US gallons), the standard refinery
# volume unit -- distinct from a generic "drum" and worth getting exactly
# right rather than leaving to an LLM's memory of the conversion factor.
_VOLUME_TO_L = {
    "ml": 0.001, "l": 1.0, "m3": 1_000.0,
    "gal": 3.785411784, "bbl": 158.987294928,
}

_FAMILIES = (_PRESSURE_TO_PA, _LENGTH_TO_M, _MASS_TO_KG, _VOLUME_TO_L)
_TEMPERATURE_UNITS = {"c", "f", "k"}


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    if from_unit == "c":
        celsius = value
    elif from_unit == "f":
        celsius = (value - 32) * 5 / 9
    elif from_unit == "k":
        celsius = value - 273.15
    else:
        raise UnitError(f"Unknown temperature unit '{from_unit}'.")

    if to_unit == "c":
        return celsius
    if to_unit == "f":
        return celsius * 9 / 5 + 32
    if to_unit == "k":
        return celsius + 273.15
    raise UnitError(f"Unknown temperature unit '{to_unit}'.")


def _find_family(unit: str) -> Optional[dict]:
    for table in _FAMILIES:
        if unit in table:
            return table
    return None


def convert_unit(value: float, from_unit: str, to_unit: str) -> float:
    """Convert `value` from `from_unit` to `to_unit`. Raises UnitError
    (never a raw exception) if either unit is unrecognized or the two units
    belong to different families (e.g. converting a pressure to a length)."""
    f = from_unit.strip().lower()
    t = to_unit.strip().lower()

    if f in _TEMPERATURE_UNITS or t in _TEMPERATURE_UNITS:
        if f not in _TEMPERATURE_UNITS or t not in _TEMPERATURE_UNITS:
            raise UnitError(f"Cannot convert between '{from_unit}' and '{to_unit}' -- mismatched unit types.")
        return _convert_temperature(value, f, t)

    from_family = _find_family(f)
    to_family = _find_family(t)
    if from_family is None:
        raise UnitError(f"Unrecognized unit '{from_unit}'.")
    if to_family is None:
        raise UnitError(f"Unrecognized unit '{to_unit}'.")
    if from_family is not to_family:
        raise UnitError(f"Cannot convert between '{from_unit}' and '{to_unit}' -- they measure different things.")

    return value * from_family[f] / to_family[t]
