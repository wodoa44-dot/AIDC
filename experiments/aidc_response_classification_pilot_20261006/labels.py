"""Compatibility imports; implementation lives in labeling.py."""
from labeling import (
    get as get,
    validate_spec as validate_spec,
    physical_check as physical_check,
    feasibility as feasibility,
    calculate_base_power_w as calculate_base_power_w,
    judge as judge,
)

__all__ = [
    "get",
    "validate_spec",
    "physical_check",
    "feasibility",
    "calculate_base_power_w",
    "judge",
]
