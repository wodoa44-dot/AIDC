"""Compatibility imports; implementation lives in data.py."""
from data import (
    history_selection as history_selection,
    prepare as prepare,
    continuous as continuous,
    PREPARED as PREPARED,
    GROUPS as GROUPS,
    SELECTION as SELECTION,
)

__all__ = [
    "history_selection",
    "prepare",
    "continuous",
    "PREPARED",
    "GROUPS",
    "SELECTION",
]
