"""Compatibility entrypoint for the relocated core verification suite."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))
from test_labeling import (
    LabelTests as LabelTests,
    CoreContractTests as CoreContractTests,
)

if __name__ == "__main__":
    unittest.main()

__all__ = ["LabelTests", "CoreContractTests"]
