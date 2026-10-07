"""Compatibility imports; implementation lives in training.py."""
from training import (
    fit_predict as fit_predict,
    train_models as train_models,
    assert_split as assert_split,
)
from evaluation import evaluate as evaluate, aggregate as aggregate

__all__ = ["fit_predict", "train_models", "assert_split", "evaluate", "aggregate"]
