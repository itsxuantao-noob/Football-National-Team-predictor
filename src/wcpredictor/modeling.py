"""Candidate models and their hyper-parameter grids.

Each candidate is a full `Pipeline`, so any preprocessing is refitted inside
every cross-validation fold rather than once on the whole training set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CLASSES


@dataclass
class ModelSpec:
    """A named estimator plus the grid to search over it."""

    name: str
    estimator: BaseEstimator
    param_grid: dict[str, Any] = field(default_factory=dict)
    needs_scaling: bool = False


def _logistic(fast: bool) -> ModelSpec:
    # Elastic net keeps the wide, partly redundant feature pool honest: l1_ratio
    # is only meaningful with penalty="elasticnet", which also forces the saga solver.
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=0.5,
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )
    grid = (
        {"clf__C": [0.01, 0.1, 1, 10, 100], "clf__l1_ratio": [0.0, 0.5, 1.0]}
        if fast
        else {"clf__C": [0.01, 0.1, 1, 10, 100], "clf__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0]}
    )
    return ModelSpec("logistic_regression", pipeline, grid, needs_scaling=True)


def _random_forest(fast: bool) -> ModelSpec:
    pipeline = Pipeline([("clf", RandomForestClassifier(random_state=42, n_jobs=-1))])
    grid = (
        {
            "clf__n_estimators": [300],
            "clf__max_depth": [6, 10, 15],
            "clf__min_samples_leaf": [20, 50],
        }
        if fast
        else {
            "clf__n_estimators": [200, 300, 500],
            "clf__max_depth": [4, 6, 8, 10, 15, 20],
            "clf__min_samples_leaf": [5, 10, 20, 50, 100],
        }
    )
    return ModelSpec("random_forest", pipeline, grid)


def _xgboost(fast: bool) -> ModelSpec | None:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None

    pipeline = Pipeline(
        [
            (
                "clf",
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=len(CLASSES),
                    tree_method="hist",
                    random_state=42,
                    n_jobs=-1,
                ),
            )
        ]
    )
    grid = (
        {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [3, 5],
            "clf__learning_rate": [0.05, 0.1],
        }
        if fast
        else {
            "clf__n_estimators": [200, 400, 600],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.01, 0.05, 0.1],
        }
    )
    return ModelSpec("xgboost", pipeline, grid)


def get_model_specs(fast: bool = True) -> list[ModelSpec]:
    """Return the candidate models, skipping XGBoost if it is not installed."""
    specs = [_logistic(fast), _random_forest(fast)]
    xgb_spec = _xgboost(fast)
    if xgb_spec is not None:
        specs.append(xgb_spec)
    return specs
