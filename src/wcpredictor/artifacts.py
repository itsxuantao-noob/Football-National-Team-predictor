"""Reading and writing the trained model bundle.

Everything prediction needs travels in one file: the fitted estimator, the
feature order it expects, the final ELO table and each team's current form.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import joblib
import pandas as pd

from .config import MODEL_BUNDLE_PATH

BUNDLE_VERSION = 2


class ModelBundle(TypedDict):
    version: int
    model: Any
    feature_columns: list[str]
    classes: list[str]
    elo_ratings: dict[str, float]
    elo_config: Any
    team_state: pd.DataFrame
    metadata: dict[str, Any]


class BundleNotFoundError(FileNotFoundError):
    """Raised when no trained model has been saved yet."""


def save_bundle(bundle: ModelBundle, path: str | Path = MODEL_BUNDLE_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: str | Path = MODEL_BUNDLE_PATH) -> ModelBundle:
    path = Path(path)
    if not path.exists():
        raise BundleNotFoundError(f"No model bundle at {path}. Run `wcp train` first.")

    bundle = joblib.load(path)
    version = bundle.get("version")
    if version != BUNDLE_VERSION:
        raise ValueError(
            f"{path} was written by bundle version {version}, this code expects "
            f"{BUNDLE_VERSION}. Retrain with `wcp train`."
        )
    return bundle
