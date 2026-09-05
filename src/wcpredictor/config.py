"""Paths, constants and feature definitions shared across the project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_RESULTS_CSV = DATA_DIR / "results.csv"
MODEL_BUNDLE_PATH = MODELS_DIR / "model_bundle.joblib"

# Time-ordered split. Everything before TRAIN_END trains the models, the window
# up to TEST_START selects between them, and TEST_START onwards is touched once.
TRAIN_END = "2016-01-01"
TEST_START = "2019-01-01"

# Outcome labels from the home team's perspective, ordered worst to best so that
# predict_proba columns read left to right as away win / draw / home win.
CLASSES: tuple[str, str, str] = ("lose", "draw", "win")
CLASS_TO_INDEX = {label: i for i, label in enumerate(CLASSES)}

# Rolling windows used for recent-form features.
FORM_WINDOWS: tuple[int, ...] = (3, 5, 10)
GOAL_DIFF_WINDOW = 5

# Finals of the top-tier world and continental competitions (qualifiers excluded).
MAJOR_TOURNAMENTS: frozenset[str] = frozenset(
    {
        "FIFA World Cup",
        "UEFA Euro",
        "Copa América",
        "AFC Asian Cup",
        "African Cup of Nations",
    }
)

# Per-team features computed on the long "one row per team per match" view.
TEAM_FEATURES: tuple[str, ...] = (
    "pre_win_rate",
    "pre_draw_rate",
    "last3_win_rate",
    "last5_win_rate",
    "last10_win_rate",
    "last3_draw_rate",
    "last5_draw_rate",
    "last10_draw_rate",
    "cum_goal_diff",
    "last5_goal_diff",
)

# Model input. Kept deliberately wide: the elastic-net penalty decides which of
# the overlapping ELO and form features actually carry signal.
FEATURE_COLUMNS: tuple[str, ...] = (
    "home_pre_win_rate",
    "home_pre_draw_rate",
    "away_pre_win_rate",
    "away_pre_draw_rate",
    "elo_diff",
    "home_elo",
    "away_elo",
    "neutral_int",
    "is_friendly",
    "is_qualification",
    "is_major",
    "home_cum_goal_diff",
    "away_cum_goal_diff",
    "home_last5_goal_diff",
    "away_last5_goal_diff",
    "home_last5_win_rate",
    "away_last5_win_rate",
    "home_last10_win_rate",
    "away_last10_win_rate",
)


@dataclass(frozen=True)
class EloConfig:
    """Parameters of the from-scratch ELO implementation."""

    initial: float = 1500.0
    k: float = 30.0
    home_advantage: float = 65.0
    scale: float = 400.0


DEFAULT_ELO = EloConfig()
