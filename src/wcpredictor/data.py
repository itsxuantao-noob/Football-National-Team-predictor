"""Loading and cleaning the raw international results dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLASSES, RAW_RESULTS_CSV

RAW_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
]


class MissingDataError(FileNotFoundError):
    """Raised when the Kaggle results file has not been downloaded yet."""


def load_raw(path: str | Path = RAW_RESULTS_CSV) -> pd.DataFrame:
    """Read `results.csv` and normalise dtypes."""
    path = Path(path)
    if not path.exists():
        raise MissingDataError(
            f"{path} not found. Download results.csv from "
            "https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017 "
            f"and place it in {path.parent}."
        )

    df = pd.read_csv(path)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = df["neutral"].astype(bool)
    for team_col in ("home_team", "away_team"):
        df[team_col] = df[team_col].str.strip()
    return df


def split_played_and_fixtures(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate finished matches from scheduled fixtures (which have no score)."""
    finished = df["home_score"].notna() & df["away_score"].notna()
    return df[finished].copy(), df[~finished].copy()


def label_results(played: pd.DataFrame) -> pd.DataFrame:
    """Add the `result` column: win / draw / lose from the home team's view."""
    played = played.copy()
    conditions = [
        played["home_score"] > played["away_score"],
        played["home_score"] == played["away_score"],
        played["home_score"] < played["away_score"],
    ]
    played["result"] = np.select(conditions, ["win", "draw", "lose"], default=None)
    if played["result"].isna().any():
        raise ValueError("Some matches could not be labelled win/draw/lose.")
    return played


def load_matches(path: str | Path = RAW_RESULTS_CSV) -> pd.DataFrame:
    """Return finished matches, chronologically ordered, labelled and keyed.

    `match_id` is a stable chronological key. Every later stage joins on it
    instead of on (date, team), which is not unique: a handful of teams played
    two matches on the same day in the early 1900s.
    """
    played, _ = split_played_and_fixtures(load_raw(path))
    played = label_results(played)
    played = played.sort_values(["date", "home_team", "away_team"], kind="stable")
    played = played.reset_index(drop=True)
    played.insert(0, "match_id", np.arange(len(played), dtype=np.int64))
    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)
    return played


def load_fixtures(path: str | Path = RAW_RESULTS_CSV) -> pd.DataFrame:
    """Return the scheduled, not-yet-played fixtures in the dataset."""
    _, fixtures = split_played_and_fixtures(load_raw(path))
    return fixtures.sort_values("date").reset_index(drop=True)


def encode_labels(results: pd.Series) -> np.ndarray:
    """Map win/draw/lose strings onto the integer class order in `CLASSES`."""
    lookup = {label: i for i, label in enumerate(CLASSES)}
    encoded = results.map(lookup)
    if encoded.isna().any():
        unknown = sorted(set(results[encoded.isna()]))
        raise ValueError(f"Unexpected result labels: {unknown}")
    return encoded.to_numpy(dtype=np.int64)
