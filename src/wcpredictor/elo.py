"""From-scratch ELO ratings for international teams."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DEFAULT_ELO, EloConfig


@dataclass
class EloResult:
    """Pre-match ratings for every match, plus the final rating of every team."""

    home_elo: np.ndarray
    away_elo: np.ndarray
    ratings: dict[str, float]


def expected_score(home_elo: float, away_elo: float, home_advantage: float, scale: float) -> float:
    """Probability-like expectation that the home team takes the points."""
    return 1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + home_advantage)) / scale))


def compute_elo(matches: pd.DataFrame, config: EloConfig = DEFAULT_ELO) -> EloResult:
    """Walk matches in chronological order and update ratings after each one.

    The rating attached to a match is the one held *before* it was played, so the
    resulting columns are safe to use as features.
    """
    if not matches["date"].is_monotonic_increasing:
        raise ValueError("compute_elo expects matches sorted by date.")

    ratings: dict[str, float] = {}
    n = len(matches)
    home_elos = np.empty(n, dtype=float)
    away_elos = np.empty(n, dtype=float)

    actual_by_result = {"win": 1.0, "draw": 0.5, "lose": 0.0}

    home_teams = matches["home_team"].to_numpy()
    away_teams = matches["away_team"].to_numpy()
    results = matches["result"].to_numpy()
    neutrals = matches["neutral"].to_numpy()

    for i in range(n):
        home, away = home_teams[i], away_teams[i]
        home_elo = ratings.get(home, config.initial)
        away_elo = ratings.get(away, config.initial)
        home_elos[i] = home_elo
        away_elos[i] = away_elo

        advantage = 0.0 if neutrals[i] else config.home_advantage
        expected_home = expected_score(home_elo, away_elo, advantage, config.scale)
        actual_home = actual_by_result[results[i]]
        adjustment = config.k * (actual_home - expected_home)

        ratings[home] = home_elo + adjustment
        ratings[away] = away_elo - adjustment

    return EloResult(home_elo=home_elos, away_elo=away_elos, ratings=ratings)


def add_elo_features(
    matches: pd.DataFrame,
    config: EloConfig = DEFAULT_ELO,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Attach `home_elo`, `away_elo` and `elo_diff`; return the final ratings too."""
    elo = compute_elo(matches, config)
    out = matches.copy()
    out["home_elo"] = elo.home_elo
    out["away_elo"] = elo.away_elo
    out["elo_diff"] = out["home_elo"] - out["away_elo"]
    return out, elo.ratings
