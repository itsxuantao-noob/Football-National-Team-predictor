from __future__ import annotations

import pandas as pd
import pytest

from wcpredictor.config import EloConfig
from wcpredictor.elo import add_elo_features, compute_elo, expected_score


def build_matches(rows: list[tuple[str, str, str, bool]]) -> pd.DataFrame:
    """Minimal match table: (home, away, result, neutral) in chronological order."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2000-01-01", periods=len(rows), freq="D"),
            "home_team": [r[0] for r in rows],
            "away_team": [r[1] for r in rows],
            "result": [r[2] for r in rows],
            "neutral": [r[3] for r in rows],
        }
    )


def test_expected_score_is_even_between_equals_on_neutral_ground():
    assert expected_score(1500, 1500, home_advantage=0, scale=400) == pytest.approx(0.5)


def test_home_advantage_raises_the_expectation():
    neutral = expected_score(1500, 1500, home_advantage=0, scale=400)
    at_home = expected_score(1500, 1500, home_advantage=65, scale=400)
    assert at_home > neutral


def test_draw_between_equals_on_neutral_ground_changes_nothing():
    result = compute_elo(build_matches([("A", "B", "draw", True)]))
    assert result.ratings["A"] == pytest.approx(1500)
    assert result.ratings["B"] == pytest.approx(1500)


def test_a_win_moves_rating_up_and_the_opponent_down_by_the_same_amount():
    result = compute_elo(build_matches([("A", "B", "win", True)]))
    assert result.ratings["A"] > 1500 > result.ratings["B"]
    assert result.ratings["A"] - 1500 == pytest.approx(1500 - result.ratings["B"])


def test_ratings_are_zero_sum_across_a_season():
    rows = [
        ("A", "B", "win", True),
        ("B", "C", "draw", False),
        ("C", "A", "lose", True),
        ("A", "C", "win", False),
    ]
    result = compute_elo(build_matches(rows))
    assert sum(result.ratings.values()) == pytest.approx(1500 * len(result.ratings))


def test_beating_a_stronger_opponent_earns_more_than_beating_a_weaker_one():
    strong_first = compute_elo(build_matches([("A", "B", "win", True), ("A", "B", "win", True)]))
    # The second win against a now-weaker B is worth less than the first.
    history = compute_elo(build_matches([("A", "B", "win", True)]))
    first_gain = history.ratings["A"] - 1500
    second_gain = strong_first.ratings["A"] - history.ratings["A"]
    assert second_gain < first_gain


def test_recorded_ratings_are_pre_match():
    matches = build_matches([("A", "B", "win", True), ("A", "B", "win", True)])
    result = compute_elo(matches)
    assert result.home_elo[0] == 1500
    assert result.home_elo[1] > 1500
    assert result.home_elo[1] != result.ratings["A"]


def test_k_factor_scales_the_update():
    small = compute_elo(build_matches([("A", "B", "win", True)]), EloConfig(k=10))
    large = compute_elo(build_matches([("A", "B", "win", True)]), EloConfig(k=40))
    assert (large.ratings["A"] - 1500) == pytest.approx(4 * (small.ratings["A"] - 1500))


def test_compute_elo_requires_sorted_input():
    matches = build_matches([("A", "B", "win", True), ("A", "B", "win", True)])
    with pytest.raises(ValueError):
        compute_elo(matches.iloc[::-1])


def test_add_elo_features_adds_the_three_columns(matches):
    with_elo, ratings = add_elo_features(matches)
    assert {"home_elo", "away_elo", "elo_diff"} <= set(with_elo.columns)
    assert (with_elo["elo_diff"] == with_elo["home_elo"] - with_elo["away_elo"]).all()
    assert set(ratings) == set(matches["home_team"]) | set(matches["away_team"])
