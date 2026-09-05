from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wcpredictor.config import FEATURE_COLUMNS
from wcpredictor.data import load_matches
from wcpredictor.features import (
    add_form_features,
    build_current_team_state,
    build_team_view,
    drop_incomplete,
    widen_team_features,
)


def test_team_view_has_two_rows_per_match(matches):
    team_view = build_team_view(matches)
    assert len(team_view) == 2 * len(matches)
    assert team_view.groupby("match_id").size().eq(2).all()


def test_team_view_flips_the_result_for_the_away_side(matches):
    team_view = build_team_view(matches)
    home = team_view[team_view["is_home"] == 1].set_index("match_id")["result"]
    away = team_view[team_view["is_home"] == 0].set_index("match_id")["result"]
    paired = pd.DataFrame({"home": home, "away": away})
    assert (paired[paired["home"] == "win"]["away"] == "lose").all()
    assert (paired[paired["home"] == "draw"]["away"] == "draw").all()


def test_form_features_use_only_earlier_matches(matches):
    """Recompute one team's features by hand and compare against the pipeline."""
    with_form = add_form_features(build_team_view(matches))
    team = with_form["team"].value_counts().index[0]
    history = with_form[with_form["team"] == team].sort_values("match_id").reset_index(drop=True)

    wins = history["is_win"].to_numpy()
    goal_diff = history["goal_difference"].to_numpy()

    assert np.isnan(history.loc[0, "pre_win_rate"])
    for i in (1, 5, 25, len(history) - 1):
        assert history.loc[i, "pre_win_rate"] == pytest.approx(wins[:i].mean())
        assert history.loc[i, "cum_goal_diff"] == pytest.approx(goal_diff[:i].mean())

    assert np.isnan(history.loc[4, "last5_win_rate"])
    for i in (5, 30, len(history) - 1):
        assert history.loc[i, "last5_win_rate"] == pytest.approx(wins[i - 5 : i].mean())
    for i in (10, 30, len(history) - 1):
        assert history.loc[i, "last10_win_rate"] == pytest.approx(wins[i - 10 : i].mean())


def test_widening_survives_two_matches_on_the_same_day(tmp_path):
    """Regression test: joining on (date, team) used to explode these into a Cartesian product."""
    same_day = pd.DataFrame(
        {
            "date": ["1916-07-02", "1916-07-02", "1916-07-05"],
            "home_team": ["Argentina", "Argentina", "Chile"],
            "away_team": ["Chile", "Brazil", "Brazil"],
            "home_score": [1, 2, 0],
            "away_score": [0, 1, 1],
            "tournament": ["Copa América"] * 3,
            "city": ["Buenos Aires"] * 3,
            "country": ["Argentina"] * 3,
            "neutral": [False, False, True],
        }
    )
    path = tmp_path / "results.csv"
    same_day.to_csv(path, index=False)

    loaded = load_matches(path)
    team_view = add_form_features(build_team_view(loaded))
    wide = widen_team_features(team_view)

    assert len(wide) == len(loaded) == 3
    assert wide.index.is_unique


def test_feature_table_keeps_one_row_per_match_and_has_every_model_column(feature_table):
    table, _, _ = feature_table
    assert table["match_id"].is_unique
    assert set(FEATURE_COLUMNS) <= set(table.columns)


def test_drop_incomplete_removes_only_early_history(feature_table):
    table, _, _ = feature_table
    clean = drop_incomplete(table, FEATURE_COLUMNS)
    assert not clean[list(FEATURE_COLUMNS)].isna().any().any()
    assert 0 < len(clean) < len(table)
    # The rows that survive are the later ones, once every rolling window is filled.
    assert clean["date"].min() >= table["date"].min()


def test_current_state_includes_the_most_recent_match(matches):
    """The prediction-time state must not be one match stale, unlike the training features."""
    team_view = build_team_view(matches)
    with_form = add_form_features(team_view)
    state = build_current_team_state(team_view)

    team = state.index[0]
    history = team_view[team_view["team"] == team].sort_values("match_id")
    stale = with_form[with_form["team"] == team].sort_values("match_id").iloc[-1]

    assert state.at[team, "pre_win_rate"] == pytest.approx(history["is_win"].mean())
    assert state.at[team, "matches_played"] == len(history)
    assert state.at[team, "last5_goal_diff"] == pytest.approx(history["goal_difference"].tail(5).mean())
    # Excluding vs including the final result gives different numbers.
    assert state.at[team, "pre_win_rate"] != pytest.approx(stale["pre_win_rate"])


def test_tournament_flags_are_mutually_exclusive(feature_table):
    table, _, _ = feature_table
    flags = table[["is_friendly", "is_qualification", "is_major"]]
    assert flags.sum(axis=1).max() <= 1
    assert flags.to_numpy().max() == 1


def test_build_feature_table_returns_state_for_every_team(feature_table, matches):
    _, elo_ratings, team_state = feature_table
    expected = set(matches["home_team"]) | set(matches["away_team"])
    assert set(team_state.index) == expected
    assert set(elo_ratings) == expected
