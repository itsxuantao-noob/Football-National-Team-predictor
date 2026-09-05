"""Leakage-safe feature engineering.

Every feature attached to a match is computed from matches played strictly
before it. The long "team view" (one row per team per match) does the work: form
statistics are shifted by one within each team, then folded back onto the match
table via `match_id`.
"""

from __future__ import annotations

import pandas as pd

from .config import (
    DEFAULT_ELO,
    FORM_WINDOWS,
    GOAL_DIFF_WINDOW,
    MAJOR_TOURNAMENTS,
    TEAM_FEATURES,
    EloConfig,
)
from .elo import add_elo_features

_FLIP_RESULT = {"win": "lose", "lose": "win", "draw": "draw"}


def build_team_view(matches: pd.DataFrame) -> pd.DataFrame:
    """Reshape the match table into one row per team per match.

    Each match contributes two rows, one from each team's perspective, so a
    single `groupby("team")` can produce form features for home and away sides
    at once.
    """
    columns = ["match_id", "date", "team", "opponent", "result", "goals_for", "goals_against"]

    home = matches[
        ["match_id", "date", "home_team", "away_team", "result", "home_score", "away_score"]
    ].copy()
    home.columns = columns
    home["is_home"] = 1

    away = matches[
        ["match_id", "date", "away_team", "home_team", "result", "away_score", "home_score"]
    ].copy()
    away.columns = columns
    away["is_home"] = 0
    away["result"] = away["result"].map(_FLIP_RESULT)

    team_view = pd.concat([home, away], ignore_index=True)
    team_view["goal_difference"] = team_view["goals_for"] - team_view["goals_against"]
    team_view["is_win"] = (team_view["result"] == "win").astype(int)
    team_view["is_draw"] = (team_view["result"] == "draw").astype(int)
    return team_view.sort_values(["match_id", "is_home"], kind="stable").reset_index(drop=True)


def add_form_features(team_view: pd.DataFrame) -> pd.DataFrame:
    """Add career-to-date and recent-form statistics, shifted to exclude the current match."""
    out = team_view.sort_values("match_id", kind="stable").copy()
    grouped = out.groupby("team", sort=False)

    out["pre_win_rate"] = grouped["is_win"].transform(lambda s: s.shift().expanding().mean())
    out["pre_draw_rate"] = grouped["is_draw"].transform(lambda s: s.shift().expanding().mean())
    out["cum_goal_diff"] = grouped["goal_difference"].transform(lambda s: s.shift().expanding().mean())

    for window in FORM_WINDOWS:
        out[f"last{window}_win_rate"] = grouped["is_win"].transform(
            lambda s, w=window: s.shift().rolling(w).mean()
        )
        out[f"last{window}_draw_rate"] = grouped["is_draw"].transform(
            lambda s, w=window: s.shift().rolling(w).mean()
        )

    out[f"last{GOAL_DIFF_WINDOW}_goal_diff"] = grouped["goal_difference"].transform(
        lambda s: s.shift().rolling(GOAL_DIFF_WINDOW).mean()
    )
    return out


def widen_team_features(team_view: pd.DataFrame, columns: tuple[str, ...] = TEAM_FEATURES) -> pd.DataFrame:
    """Fold the two per-team rows of each match back into one `home_*`/`away_*` row.

    Joining on `match_id` rather than on (date, team) is what keeps this exact.
    An earlier version of this pipeline merged on (date, team) and silently
    produced a Cartesian blow-up for teams that played twice in one day.
    """
    home = team_view[team_view["is_home"] == 1].set_index("match_id")[list(columns)]
    away = team_view[team_view["is_home"] == 0].set_index("match_id")[list(columns)]
    home = home.add_prefix("home_")
    away = away.add_prefix("away_")

    wide = home.join(away, how="outer")
    if len(wide) != team_view["match_id"].nunique():
        raise AssertionError("Widening changed the number of matches; team view is malformed.")
    return wide


def add_tournament_flags(matches: pd.DataFrame) -> pd.DataFrame:
    """Encode competition importance as three independent binary flags."""
    out = matches.copy()
    tournament = out["tournament"].fillna("")
    out["is_friendly"] = (tournament == "Friendly").astype(int)
    out["is_qualification"] = tournament.str.contains("qualification", case=False).astype(int)
    out["is_major"] = tournament.isin(MAJOR_TOURNAMENTS).astype(int)
    return out


def build_current_team_state(team_view: pd.DataFrame) -> pd.DataFrame:
    """Per-team statistics *including* every match played, indexed by team name.

    These are the values a team carries into its next fixture, so this is what
    prediction uses. It differs from the last row of `add_form_features`, which
    deliberately excludes the most recent result.
    """
    ordered = team_view.sort_values("match_id", kind="stable")
    grouped = ordered.groupby("team", sort=True)

    state = pd.DataFrame(
        {
            "pre_win_rate": grouped["is_win"].mean(),
            "pre_draw_rate": grouped["is_draw"].mean(),
            "cum_goal_diff": grouped["goal_difference"].mean(),
            "matches_played": grouped.size(),
            "last_match_date": grouped["date"].max(),
        }
    )

    for window in FORM_WINDOWS:
        state[f"last{window}_win_rate"] = grouped["is_win"].apply(lambda s, w=window: s.tail(w).mean())
        state[f"last{window}_draw_rate"] = grouped["is_draw"].apply(lambda s, w=window: s.tail(w).mean())
    state[f"last{GOAL_DIFF_WINDOW}_goal_diff"] = grouped["goal_difference"].apply(
        lambda s: s.tail(GOAL_DIFF_WINDOW).mean()
    )
    return state


def build_feature_table(
    matches: pd.DataFrame,
    elo_config: EloConfig = DEFAULT_ELO,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    """Run the whole pipeline: ELO, form features, tournament flags.

    Returns the modelling table (rows with any missing feature dropped), the
    final ELO ratings, and the current per-team state used for prediction.
    """
    with_elo, elo_ratings = add_elo_features(matches, elo_config)
    with_elo = add_tournament_flags(with_elo)
    with_elo["neutral_int"] = with_elo["neutral"].astype(int)

    team_view = add_form_features(build_team_view(matches))
    wide_team = widen_team_features(team_view)

    table = with_elo.set_index("match_id").join(wide_team, how="left")
    table = table.reset_index()

    team_state = build_current_team_state(team_view)
    return table, elo_ratings, team_state


def drop_incomplete(table: pd.DataFrame, feature_columns) -> pd.DataFrame:
    """Drop early-history rows where form features are still undefined."""
    return table.dropna(subset=list(feature_columns)).reset_index(drop=True)
