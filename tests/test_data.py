from __future__ import annotations

import pandas as pd
import pytest

from wcpredictor.data import (
    MissingDataError,
    encode_labels,
    label_results,
    load_fixtures,
    load_raw,
    split_played_and_fixtures,
)


def test_load_raw_rejects_missing_file(tmp_path):
    with pytest.raises(MissingDataError):
        load_raw(tmp_path / "nope.csv")


def test_played_and_fixtures_are_separated(results_csv):
    played, fixtures = split_played_and_fixtures(load_raw(results_csv))
    assert played["home_score"].notna().all()
    assert len(fixtures) == 1
    assert fixtures["home_score"].isna().all()


def test_load_matches_is_chronological_and_keyed(matches):
    assert matches["date"].is_monotonic_increasing
    assert matches["match_id"].is_unique
    assert list(matches["match_id"]) == list(range(len(matches)))


def test_result_labels_match_the_scoreline(matches):
    wins = matches[matches["result"] == "win"]
    draws = matches[matches["result"] == "draw"]
    losses = matches[matches["result"] == "lose"]

    assert (wins["home_score"] > wins["away_score"]).all()
    assert (draws["home_score"] == draws["away_score"]).all()
    assert (losses["home_score"] < losses["away_score"]).all()
    assert len(wins) + len(draws) + len(losses) == len(matches)


def test_label_results_raises_on_missing_scores():
    frame = pd.DataFrame({"home_score": [1, None], "away_score": [0, 2]})
    with pytest.raises(ValueError):
        label_results(frame)


def test_encode_labels_uses_the_configured_class_order():
    encoded = encode_labels(pd.Series(["lose", "draw", "win"]))
    assert list(encoded) == [0, 1, 2]

    with pytest.raises(ValueError):
        encode_labels(pd.Series(["win", "abandoned"]))


def test_load_fixtures_returns_unplayed_matches(results_csv):
    fixtures = load_fixtures(results_csv)
    assert len(fixtures) == 1
    assert fixtures.loc[0, "home_team"] == "Alphaland"
