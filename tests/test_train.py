from __future__ import annotations

import json

import pandas as pd
import pytest

from wcpredictor.artifacts import BUNDLE_VERSION, load_bundle
from wcpredictor.config import CLASSES, FEATURE_COLUMNS
from wcpredictor.evaluate import multiclass_brier, score
from wcpredictor.features import drop_incomplete
from wcpredictor.train import CALIBRATION_METHODS, split_by_date


def test_splits_are_chronological_and_disjoint(feature_table):
    table = drop_incomplete(feature_table[0], FEATURE_COLUMNS)
    train, validation, test = split_by_date(table)

    assert len(train) + len(validation) + len(test) == len(table)
    assert len(train) > 0 and len(validation) > 0 and len(test) > 0

    dates = table["date"]
    train_dates = dates[dates < pd.Timestamp("2016-01-01")]
    assert len(train_dates) == len(train)
    assert list(train.X.columns) == list(FEATURE_COLUMNS)


def test_split_by_date_rejects_impossible_dates(feature_table):
    table = drop_incomplete(feature_table[0], FEATURE_COLUMNS)
    train, validation, _ = split_by_date(table, train_end="1900-01-01", test_start="1901-01-01")
    assert len(train) == 0 and len(validation) == 0


def test_score_rewards_confident_correct_predictions():
    import numpy as np

    y = np.array([2, 2, 2])
    confident = np.array([[0.05, 0.05, 0.90]] * 3)
    hedged = np.array([[0.33, 0.33, 0.34]] * 3)

    assert score(y, confident)["log_loss"] < score(y, hedged)["log_loss"]
    assert multiclass_brier(y, confident) < multiclass_brier(y, hedged)


def test_training_produces_a_complete_report(trained_bundle):
    _, report = trained_bundle

    assert report["selected_model"] == "logistic_regression"
    assert report["selected_calibration"] in CALIBRATION_METHODS
    assert set(report["calibration_search"]) == set(CALIBRATION_METHODS)
    assert report["dataset"]["n_matches"] > 0
    assert report["feature_importance"]
    assert set(report["feature_importance"]) == set(FEATURE_COLUMNS)


def test_the_model_beats_the_class_prior_baseline_on_held_out_data(trained_bundle):
    _, report = trained_bundle
    model = report["test"]["model"]
    prior = report["test"]["baselines"]["class_prior"]

    assert model["log_loss"] < prior["log_loss"]
    assert model["accuracy"] > report["test"]["baselines"]["majority_class"]["accuracy"]


def test_reliability_bins_cover_the_test_predictions(trained_bundle):
    _, report = trained_bundle
    rows = report["test"]["reliability_win"]
    assert rows
    assert sum(row["count"] for row in rows) == report["split"]["sizes"]["test"]
    for row in rows:
        assert 0.0 <= row["mean_predicted"] <= 1.0
        assert 0.0 <= row["observed_frequency"] <= 1.0


def test_bundle_round_trips_with_everything_prediction_needs(trained_bundle):
    bundle_path, _ = trained_bundle
    bundle = load_bundle(bundle_path)

    assert bundle["version"] == BUNDLE_VERSION
    assert bundle["feature_columns"] == list(FEATURE_COLUMNS)
    assert bundle["classes"] == list(CLASSES)
    assert bundle["elo_ratings"] and bundle["elo_config"]["initial"] == 1500.0
    assert not bundle["team_state"].empty


def test_reports_are_written_to_disk(trained_bundle):
    bundle_path, report = trained_bundle
    reports_dir = bundle_path.parent / "reports"

    metrics = json.loads((reports_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["selected_model"] == report["selected_model"]

    markdown = (reports_dir / "report.md").read_text(encoding="utf-8")
    assert "Held-out test" in markdown
    assert "logistic_regression" in markdown


def test_loading_a_missing_bundle_is_a_clear_error(tmp_path):
    from wcpredictor.artifacts import BundleNotFoundError

    with pytest.raises(BundleNotFoundError):
        load_bundle(tmp_path / "absent.joblib")
