"""End-to-end training: search, select, calibrate, test once, then ship.

The protocol is deliberately strict about what each slice of time is allowed to
influence:

  train (< 2016)          hyper-parameter search, via TimeSeriesSplit inside it
  validation (2016-2018)  choosing the model family and the calibration method
  test (>= 2019)          scored exactly once, at the very end

Only after the test score is recorded does the chosen configuration get refit on
the full history to produce the deployment artifact.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from .artifacts import BUNDLE_VERSION, ModelBundle, save_bundle
from .config import (
    CLASSES,
    DEFAULT_ELO,
    FEATURE_COLUMNS,
    MODEL_BUNDLE_PATH,
    RAW_RESULTS_CSV,
    REPORTS_DIR,
    TEST_START,
    TRAIN_END,
    EloConfig,
)
from .data import encode_labels, load_matches
from .evaluate import baseline_scores, plot_calibration, reliability_table, score
from .features import build_feature_table, drop_incomplete
from .modeling import get_model_specs

log = logging.getLogger(__name__)

CALIBRATION_METHODS = ("none", "isotonic", "sigmoid")


@dataclass
class Split:
    """One time slice of the modelling table."""

    name: str
    X: pd.DataFrame
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


def split_by_date(
    table: pd.DataFrame,
    feature_columns=FEATURE_COLUMNS,
    train_end: str = TRAIN_END,
    test_start: str = TEST_START,
) -> tuple[Split, Split, Split]:
    """Cut the table into chronological train / validation / test slices."""
    columns = list(feature_columns)
    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)

    def make(name: str, mask: pd.Series) -> Split:
        rows = table[mask]
        return Split(name, rows[columns], encode_labels(rows["result"]))

    return (
        make("train", table["date"] < train_end_ts),
        make("validation", (table["date"] >= train_end_ts) & (table["date"] < test_start_ts)),
        make("test", table["date"] >= test_start_ts),
    )


def _wrap_calibration(estimator, method: str, n_splits: int = 3):
    """Return the estimator as-is, or wrapped in a time-series calibrator."""
    if method == "none":
        return clone(estimator)
    return CalibratedClassifierCV(
        estimator=clone(estimator),
        method=method,
        cv=TimeSeriesSplit(n_splits=n_splits),
    )


def search_models(
    train: Split,
    fast: bool = True,
    n_splits: int = 5,
    n_jobs: int = -1,
) -> dict[str, dict[str, Any]]:
    """Grid-search every candidate on the training slice, scored by log loss."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results: dict[str, dict[str, Any]] = {}

    for spec in get_model_specs(fast=fast):
        log.info("Grid searching %s over %d rows", spec.name, len(train))
        grid = GridSearchCV(
            spec.estimator,
            spec.param_grid,
            cv=tscv,
            scoring="neg_log_loss",
            n_jobs=n_jobs,
            refit=True,
        )
        grid.fit(train.X, train.y)
        results[spec.name] = {
            "estimator": grid.best_estimator_,
            "best_params": grid.best_params_,
            "cv_log_loss": float(-grid.best_score_),
        }
        log.info("  %s: CV log loss %.4f  %s", spec.name, -grid.best_score_, grid.best_params_)

    return results


def select_model(
    search_results: dict[str, dict[str, Any]],
    validation: Split,
) -> tuple[str, dict[str, dict[str, float]]]:
    """Pick the model family with the lowest validation log loss."""
    validation_scores = {
        name: score(validation.y, result["estimator"].predict_proba(validation.X))
        for name, result in search_results.items()
    }
    best = min(validation_scores, key=lambda n: validation_scores[n]["log_loss"])
    return best, validation_scores


def select_calibration(
    estimator,
    train: Split,
    validation: Split,
) -> tuple[str, dict[str, dict[str, float]]]:
    """Compare raw, isotonic and Platt-scaled probabilities on the validation slice."""
    scores: dict[str, dict[str, float]] = {}
    for method in CALIBRATION_METHODS:
        candidate = _wrap_calibration(estimator, method)
        candidate.fit(train.X, train.y)
        scores[method] = score(validation.y, candidate.predict_proba(validation.X))
        log.info("  calibration=%s validation log loss %.4f", method, scores[method]["log_loss"])

    best = min(scores, key=lambda m: scores[m]["log_loss"])
    return best, scores


def feature_importance(estimator, feature_columns) -> dict[str, float]:
    """Average absolute coefficient, or impurity importance, whichever the model exposes."""
    model = estimator
    if hasattr(model, "named_steps"):
        model = model.named_steps["clf"]

    if hasattr(model, "coef_"):
        values = np.abs(model.coef_).mean(axis=0)
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    else:
        return {}

    ranked = sorted(zip(feature_columns, values, strict=True), key=lambda pair: -pair[1])
    return {name: float(value) for name, value in ranked}


def _write_report(report: dict[str, Any], reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Training report",
        "",
        f"Generated {report['trained_at']} on {report['dataset']['n_matches']} usable matches "
        f"({report['dataset']['date_range'][0]} to {report['dataset']['date_range'][1]}).",
        "",
        "## Model search (log loss, lower is better)",
        "",
        "| Model | CV log loss (train) | Validation log loss | Validation accuracy |",
        "|---|---|---|---|",
    ]
    for name, entry in report["model_search"].items():
        lines.append(
            f"| {name} | {entry['cv_log_loss']:.4f} | {entry['validation']['log_loss']:.4f} "
            f"| {entry['validation']['accuracy']:.4f} |"
        )

    lines += [
        "",
        f"Selected: **{report['selected_model']}** with calibration **{report['selected_calibration']}**.",
        "",
        "## Calibration choice (validation)",
        "",
        "| Method | Log loss | Brier |",
        "|---|---|---|",
    ]
    for method, entry in report["calibration_search"].items():
        lines.append(f"| {method} | {entry['log_loss']:.4f} | {entry['brier']:.4f} |")

    test = report["test"]
    lines += [
        "",
        "## Held-out test (2019 onwards, scored once)",
        "",
        "| | Log loss | Accuracy | Brier |",
        "|---|---|---|---|",
        f"| Model | {test['model']['log_loss']:.4f} | {test['model']['accuracy']:.4f} "
        f"| {test['model']['brier']:.4f} |",
        f"| Class-prior baseline | {test['baselines']['class_prior']['log_loss']:.4f} "
        f"| {test['baselines']['class_prior']['accuracy']:.4f} "
        f"| {test['baselines']['class_prior']['brier']:.4f} |",
        f"| Majority-class baseline | n/a | {test['baselines']['majority_class']['accuracy']:.4f} "
        f"| {test['baselines']['majority_class']['brier']:.4f} |",
        "",
        "## Feature importance (selected model, refit on all data)",
        "",
        "| Feature | Importance |",
        "|---|---|",
    ]
    for name, value in report["feature_importance"].items():
        lines.append(f"| {name} | {value:.4f} |")

    (reports_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(
    data_path: str | Path = RAW_RESULTS_CSV,
    bundle_path: str | Path = MODEL_BUNDLE_PATH,
    reports_dir: str | Path = REPORTS_DIR,
    elo_config: EloConfig = DEFAULT_ELO,
    fast: bool = True,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Run the full protocol and write the model bundle plus a training report."""
    reports_dir = Path(reports_dir)

    log.info("Loading matches from %s", data_path)
    matches = load_matches(data_path)
    table, elo_ratings, team_state = build_feature_table(matches, elo_config)
    table = drop_incomplete(table, FEATURE_COLUMNS)
    log.info("Usable matches after feature engineering: %d", len(table))

    train_split, validation_split, test_split = split_by_date(table)
    log.info(
        "Split sizes - train %d, validation %d, test %d",
        len(train_split),
        len(validation_split),
        len(test_split),
    )
    for split in (train_split, validation_split, test_split):
        if len(split) == 0:
            raise ValueError(f"The {split.name} split is empty; check the split dates.")

    search_results = search_models(train_split, fast=fast, n_jobs=n_jobs)
    best_name, validation_scores = select_model(search_results, validation_split)
    log.info("Selected %s on validation log loss", best_name)

    best_estimator = search_results[best_name]["estimator"]
    calibration, calibration_scores = select_calibration(best_estimator, train_split, validation_split)
    log.info("Selected calibration: %s", calibration)

    # Refit the chosen configuration on train + validation, then score the test
    # slice a single time. Nothing below this point may change the choices above.
    dev_X = pd.concat([train_split.X, validation_split.X])
    dev_y = np.concatenate([train_split.y, validation_split.y])
    tested = _wrap_calibration(best_estimator, calibration)
    tested.fit(dev_X, dev_y)
    test_probs = tested.predict_proba(test_split.X)
    test_scores = score(test_split.y, test_probs)
    log.info("Held-out test log loss %.4f, accuracy %.4f", test_scores["log_loss"], test_scores["accuracy"])

    calibration_plot = plot_calibration(
        test_split.y,
        test_probs,
        reports_dir / "calibration_test.png",
        title=f"{best_name} ({calibration}) - held-out test",
    )

    # Deployment artifact: same configuration, refit on every usable match.
    all_X = table[list(FEATURE_COLUMNS)]
    all_y = encode_labels(table["result"])
    final_model = _wrap_calibration(best_estimator, calibration)
    final_model.fit(all_X, all_y)

    report: dict[str, Any] = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "path": str(data_path),
            "n_matches": int(len(table)),
            "date_range": [str(table["date"].min().date()), str(table["date"].max().date())],
            "n_teams": int(len(team_state)),
        },
        "split": {
            "train_end": TRAIN_END,
            "test_start": TEST_START,
            "sizes": {
                "train": len(train_split),
                "validation": len(validation_split),
                "test": len(test_split),
            },
        },
        "model_search": {
            name: {
                "cv_log_loss": result["cv_log_loss"],
                "best_params": {k: str(v) for k, v in result["best_params"].items()},
                "validation": validation_scores[name],
            }
            for name, result in search_results.items()
        },
        "selected_model": best_name,
        "selected_calibration": calibration,
        "calibration_search": calibration_scores,
        "test": {
            "model": test_scores,
            "baselines": baseline_scores(train_split.y, test_split.y),
            "reliability_win": reliability_table(test_split.y, test_probs, "win"),
            "calibration_plot": str(calibration_plot) if calibration_plot else None,
        },
        "feature_importance": feature_importance(
            search_results[best_name]["estimator"], FEATURE_COLUMNS
        ),
    }

    bundle: ModelBundle = {
        "version": BUNDLE_VERSION,
        "model": final_model,
        "feature_columns": list(FEATURE_COLUMNS),
        "classes": list(CLASSES),
        "elo_ratings": elo_ratings,
        "elo_config": asdict(elo_config),
        "team_state": team_state,
        "metadata": report,
    }
    saved_to = save_bundle(bundle, bundle_path)
    log.info("Saved model bundle to %s", saved_to)

    _write_report(report, reports_dir)
    log.info("Wrote report to %s", reports_dir / "report.md")
    return report
