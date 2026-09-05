"""Shared fixtures. Everything here is synthetic so the suite runs without the Kaggle download."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TEAMS = [
    ("Alphaland", 1900),
    ("Betaria", 1820),
    ("Gammastan", 1760),
    ("Deltania", 1700),
    ("Epsilonia", 1640),
    ("Zetavia", 1580),
    ("Etaburg", 1520),
    ("Thetamark", 1460),
    ("Iotaland", 1400),
    ("Kappania", 1340),
]

TOURNAMENTS = [
    "Friendly",
    "FIFA World Cup qualification",
    "FIFA World Cup",
    "UEFA Euro",
    "Nations League",
]


def make_results(n_matches: int = 2400, seed: int = 7) -> pd.DataFrame:
    """Generate a results.csv-shaped table with a plausible strength signal.

    Outcomes are drawn from the latent strength gap, so form and ELO features
    carry real information and models trained on it behave sensibly.
    """
    rng = np.random.default_rng(seed)
    names = [name for name, _ in TEAMS]
    strengths = {name: rating for name, rating in TEAMS}

    dates = pd.to_datetime("2000-01-01") + pd.to_timedelta(
        np.sort(rng.integers(0, 365 * 26, size=n_matches)), unit="D"
    )

    rows = []
    for date in dates:
        home, away = rng.choice(len(names), size=2, replace=False)
        home_name, away_name = names[home], names[away]
        neutral = bool(rng.random() < 0.35)

        gap = strengths[home_name] - strengths[away_name] + (0 if neutral else 70)
        home_rate = np.exp(gap / 500) * 1.35
        away_rate = np.exp(-gap / 500) * 1.35
        home_score = int(rng.poisson(home_rate))
        away_score = int(rng.poisson(away_rate))

        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "home_team": home_name,
                "away_team": away_name,
                "home_score": home_score,
                "away_score": away_score,
                "tournament": str(rng.choice(TOURNAMENTS)),
                "city": "Testville",
                "country": "Testland" if not neutral else "Neutralia",
                "neutral": neutral,
            }
        )

    played = pd.DataFrame(rows)

    # A couple of scheduled fixtures with no score, mirroring the real dataset.
    upcoming = pd.DataFrame(
        [
            {
                "date": "2026-06-11",
                "home_team": "Alphaland",
                "away_team": "Betaria",
                "home_score": np.nan,
                "away_score": np.nan,
                "tournament": "FIFA World Cup",
                "city": "Testville",
                "country": "Neutralia",
                "neutral": True,
            }
        ]
    )
    return pd.concat([played, upcoming], ignore_index=True)


@pytest.fixture(scope="session")
def results_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "results.csv"
    make_results().to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def matches(results_csv: Path) -> pd.DataFrame:
    from wcpredictor.data import load_matches

    return load_matches(results_csv)


@pytest.fixture(scope="session")
def feature_table(matches: pd.DataFrame):
    from wcpredictor.features import build_feature_table

    return build_feature_table(matches)


@pytest.fixture(scope="session")
def trained_bundle(results_csv: Path, tmp_path_factory: pytest.TempPathFactory):
    """Run the real training protocol, but over a single one-point grid to keep it quick."""
    from unittest.mock import patch

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from wcpredictor import train as train_module
    from wcpredictor.modeling import ModelSpec

    spec = ModelSpec(
        name="logistic_regression",
        estimator=Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=0))]
        ),
        param_grid={"clf__C": [1.0]},
        needs_scaling=True,
    )

    out_dir = tmp_path_factory.mktemp("artifacts")
    bundle_path = out_dir / "model_bundle.joblib"
    with patch.object(train_module, "get_model_specs", lambda fast=True: [spec]):
        report = train_module.train(
            data_path=results_csv,
            bundle_path=bundle_path,
            reports_dir=out_dir / "reports",
            n_jobs=1,
        )
    return bundle_path, report


@pytest.fixture(scope="session")
def predictor(trained_bundle):
    from wcpredictor.predict import Predictor

    bundle_path, _ = trained_bundle
    return Predictor.load(bundle_path)
