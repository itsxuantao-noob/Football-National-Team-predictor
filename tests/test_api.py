from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", reason="the API extra is not installed")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client(trained_bundle, monkeypatch_session):
    from wcpredictor import api

    bundle_path, _ = trained_bundle
    monkeypatch_session.setenv("WCP_BUNDLE", str(bundle_path))
    api.get_predictor.cache_clear()
    with TestClient(api.app) as test_client:
        yield test_client
    api.get_predictor.cache_clear()


@pytest.fixture(scope="session")
def monkeypatch_session():
    from _pytest.monkeypatch import MonkeyPatch

    patcher = MonkeyPatch()
    yield patcher
    patcher.undo()


def test_health_reports_the_loaded_model(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "logistic_regression"
    assert body["teams"] > 0


def test_teams_are_returned_sorted_by_elo(client):
    teams = client.get("/api/teams").json()
    assert teams[0]["team"] == "Alphaland"
    assert [t["elo"] for t in teams] == sorted((t["elo"] for t in teams), reverse=True)


def test_predict_returns_a_distribution(client):
    response = client.post(
        "/api/predict",
        json={"home_team": "Alphaland", "away_team": "Kappania", "neutral": True, "knockout": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["home_win"] + body["draw"] + body["away_win"] == pytest.approx(1.0)
    assert body["home_advance"] + body["away_advance"] == pytest.approx(1.0)


def test_unknown_team_is_a_404_with_a_suggestion(client):
    response = client.post("/api/predict", json={"home_team": "Alphalnd", "away_team": "Betaria"})
    assert response.status_code == 404
    assert "Alphaland" in response.json()["detail"]


def test_a_team_playing_itself_is_a_400(client):
    response = client.post("/api/predict", json={"home_team": "Betaria", "away_team": "Betaria"})
    assert response.status_code == 400


def test_missing_fields_are_rejected_by_validation(client):
    assert client.post("/api/predict", json={"home_team": "Betaria"}).status_code == 422


def test_metrics_endpoint_exposes_the_training_report(client):
    metrics = client.get("/api/metrics").json()
    assert metrics["selected_model"] == "logistic_regression"
    assert "test" in metrics


def test_the_web_ui_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "National Team Predictor" in response.text
    assert client.get("/static/app.js").status_code == 200
