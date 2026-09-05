"""FastAPI backend serving predictions and the single-page web UI."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .artifacts import BundleNotFoundError
from .config import MODEL_BUNDLE_PATH
from .predict import COMPETITIONS, Predictor, UnknownTeamError

WEB_DIR = Path(__file__).parent / "web"


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """Load the bundle once per process. Override the path with WCP_BUNDLE."""
    path = os.environ.get("WCP_BUNDLE", str(MODEL_BUNDLE_PATH))
    return Predictor.load(path)


def _predictor_or_503() -> Predictor:
    try:
        return get_predictor()
    except BundleNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


class PredictRequest(BaseModel):
    home_team: str = Field(..., description="Team listed first, treated as the home side")
    away_team: str
    neutral: bool = Field(True, description="True for a neutral venue such as a World Cup match")
    competition: str = Field("major", description=f"One of {', '.join(COMPETITIONS)}")
    knockout: bool = Field(False, description="Also return who advances after extra time and penalties")


class PredictResponse(BaseModel):
    home_team: str
    away_team: str
    home_elo: float
    away_elo: float
    neutral: bool
    competition: str
    home_win: float
    draw: float
    away_win: float
    home_advance: float | None = None
    away_advance: float | None = None


class TeamInfo(BaseModel):
    team: str
    elo: float
    matches_played: int
    last_match: str


app = FastAPI(
    title="National Team Predictor",
    version="1.0.0",
    description="Win / draw / loss probabilities for men's international football fixtures.",
)


@app.get("/api/health")
def health() -> dict:
    """Report whether a trained bundle is available."""
    try:
        predictor = get_predictor()
    except BundleNotFoundError as error:
        return {"status": "no_model", "detail": str(error)}
    return {
        "status": "ok",
        "model": predictor.metadata["selected_model"],
        "calibration": predictor.metadata["selected_calibration"],
        "trained_at": predictor.metadata["trained_at"],
        "teams": len(predictor.team_state),
    }


@app.get("/api/teams", response_model=list[TeamInfo])
def teams() -> list[dict]:
    """Every known team with its current ELO, strongest first."""
    return _predictor_or_503().team_table()


@app.get("/api/metrics")
def metrics() -> dict:
    """The training report stored alongside the model."""
    return _predictor_or_503().metadata


@app.post("/api/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> dict:
    """Outcome probabilities for one fixture."""
    predictor = _predictor_or_503()
    try:
        prediction = predictor.predict(
            request.home_team,
            request.away_team,
            neutral=request.neutral,
            competition=request.competition,
            knockout=request.knockout,
        )
    except UnknownTeamError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return prediction.to_dict()


if WEB_DIR.exists():

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
