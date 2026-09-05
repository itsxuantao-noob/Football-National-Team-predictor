"""Turning a trained bundle into match predictions."""

from __future__ import annotations

import difflib
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .artifacts import ModelBundle, load_bundle
from .config import MODEL_BUNDLE_PATH

Competition = str
COMPETITIONS: tuple[Competition, ...] = ("friendly", "qualification", "major", "other")

# Extra time and penalties are close to a coin flip, but not quite: the stronger
# side keeps a small edge. The cap keeps that edge below ~62% however lopsided
# the rating gap is.
TIEBREAK_ELO_SCALE = 800.0
TIEBREAK_MAX_EDGE = 0.12


class UnknownTeamError(KeyError):
    """Raised when a team name is not present in the trained bundle."""

    def __init__(self, team: str, suggestions: list[str]):
        self.team = team
        self.suggestions = suggestions
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        super().__init__(f"Unknown team '{team}'.{hint}")


@dataclass
class MatchPrediction:
    """Outcome probabilities for a single fixture, from the home team's view."""

    home_team: str
    away_team: str
    home_elo: float
    away_elo: float
    neutral: bool
    competition: Competition
    home_win: float
    draw: float
    away_win: float
    home_advance: float | None = None
    away_advance: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def format(self) -> str:
        venue = "neutral venue" if self.neutral else f"{self.home_team} at home"
        lines = [
            f"{self.home_team} vs {self.away_team}  ({venue}, {self.competition})",
            f"  ELO {self.home_elo:.0f} vs {self.away_elo:.0f}  (diff {self.home_elo - self.away_elo:+.0f})",
            f"  {self.home_team} win {self.home_win * 100:5.1f}%"
            f"   Draw {self.draw * 100:5.1f}%"
            f"   {self.away_team} win {self.away_win * 100:5.1f}%",
        ]
        if self.home_advance is not None:
            lines.append(
                f"  Advances: {self.home_team} {self.home_advance * 100:5.1f}%"
                f"   {self.away_team} {self.away_advance * 100:5.1f}%"
            )
        return "\n".join(lines)


def _competition_flags(competition: Competition) -> dict[str, int]:
    if competition not in COMPETITIONS:
        raise ValueError(f"competition must be one of {COMPETITIONS}, got '{competition}'")
    return {
        "is_friendly": int(competition == "friendly"),
        "is_qualification": int(competition == "qualification"),
        "is_major": int(competition == "major"),
    }


class Predictor:
    """Loads a bundle once and answers prediction queries against it."""

    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle
        self.model = bundle["model"]
        self.feature_columns = list(bundle["feature_columns"])
        self.classes = list(bundle["classes"])
        self.elo_ratings: dict[str, float] = bundle["elo_ratings"]
        self.team_state: pd.DataFrame = bundle["team_state"]
        self.initial_elo = float(bundle["elo_config"]["initial"])

    @classmethod
    def load(cls, path: str | Path = MODEL_BUNDLE_PATH) -> Predictor:
        return cls(load_bundle(path))

    @property
    def metadata(self) -> dict:
        return self.bundle["metadata"]

    def teams(self) -> list[str]:
        """Every team the model knows, strongest first."""
        names = list(self.team_state.index)
        return sorted(names, key=lambda t: -self.elo(t))

    def elo(self, team: str) -> float:
        return float(self.elo_ratings.get(team, self.initial_elo))

    def team_table(self) -> list[dict]:
        """Team name, current ELO and match count, for populating a UI."""
        state = self.team_state
        return [
            {
                "team": team,
                "elo": round(self.elo(team), 1),
                "matches_played": int(state.at[team, "matches_played"]),
                "last_match": str(pd.Timestamp(state.at[team, "last_match_date"]).date()),
            }
            for team in self.teams()
        ]

    def _require_team(self, team: str) -> str:
        if team in self.team_state.index:
            return team
        suggestions = difflib.get_close_matches(team, list(self.team_state.index), n=3, cutoff=0.6)
        raise UnknownTeamError(team, suggestions)

    def _feature_row(
        self,
        home_team: str,
        away_team: str,
        neutral: bool,
        competition: Competition,
    ) -> pd.DataFrame:
        home_state = self.team_state.loc[home_team]
        away_state = self.team_state.loc[away_team]
        home_elo, away_elo = self.elo(home_team), self.elo(away_team)

        context = {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "neutral_int": int(neutral),
            **_competition_flags(competition),
        }

        values = []
        for column in self.feature_columns:
            if column in context:
                values.append(context[column])
            elif column.startswith("home_"):
                values.append(home_state[column.removeprefix("home_")])
            elif column.startswith("away_"):
                values.append(away_state[column.removeprefix("away_")])
            else:
                raise KeyError(f"No way to build feature '{column}' at prediction time.")

        return pd.DataFrame([values], columns=self.feature_columns)

    def predict(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
        competition: Competition = "major",
        knockout: bool = False,
    ) -> MatchPrediction:
        """Probabilities for one fixture, optionally including who advances."""
        home_team = self._require_team(home_team)
        away_team = self._require_team(away_team)
        if home_team == away_team:
            raise ValueError("A team cannot play itself.")

        features = self._feature_row(home_team, away_team, neutral, competition)
        probabilities = self.model.predict_proba(features)[0]
        by_class = dict(zip(self.classes, (float(p) for p in probabilities), strict=True))

        home_elo, away_elo = self.elo(home_team), self.elo(away_team)
        prediction = MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            home_elo=home_elo,
            away_elo=away_elo,
            neutral=neutral,
            competition=competition,
            home_win=by_class["win"],
            draw=by_class["draw"],
            away_win=by_class["lose"],
        )

        if knockout:
            edge = (home_elo - away_elo) / TIEBREAK_ELO_SCALE
            home_share = 0.5 + max(-TIEBREAK_MAX_EDGE, min(TIEBREAK_MAX_EDGE, edge))
            prediction.home_advance = prediction.home_win + prediction.draw * home_share
            prediction.away_advance = prediction.away_win + prediction.draw * (1 - home_share)

        return prediction

    def predict_many(self, fixtures: list[dict]) -> list[MatchPrediction]:
        """Predict a list of `{home_team, away_team, ...}` dicts."""
        return [self.predict(**fixture) for fixture in fixtures]
