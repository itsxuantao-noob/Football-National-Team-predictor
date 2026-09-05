from __future__ import annotations

import pytest

from wcpredictor.predict import Predictor, UnknownTeamError


def test_probabilities_form_a_distribution(predictor: Predictor):
    prediction = predictor.predict("Alphaland", "Kappania")
    total = prediction.home_win + prediction.draw + prediction.away_win
    assert total == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in (prediction.home_win, prediction.draw, prediction.away_win))


def test_the_stronger_team_is_favoured(predictor: Predictor):
    prediction = predictor.predict("Alphaland", "Kappania", neutral=True)
    assert prediction.home_win > prediction.away_win
    assert prediction.home_elo > prediction.away_elo


def test_swapping_sides_swaps_the_probabilities_on_neutral_ground(predictor: Predictor):
    forward = predictor.predict("Betaria", "Etaburg", neutral=True)
    reverse = predictor.predict("Etaburg", "Betaria", neutral=True)

    assert forward.home_win == pytest.approx(reverse.away_win, abs=0.05)
    assert forward.draw == pytest.approx(reverse.draw, abs=0.05)


def test_home_advantage_helps_the_home_side(predictor: Predictor):
    neutral = predictor.predict("Deltania", "Epsilonia", neutral=True)
    at_home = predictor.predict("Deltania", "Epsilonia", neutral=False)
    assert at_home.home_win > neutral.home_win


def test_knockout_advance_probabilities_absorb_the_draw(predictor: Predictor):
    prediction = predictor.predict("Alphaland", "Gammastan", knockout=True)
    assert prediction.home_advance + prediction.away_advance == pytest.approx(1.0)
    assert prediction.home_advance > prediction.home_win
    # The stronger side takes the larger share of the drawn matches.
    assert prediction.home_advance - prediction.home_win > prediction.draw / 2


def test_advance_probabilities_are_omitted_outside_knockouts(predictor: Predictor):
    prediction = predictor.predict("Alphaland", "Gammastan", knockout=False)
    assert prediction.home_advance is None
    assert prediction.away_advance is None


def test_unknown_team_suggests_a_close_match(predictor: Predictor):
    with pytest.raises(UnknownTeamError) as excinfo:
        predictor.predict("Alphaand", "Betaria")
    assert "Alphaland" in excinfo.value.suggestions


def test_a_team_cannot_play_itself(predictor: Predictor):
    with pytest.raises(ValueError):
        predictor.predict("Betaria", "Betaria")


def test_unknown_competition_is_rejected(predictor: Predictor):
    with pytest.raises(ValueError):
        predictor.predict("Betaria", "Etaburg", competition="testimonial")


def test_competition_changes_the_prediction(predictor: Predictor):
    major = predictor.predict("Betaria", "Iotaland", competition="major")
    friendly = predictor.predict("Betaria", "Iotaland", competition="friendly")
    assert major.home_win != friendly.home_win


def test_team_table_is_sorted_by_elo(predictor: Predictor):
    table = predictor.team_table()
    elos = [row["elo"] for row in table]
    assert elos == sorted(elos, reverse=True)
    assert table[0]["matches_played"] > 0


def test_formatted_output_names_both_teams(predictor: Predictor):
    text = predictor.predict("Alphaland", "Zetavia", knockout=True).format()
    assert "Alphaland" in text and "Zetavia" in text
    assert "Advances" in text


def test_predict_many_handles_a_fixture_list(predictor: Predictor):
    predictions = predictor.predict_many(
        [
            {"home_team": "Alphaland", "away_team": "Betaria"},
            {"home_team": "Gammastan", "away_team": "Deltania", "knockout": True},
        ]
    )
    assert len(predictions) == 2
    assert predictions[1].home_advance is not None
