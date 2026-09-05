from __future__ import annotations

import json

import pytest

from wcpredictor.cli import main


def test_predict_prints_a_readable_line(trained_bundle, capsys):
    bundle_path, _ = trained_bundle
    exit_code = main(["predict", "Alphaland", "Kappania", "--bundle", str(bundle_path), "--knockout"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Alphaland vs Kappania" in output
    assert "Advances" in output


def test_predict_json_output_is_parseable(trained_bundle, capsys):
    bundle_path, _ = trained_bundle
    main(["predict", "Betaria", "Etaburg", "--bundle", str(bundle_path), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["home_team"] == "Betaria"
    assert payload["home_win"] + payload["draw"] + payload["away_win"] == pytest.approx(1.0)


def test_teams_can_be_filtered(trained_bundle, capsys):
    bundle_path, _ = trained_bundle
    main(["teams", "--bundle", str(bundle_path), "--search", "alpha"])
    output = capsys.readouterr().out

    assert "Alphaland" in output
    assert "Kappania" not in output


def test_report_summarises_the_stored_metrics(trained_bundle, capsys):
    bundle_path, _ = trained_bundle
    main(["report", "--bundle", str(bundle_path)])
    output = capsys.readouterr().out

    assert "logistic_regression" in output
    assert "Held-out test" in output


def test_unknown_team_exits_nonzero_with_a_message(trained_bundle, capsys):
    bundle_path, _ = trained_bundle
    exit_code = main(["predict", "Nowhereland", "Betaria", "--bundle", str(bundle_path)])

    assert exit_code == 1
    assert "Unknown team" in capsys.readouterr().err


def test_missing_bundle_exits_nonzero(tmp_path, capsys):
    exit_code = main(["teams", "--bundle", str(tmp_path / "absent.joblib")])

    assert exit_code == 1
    assert "wcp train" in capsys.readouterr().err
