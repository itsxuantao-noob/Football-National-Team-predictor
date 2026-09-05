"""Command line interface: `wcp train`, `wcp predict`, `wcp teams`, `wcp serve`."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .artifacts import BundleNotFoundError
from .config import MODEL_BUNDLE_PATH, RAW_RESULTS_CSV, REPORTS_DIR
from .data import MissingDataError
from .predict import COMPETITIONS, Predictor, UnknownTeamError


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def _cmd_train(args: argparse.Namespace) -> int:
    from .train import train

    report = train(
        data_path=args.data,
        bundle_path=args.bundle,
        reports_dir=args.reports,
        fast=not args.full_grid,
        n_jobs=args.jobs,
    )
    test = report["test"]["model"]
    print()
    print(f"Selected {report['selected_model']} (calibration: {report['selected_calibration']})")
    print(f"Held-out test  log loss {test['log_loss']:.4f}   accuracy {test['accuracy']:.4f}")
    print(f"Bundle: {args.bundle}")
    print(f"Report: {args.reports}/report.md")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    predictor = Predictor.load(args.bundle)
    prediction = predictor.predict(
        args.home,
        args.away,
        neutral=not args.home_advantage,
        competition=args.competition,
        knockout=args.knockout,
    )
    if args.json:
        print(json.dumps(prediction.to_dict(), indent=2))
    else:
        print(prediction.format())
    return 0


def _cmd_teams(args: argparse.Namespace) -> int:
    predictor = Predictor.load(args.bundle)
    rows = predictor.team_table()
    if args.search:
        needle = args.search.lower()
        rows = [row for row in rows if needle in row["team"].lower()]
    rows = rows[: args.limit]

    width = max((len(row["team"]) for row in rows), default=4)
    print(f"{'TEAM'.ljust(width)}   ELO   MATCHES  LAST MATCH")
    for row in rows:
        print(
            f"{row['team'].ljust(width)}  {row['elo']:6.0f}  "
            f"{row['matches_played']:7d}  {row['last_match']}"
        )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    predictor = Predictor.load(args.bundle)
    metadata = predictor.metadata
    if args.json:
        print(json.dumps(metadata, indent=2, default=str))
        return 0

    print(f"Trained at      {metadata['trained_at']}")
    print(f"Matches         {metadata['dataset']['n_matches']} "
          f"({metadata['dataset']['date_range'][0]} to {metadata['dataset']['date_range'][1]})")
    print(f"Selected model  {metadata['selected_model']} (calibration: {metadata['selected_calibration']})")
    print()
    print(f"{'MODEL':22} {'CV LOGLOSS':>11} {'VAL LOGLOSS':>12} {'VAL ACC':>8}")
    for name, entry in metadata["model_search"].items():
        print(f"{name:22} {entry['cv_log_loss']:11.4f} "
              f"{entry['validation']['log_loss']:12.4f} {entry['validation']['accuracy']:8.4f}")
    test = metadata["test"]
    print()
    print("Held-out test (2019+):")
    print(f"  model            log loss {test['model']['log_loss']:.4f}   "
          f"accuracy {test['model']['accuracy']:.4f}   brier {test['model']['brier']:.4f}")
    print(f"  class prior      log loss {test['baselines']['class_prior']['log_loss']:.4f}   "
          f"accuracy {test['baselines']['class_prior']['accuracy']:.4f}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("The web app needs extra packages: pip install -e .[api]", file=sys.stderr)
        return 1

    uvicorn.run("wcpredictor.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wcp", description="International football match predictor.")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train, evaluate and save the model bundle")
    train_parser.add_argument("--data", default=str(RAW_RESULTS_CSV), help="path to results.csv")
    train_parser.add_argument("--bundle", default=str(MODEL_BUNDLE_PATH), help="where to write the bundle")
    train_parser.add_argument("--reports", default=str(REPORTS_DIR), help="where to write metrics and plots")
    train_parser.add_argument("--full-grid", action="store_true", help="search the wide grids (much slower)")
    train_parser.add_argument("--jobs", type=int, default=-1, help="parallel jobs for grid search")
    train_parser.set_defaults(func=_cmd_train)

    predict_parser = subparsers.add_parser("predict", help="predict a single fixture")
    predict_parser.add_argument("home")
    predict_parser.add_argument("away")
    predict_parser.add_argument("--bundle", default=str(MODEL_BUNDLE_PATH))
    predict_parser.add_argument(
        "--home-advantage",
        action="store_true",
        help="the match is at the home team's ground (default assumes a neutral venue)",
    )
    predict_parser.add_argument("--competition", choices=COMPETITIONS, default="major")
    predict_parser.add_argument("--knockout", action="store_true", help="also report advance probabilities")
    predict_parser.add_argument("--json", action="store_true")
    predict_parser.set_defaults(func=_cmd_predict)

    teams_parser = subparsers.add_parser("teams", help="list known teams by ELO")
    teams_parser.add_argument("--bundle", default=str(MODEL_BUNDLE_PATH))
    teams_parser.add_argument("--search", help="filter by substring")
    teams_parser.add_argument("--limit", type=int, default=30)
    teams_parser.set_defaults(func=_cmd_teams)

    report_parser = subparsers.add_parser("report", help="show the metrics stored in the bundle")
    report_parser.add_argument("--bundle", default=str(MODEL_BUNDLE_PATH))
    report_parser.add_argument("--json", action="store_true")
    report_parser.set_defaults(func=_cmd_report)

    serve_parser = subparsers.add_parser("serve", help="run the prediction API and web UI")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    serve_parser.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return args.func(args)
    except (MissingDataError, BundleNotFoundError, UnknownTeamError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
