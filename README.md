# National Team Predictor ⚽

A machine learning model that predicts win / draw / loss probabilities for
men's international football matches. Built to output calibrated probabilities
for match-outcome analysis and potential probability trading.

## Overview

This project predicts the outcome of any men's national team fixture using
historical international results from 1872 to 2026. It engineers team-strength
and form features, compares several models under leakage-safe validation, and
produces probability estimates for upcoming matches.

The model is general-purpose (any two national teams) — predicting the 2026
World Cup knockout stage is included as an example application.

## Data

Built on the [International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
dataset (men's national team matches, 1872–present). Download `results.csv` and
place it in the `data/` directory.

## Feature Engineering

All features are computed leakage-safe (only information available before each
match is used):

- **ELO ratings** — implemented from scratch, updated match by match, with home
  advantage and neutral-venue handling
- **Win / draw rates** — pre-match cumulative rates per team
- **Goal difference** — cumulative and recent (last-5) average margin
- **Tournament importance** — friendly / qualifier / major-tournament flags
- **Neutral venue** — home-advantage adjustment

## Modeling

Models were tuned with `GridSearchCV` using `TimeSeriesSplit` (time-ordered
cross-validation, no future leakage) and scored by **log loss**, the metric that
matters for probability quality.

| Model | CV Log Loss |
|-------|-------------|
| **Logistic Regression** (selected) | **0.909** |
| XGBoost | 0.912 |
| Random Forest | 0.917 |

**Logistic Regression was selected.** International football is high-noise and
largely driven by strength difference, an approximately linear signal — so a
simple, regularized linear model generalizes better than more complex tree-based
models, which risk fitting noise.

Validation accuracy is ~58%, in line with strong football models (bookmaker
models typically reach 53–55%); much of the outcome is inherently unpredictable.

## Project Structure
## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Open `explore.ipynb` in Jupyter or VS Code and run all cells.

## Roadmap

- [x] Feature engineering (ELO, form, goal difference, tournament importance)
- [x] Model comparison and selection (leakage-safe, log-loss based)
- [ ] Probability calibration (isotonic / Platt scaling)
- [ ] Final held-out test evaluation
- [ ] Backend API + web frontend for live match predictions

## Notes

A personal project focused on building a complete ML pipeline end to end — from
raw data through feature engineering, model selection, and (eventually)
deployment.