# Data

This directory holds the raw dataset, which is not checked in.

Download `results.csv` from
[International Football Results 1872–present](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
and place it here as `data/results.csv`.

Expected columns:

```
date, home_team, away_team, home_score, away_score, tournament, city, country, neutral
```

Rows with a missing score are treated as scheduled fixtures rather than results,
so a file that includes upcoming matches works fine.
