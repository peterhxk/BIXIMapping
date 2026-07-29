# BIXI Anomaly Detection — Predicting La Ronde Fireworks Nights

Detecting and predicting **L'International des Feux Loto-Québec** fireworks nights from BIXI
bike-share demand. When the shells finish bursting over the St. Lawrence at 22:00, the métro and
buses jam solid — so a large share of the crowd **escapes by BIXI**, producing a sharp, one-way
late-evening demand spike across the whole riverfront *viewshed* (Old Port, the Faubourg /
Sainte-Marie riverfront, the Jacques-Cartier bridge approach, and Parc Jean-Drapeau). This project
turns that exodus into features, trains a classifier on four seasons of trips, and ships a predictor
that scores an unlabelled season's CSV for "was there a show tonight?"

The key idea is the **one-way exodus**: other big riverfront events (Osheaga, ÎleSoniq) are just as
busy, but they bring balanced in/out flow. Fireworks empty the zone all at once — so the model keys
on *net outflow shares*, not raw volume.

## What's here

| File | What it does |
|---|---|
| **`firework_anomaly_train.ipynb`** | The training pipeline. Loads 2022–2025, builds features, evaluates **leave-one-year-out** (train on three seasons, test on the held-out one), tunes the decision threshold, refits on all years, and exports the model bundle to `output/firework_model.joblib`. Includes an error-inspection section that lists the misses / false alarms with their features and weather. |
| **`predict_fireworks.py`** | Inference-only CLI. Loads the saved bundle, runs the *same* preprocessing on a new season's trip CSV, and ranks the nights most likely to have hosted a show. No training happens here. |
| **`firework_features.py`** | Shared feature engineering, imported by **both** the notebook and the CLI so train-time and predict-time preprocessing can never drift apart. |
| **`fireworks_dates.json`** | The ground-truth labels — the IFLQ calendars for 2022–2026, with a `confidence` and `source` per season. Single source of truth for both the notebook and the CLI. |
| **`jean_drapeau_anomaly/`** | Companion study on Parc Jean-Drapeau island demand (F1, Osheaga, ÎleSoniq, Piknic). Its per-(month, weekday) baseline + robust MAD z-score method is what the fireworks detector's statistical approach was originally modelled on. |

## How it works

**Train once, serve many.** `firework_anomaly_train.ipynb` produces `output/firework_model.joblib`
(the fitted models + feature list + tuned threshold + provenance). `predict_fireworks.py` just loads
that artifact and scores new data — retrain only when the labels or feature code change.

**Coordinate-matched viewshed.** A trip counts if either endpoint falls inside a fixed lat/lon box,
using each trip's *own* station coordinates rather than matching station names. BIXI's network drifts
year to year; a coordinate box is the same physical place every season.

**Portable features, never raw counts.** Ridership grew a lot from 2022→2025, so the model uses
within-night *shares* and weather — quantities that are comparable across years — so a model trained
on all four seasons transfers.

| Feature | Meaning |
|---|---|
| `netout_share` | Escape-window **net outflow** ÷ escape-window rides — the one-way exodus signature (the strongest signal). |
| `escape_share` | Fraction of the evening's rides that fall in the post-22:00 escape window. |
| `pre_arrival_share` | Fraction of the evening that was people *arriving* beforehand (crowd gathering). |
| `dow` | Day of week (0–6). The weekday calendar signal — deliberately kept minor. |
| `temp_max`, `precip_mm` | Daily max temperature and rainfall at La Ronde (Open-Meteo archive) — the weather confound. |

Windows are keyed to the launch: **pre-show** = 19–21h, **escape** = 22–01h (post-midnight rides are
folded back onto the correct evening via a 4-hour shift).

## Results

Evaluated **leave-one-year-out** across 2022–2025 (368 June–August nights, ~33 fireworks nights),
the logistic-regression model reaches roughly **0.95 PR-AUC** and catches nearly all shows at the
tuned threshold, with only a couple of false alarms per year — a far more honest estimate than any
single-season score, since each year is predicted by a model that never saw it. Run the notebook for
the current exact figures (they refresh on every retrain).

Two things make the result credible rather than lucky:

- **The weekday mix varies by year** — 2022 ran Wednesdays/Saturdays, 2025 ran Thursdays/Sundays. A
  model that merely memorised "it's a Thursday" could not score well across all four seasons, so the
  cross-year performance is evidence it reads the demand *shape*.
- **The honest bottleneck is data, not the model.** ~33 positive nights is tiny; treat the nightly
  *ranking* as the real deliverable, not the third decimal of any metric.

## Usage

Run everything with this folder as the working directory (so the flat `import firework_features` and
the relative `output/` paths resolve).

**Train / refresh the model** — open `firework_anomaly_train.ipynb` and run top to bottom. It writes
`output/firework_model.joblib`.

**Predict on a season:**

```bash
python predict_fireworks.py \
    --predict "/Volumes/Extreme SSD/SUMO Data/DonneesOuvertes2026_<full-season>.csv"
```

Useful flags: `--model {logreg,gb}`, `--threshold <float>` (override the saved default),
`--top-n <n>`, `--no-weather` (offline; only if the model was trained without weather),
`--out ranked.csv`. When the predicted season's calendar is in `fireworks_dates.json`, the CLI also
prints a validation line against ground truth.

## Data

Trip CSVs (one row per station-to-station trip, with inline station coordinates) come from
[Montreal's BIXI open data](https://donnees.montreal.ca/en/dataset/bixi-historique-des-deplacements?);
weather from the [Open-Meteo historical archive](https://open-meteo.com/en/docs/historical-weather-api)
(cached under `output/firework_weather_<year>.csv`). The four training seasons:

```
DonneesOuverte2022.csv
DonneesOuvertes2023_12.csv
DonneesOuvertes2024_010203040506070809101112.csv
DonneesOuvertes2025_010203040506070809101112.csv
```

Paths are set at the top of the training notebook and currently point at `/Volumes/Extreme SSD/SUMO
Data/`; edit them to match where you keep the data. All four years share the same schema, so no
per-year special-casing is needed.

## Notes

- **Labels need care.** `fireworks_dates.json` marks each season's `confidence`. 2022 and 2025 are
  from primary sources; verify the others against an official press release before treating any final
  metric as gospel. An empty season means "not yet sourced," not "no shows that year."
- **Unlabelled events cost precision.** Other riverfront festivals overlap the zone and season. Fold
  them in as labels if their exodus is being scored as a false positive against the fireworks calendar.
- **The exodus can run late** on the biggest shows; widen `ESCAPE_HOURS` in `firework_features.py` if
  it spills past 01:59.