"""Shared feature engineering for La Ronde fireworks detection.

Imported by BOTH firework_anomaly_train.ipynb (training) and predict_fireworks.py
(serving) so the preprocessing can never drift between train and predict time --
the classic source of "great in the notebook, broken in production" bugs.

Design choices that make the pipeline cross-year portable:
  * Viewshed is matched on each trip's OWN station coordinates (inline in the CSV),
    not by station name against a current station_information.json. BIXI's network
    changes year to year; a coordinate box is the same physical place every season.
  * Features are per-year-normalised or within-night ratios (never raw counts), so a
    model trained on 2022-2025 transfers despite ridership growth.
  * The robust z-score baseline is computed over EVERY night (labels not needed), so
    it is defined identically at train and predict time.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "America/Montreal"
LARONDE_LAT, LARONDE_LON = 45.5222, -73.5320

# Viewshed bounding box: Old Port + Faubourg/Sainte-Marie riverfront + Jacques-Cartier
# bridge approach + Parc Jean-Drapeau (matches firework_anomaly.ipynb).
VIEW_LAT = (45.503, 45.530)
VIEW_LON = (-73.560, -73.520)

PRESHOW_HOURS = [19, 20, 21]        # crowd arriving to watch
ESCAPE_HOURS  = [22, 23, 0, 1]      # show + the metro-is-jammed BIXI exodus
EVENING_HOURS = PRESHOW_HOURS + ESCAPE_HOURS

# Coordinate-based viewshed -> read the station lat/lon columns, not names.
TRIP_COLS = ["STARTTIMEMS",
             "STARTSTATIONLATITUDE", "STARTSTATIONLONGITUDE",
             "ENDSTATIONLATITUDE", "ENDSTATIONLONGITUDE"]

# Portable feature set (ratios / per-year-normalised / weather -- never raw counts).
BASE_FEATURES = ["robust_z", "escape_share", "netout_share", "pre_arrival_share", "dow"]
WEATHER_FEATURES = ["temp_max", "precip_mm"]

FIREWORKS_HOUR = 22   # every show launches at 22:00 local


# --------------------------------------------------------------------------- #
# Trip loading + nightly aggregation                                          #
# --------------------------------------------------------------------------- #
def _in_box(lat, lon):
    return ((VIEW_LAT[0] <= lat) & (lat <= VIEW_LAT[1]) &
            (VIEW_LON[0] <= lon) & (lon <= VIEW_LON[1]))


def load_zone_trips(path, chunksize=200_000):
    """Stream a trip CSV, keeping only trips with an endpoint inside the viewshed box.

    Returns a frame with one row per kept trip: clock_hour, night_date (00:00-03:59
    folded into the previous evening), and is_departure / is_arrival flags.
    """
    kept = []
    for chunk in pd.read_csv(path, usecols=TRIP_COLS, chunksize=chunksize, low_memory=False):
        chunk = chunk.dropna(subset=TRIP_COLS)
        start_in = _in_box(chunk["STARTSTATIONLATITUDE"], chunk["STARTSTATIONLONGITUDE"])
        end_in   = _in_box(chunk["ENDSTATIONLATITUDE"],   chunk["ENDSTATIONLONGITUDE"])
        mask = start_in | end_in
        if mask.any():
            sub = pd.DataFrame({
                "STARTTIMEMS":  chunk.loc[mask, "STARTTIMEMS"].to_numpy(),
                "is_departure": start_in[mask].to_numpy(),   # trip LEFT the zone
                "is_arrival":   end_in[mask].to_numpy(),      # trip ARRIVED in the zone
            })
            kept.append(sub)
    if not kept:
        return pd.DataFrame(columns=["clock_hour", "night_date", "is_departure", "is_arrival"])
    trips = pd.concat(kept, ignore_index=True)
    start = pd.to_datetime(trips["STARTTIMEMS"], unit="ms", utc=True).dt.tz_convert(TZ)
    trips["clock_hour"] = start.dt.hour
    trips["night_date"] = (start - pd.Timedelta(hours=4)).dt.date   # fold escape tail back
    return trips


def nightly_frame(trips, months=(6, 7, 8)):
    """One row per night (June-August by default) with windows + portable ratios."""
    def window(hours, prefix):
        sub = trips[trips["clock_hour"].isin(hours)]
        return sub.groupby("night_date").agg(
            **{f"{prefix}_rides":      ("clock_hour", "size"),
               f"{prefix}_departures": ("is_departure", "sum"),
               f"{prefix}_arrivals":   ("is_arrival", "sum")})

    idx = pd.date_range(min(trips["night_date"]), max(trips["night_date"]), freq="D")
    n = pd.concat([window(EVENING_HOURS, "eve"),
                   window(ESCAPE_HOURS, "escape"),
                   window(PRESHOW_HOURS, "pre")], axis=1).reindex(idx.date, fill_value=0)
    n.index = pd.to_datetime(n.index)
    n.index.name = "night"

    n["escape_netout"]     = n["escape_departures"] - n["escape_arrivals"]
    n["escape_share"]      = n["escape_rides"]  / n["eve_rides"].replace(0, np.nan)
    n["netout_share"]      = n["escape_netout"] / n["escape_rides"].replace(0, np.nan)
    n["pre_arrival_share"] = n["pre_arrivals"]  / n["eve_rides"].replace(0, np.nan)
    n["dow"]   = n.index.dayofweek
    n["month"] = n.index.month
    n = n[n["month"].isin(months)]
    return n


def add_robust_z(n):
    """Per-(month, weekday) robust z-score of escape-window volume.

    Computed over every night (events included) so it is label-free and therefore
    identical at train and predict time. A handful of event nights barely move a
    per-bucket median. NaN (singleton buckets / zero MAD) -> 0.
    """
    n = n.copy()
    grp = n.groupby(["month", "dow"])["escape_rides"]
    expected = grp.transform("median")
    mad = grp.transform(lambda x: (x - x.median()).abs().median())
    n["expected"] = expected
    n["robust_z"] = (n["escape_rides"] - expected) / (1.4826 * mad.replace(0, np.nan))
    n["robust_z"] = n["robust_z"].fillna(0.0)
    return n


def season_year(n):
    return int(n.index.year.value_counts().idxmax())


# --------------------------------------------------------------------------- #
# Weather (Open-Meteo historical archive, cached per year)                    #
# --------------------------------------------------------------------------- #
def fetch_weather(start_date, end_date, cache):
    cache = Path(cache)
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"]).set_index("date")
    import requests
    resp = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": LARONDE_LAT, "longitude": LARONDE_LON,
        "start_date": start_date, "end_date": end_date,
        "daily": "temperature_2m_max,precipitation_sum", "timezone": TZ}, timeout=30)
    resp.raise_for_status()
    p = resp.json()["daily"]
    w = pd.DataFrame({"date": pd.to_datetime(p["time"]),
                      "temp_max":  p["temperature_2m_max"],
                      "precip_mm": p["precipitation_sum"]}).set_index("date")
    w.to_csv(cache)
    return w


def build_features(csv_path, use_weather=True, cache_dir="output", months=(6, 7, 8)):
    """Full CSV -> nightly feature frame. Returns (frame_or_None, weather_ok)."""
    trips = load_zone_trips(csv_path)
    if trips.empty:
        return None, False
    n = add_robust_z(nightly_frame(trips, months=months))
    if n.empty:
        return None, False
    weather_ok = False
    if use_weather:
        year = season_year(n)
        try:
            w = fetch_weather(n.index.min().date().isoformat(),
                              n.index.max().date().isoformat(),
                              Path(cache_dir) / f"firework_weather_{year}.csv")
            n = n.join(w)
            weather_ok = True
        except Exception as e:                        # noqa: BLE001 -- degrade gracefully offline
            print(f"  [warn] weather fetch failed ({e}); continuing without weather",
                  file=sys.stderr)
    return n, weather_ok


# --------------------------------------------------------------------------- #
# Labels (from fireworks_dates.json -- the single source of truth)            #
# --------------------------------------------------------------------------- #
def load_fireworks_dates(path="fireworks_dates.json", confirmed_only=True):
    """Return {year: {date, ...}} for every season with shows.

    confirmed_only=True skips placeholder / unverified seasons (confirmed=false).
    """
    db = json.loads(Path(path).read_text())
    out = {}
    for year, season in db["seasons"].items():
        shows = season.get("shows") or []
        if shows and (season.get("confirmed") or not confirmed_only):
            out[int(year)] = {pd.Timestamp(s["date"]).date() for s in shows}
    return out


def label_series(index, fw_dates):
    """Boolean 'is_fireworks' for a DatetimeIndex given a set of show dates."""
    return index.to_series().dt.date.isin(set(fw_dates))


def feature_list(use_weather):
    return list(BASE_FEATURES) + (list(WEATHER_FEATURES) if use_weather else [])
