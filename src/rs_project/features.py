import numpy as np
import pandas as pd


def _season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (6, 7, 8):
        return "summer"
    return "mid"


def _runs_and_gaps(active_series: pd.Series) -> pd.Series:
    runs = []
    gaps = []
    run = 0
    gap = 0

    for is_active in active_series.astype(bool):
        if is_active:
            run += 1
            if gap > 0:
                gaps.append(gap)
                gap = 0
        else:
            gap += 1
            if run > 0:
                runs.append(run)
                run = 0

    if run > 0:
        runs.append(run)
    if gap > 0:
        gaps.append(gap)

    return pd.Series(
        {
            "n_runs": float(len(runs)),
            "mean_run_len": float(np.mean(runs)) if runs else 0.0,
            "max_run_len": float(np.max(runs)) if runs else 0.0,
            "mean_gap_len": float(np.mean(gaps)) if gaps else 0.0,
            "max_gap_len": float(np.max(gaps)) if gaps else 0.0,
        }
    )


def build_customer_features(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = daily[["id", "date", "daily_kwh"]].copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values(["id", "date"]).reset_index(drop=True)

    thresholds = (
        work.loc[work["daily_kwh"] > 0]
        .groupby("id")["daily_kwh"]
        .quantile(0.2)
        .rename("th_pdl")
        .reset_index()
    )
    work = work.merge(thresholds, on="id", how="left")
    work["is_active_day"] = (work["daily_kwh"] >= work["th_pdl"]).fillna(False)

    activity = (
        work.groupby("id", as_index=False)
        .agg(
            n_days=("id", "size"),
            n_active_days=("is_active_day", "sum"),
            active_day_rate=("is_active_day", "mean"),
            mean_daily_kwh=("daily_kwh", "mean"),
            p95_daily_kwh=("daily_kwh", lambda s: s.quantile(0.95)),
            cv_daily_kwh=(
                "daily_kwh",
                lambda s: (s.std() / s.mean()) if s.mean() else 0.0,
            ),
        )
    )

    runs_stats = (
        work.sort_values(["id", "date"])
        .groupby("id")["is_active_day"]
        .apply(_runs_and_gaps)
        .unstack()
        .reset_index()
    )

    work["dow"] = work["date"].dt.dayofweek
    work["is_weekend"] = work["dow"] >= 5

    week_pattern = (
        work.groupby(["id", "is_weekend"], as_index=False)
        .agg(
            active_rate=("is_active_day", "mean"),
            mean_kwh=("daily_kwh", "mean"),
        )
        .pivot(index="id", columns="is_weekend")
    )
    week_pattern.columns = [
        f"{metric}_{'weekend' if is_weekend else 'weekday'}"
        for metric, is_weekend in week_pattern.columns
    ]
    week_pattern = week_pattern.reset_index()
    for column in [
        "active_rate_weekday",
        "active_rate_weekend",
        "mean_kwh_weekday",
        "mean_kwh_weekend",
    ]:
        if column not in week_pattern.columns:
            week_pattern[column] = 0.0

    seasonal = work.copy()
    seasonal["month"] = seasonal["date"].dt.month
    seasonal["season"] = seasonal["month"].map(_season_from_month)

    season_stats = (
        seasonal.groupby(["id", "season"], as_index=False)
        .agg(mean_daily_kwh=("daily_kwh", "mean"))
        .pivot(index="id", columns="season", values="mean_daily_kwh")
        .reset_index()
    )
    for column in ["winter", "summer", "mid"]:
        if column not in season_stats.columns:
            season_stats[column] = 0.0

    global_mean = (
        seasonal.groupby("id", as_index=False)
        .agg(mean_daily_kwh_global=("daily_kwh", "mean"))
    )
    season_stats = season_stats.merge(global_mean, on="id", how="left")
    eps = 1e-9
    season_stats["r_global"] = 1.0
    season_stats["r_mid"] = season_stats["mid"] / (season_stats["mean_daily_kwh_global"] + eps)
    season_stats["r_summer"] = season_stats["summer"] / (season_stats["mean_daily_kwh_global"] + eps)
    season_stats["r_winter"] = season_stats["winter"] / (season_stats["mean_daily_kwh_global"] + eps)
    season_stats = season_stats[["id", "r_global", "r_mid", "r_summer", "r_winter"]]

    features = activity.merge(runs_stats, on="id", how="left")
    features = features.merge(week_pattern, on="id", how="left")
    features = features.merge(season_stats, on="id", how="left")

    features["seasonality_amp"] = (
        features[["r_mid", "r_summer", "r_winter"]].max(axis=1)
        - features[["r_mid", "r_summer", "r_winter"]].min(axis=1)
    )
    features["winter_minus_summer"] = features["r_winter"] - features["r_summer"]
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return features, work
