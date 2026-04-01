from __future__ import annotations

import numpy as np
import pandas as pd


def _extract_profile(
    profile_templates: pd.DataFrame,
    customer_type: int,
    is_weekend: bool,
) -> tuple[np.ndarray, np.ndarray]:
    profile = profile_templates.loc[
        (profile_templates["customer_type"] == customer_type)
        & (profile_templates["is_weekend"] == is_weekend)
    ].sort_values("slot")

    if profile.empty:
        fallback = profile_templates.loc[
            profile_templates["customer_type"] == customer_type
        ].sort_values(["is_weekend", "slot"])
        profile = (
            fallback.groupby("slot", as_index=False)
            .agg(mean_share=("mean_share", "mean"), std_share=("std_share", "mean"))
            .sort_values("slot")
        )

    shares = profile["mean_share"].to_numpy(dtype=float)
    stds = profile["std_share"].to_numpy(dtype=float)

    if shares.size != 48 or np.isclose(shares.sum(), 0.0):
        shares = np.full(48, 1 / 48)
        stds = np.full(48, 0.0)
    elif np.allclose(stds, 0.0):
        stds = shares * 0.05

    shares = shares / shares.sum()
    return shares, stds


def _sample_daily_energy(
    generation_daily: pd.DataFrame,
    customer_type: int,
    is_weekend: bool,
    rng: np.random.Generator,
) -> float:
    values = generation_daily.loc[
        (generation_daily["customer_type"] == customer_type)
        & (generation_daily["is_weekend"] == is_weekend),
        "daily_kwh",
    ].to_numpy(dtype=float)
    if values.size == 0:
        values = generation_daily.loc[
            generation_daily["customer_type"] == customer_type,
            "daily_kwh",
        ].to_numpy(dtype=float)
    if values.size == 0:
        values = np.array([10.0], dtype=float)
    return float(rng.choice(values))


def generate_synthetic_daily_curves(
    profile_templates: pd.DataFrame,
    generation_daily: pd.DataFrame,
    customer_type: int,
    n_days: int,
    start_date: str | pd.Timestamp,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start_date).normalize()
    records = []

    for day_offset in range(n_days):
        current_day = start + pd.Timedelta(days=day_offset)
        is_weekend = current_day.dayofweek >= 5
        shares, stds = _extract_profile(profile_templates, customer_type, is_weekend)
        noise = rng.normal(0.0, stds * 0.25, size=48)
        noisy_shares = np.clip(shares + noise, 0.0, None)
        if np.isclose(noisy_shares.sum(), 0.0):
            noisy_shares = shares
        noisy_shares = noisy_shares / noisy_shares.sum()

        daily_kwh = _sample_daily_energy(generation_daily, customer_type, is_weekend, rng)
        step_kwh = noisy_shares * daily_kwh

        for slot, value in enumerate(step_kwh):
            timestamp = current_day + pd.Timedelta(minutes=30 * slot)
            records.append(
                {
                    "date": current_day,
                    "timestamp": timestamp,
                    "slot": slot,
                    "step_kwh": float(value),
                    "customer_type": customer_type,
                    "is_weekend": is_weekend,
                }
            )

    return pd.DataFrame.from_records(records)


def evaluate_generator(
    profile_templates: pd.DataFrame,
    generation_daily: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    results = []

    for customer_type in sorted(generation_daily["customer_type"].dropna().unique()):
        for is_weekend in [False, True]:
            real_energy = generation_daily.loc[
                (generation_daily["customer_type"] == customer_type)
                & (generation_daily["is_weekend"] == is_weekend),
                "daily_kwh",
            ].to_numpy(dtype=float)
            if real_energy.size == 0:
                continue

            shares, stds = _extract_profile(profile_templates, customer_type, is_weekend)
            synthetic_profiles = []
            synthetic_energy = []
            n_samples = min(30, real_energy.size)

            for _ in range(n_samples):
                noise = rng.normal(0.0, stds * 0.25, size=48)
                synthetic_share = np.clip(shares + noise, 0.0, None)
                if np.isclose(synthetic_share.sum(), 0.0):
                    synthetic_share = shares
                synthetic_share = synthetic_share / synthetic_share.sum()
                synthetic_profiles.append(synthetic_share)
                synthetic_energy.append(float(rng.choice(real_energy)))

            synthetic_profiles = np.vstack(synthetic_profiles)
            profile_rmse = float(
                np.sqrt(np.mean((synthetic_profiles.mean(axis=0) - shares) ** 2))
            )
            results.append(
                {
                    "customer_type": int(customer_type),
                    "is_weekend": bool(is_weekend),
                    "daily_mean_real": float(real_energy.mean()),
                    "daily_mean_synth": float(np.mean(synthetic_energy)),
                    "daily_std_real": float(real_energy.std(ddof=0)),
                    "daily_std_synth": float(np.std(synthetic_energy, ddof=0)),
                    "profile_rmse": profile_rmse,
                }
            )

    return pd.DataFrame(results)
