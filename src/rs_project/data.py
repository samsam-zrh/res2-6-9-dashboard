import shutil
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import (
    ARTIFACTS_DIR,
    BALANCED_CHUNKS_DIR,
    BALANCED_LABELS_PATH,
    CHUNK_SIZE,
    DATA_DIR,
    LABEL_NAME_MAP,
    LABEL_SOURCE_CANDIDATES,
    POWER_TO_KWH_FACTOR,
    PROCESSED_DIR,
    RAW_DIR,
    RAW_LABELS_PATH,
)


def ensure_directories() -> None:
    for path in [DATA_DIR, RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR, BALANCED_CHUNKS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def copy_labels_file() -> Path:
    if RAW_LABELS_PATH.exists():
        return RAW_LABELS_PATH

    for candidate in LABEL_SOURCE_CANDIDATES:
        if candidate.exists():
            shutil.copy2(candidate, RAW_LABELS_PATH)
            return RAW_LABELS_PATH

    searched = "\n".join(str(path) for path in LABEL_SOURCE_CANDIDATES)
    raise FileNotFoundError(
        "Fichier de labels introuvable. J'ai cherche ici:\n"
        f"{searched}"
    )


def load_labels() -> pd.DataFrame:
    labels = pd.read_csv(RAW_LABELS_PATH)
    labels = labels.rename(
        columns={"label": "reference_label", "cluster": "reference_cluster"}
    )
    labels["id"] = labels["id"].astype("int64")
    labels["reference_label_name"] = labels["reference_label"].map(LABEL_NAME_MAP)
    return labels


def _read_raw_chunks(raw_source):
    if isinstance(raw_source, (str, Path)):
        sources = [raw_source]
    else:
        sources = list(raw_source)

    for source in sources:
        yield from pd.read_csv(
            source,
            sep=";",
            encoding="utf-8-sig",
            dtype={"id": "int64", "valeur": "float32"},
            chunksize=CHUNK_SIZE,
        )


def build_daily_consumption(raw_source) -> pd.DataFrame:
    daily_frames = []

    for chunk in _read_raw_chunks(raw_source):
        timestamps = pd.to_datetime(chunk["horodate"], errors="coerce", utc=True).dt.tz_convert(
            "Europe/Paris"
        )
        valid_mask = timestamps.notna()
        if not valid_mask.any():
            continue

        chunk = chunk.loc[valid_mask, ["id", "valeur"]].copy()
        chunk["date"] = timestamps.loc[valid_mask].dt.strftime("%Y-%m-%d")
        chunk["step_kwh"] = chunk["valeur"].astype("float32") * POWER_TO_KWH_FACTOR

        daily_chunk = (
            chunk.groupby(["id", "date"], as_index=False)
            .agg(daily_kwh=("step_kwh", "sum"))
        )
        daily_frames.append(daily_chunk)

    daily = pd.concat(daily_frames, ignore_index=True)
    daily = (
        daily.groupby(["id", "date"], as_index=False)
        .agg(daily_kwh=("daily_kwh", "sum"))
        .sort_values(["id", "date"])
        .reset_index(drop=True)
    )
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def build_profile_templates(
    raw_source,
    daily: pd.DataFrame,
    customer_types: pd.DataFrame,
    label_column: str,
) -> pd.DataFrame:
    daily_lookup = daily[["id", "date", "daily_kwh"]].copy()
    daily_lookup["date_key"] = pd.to_datetime(daily_lookup["date"]).dt.strftime("%Y-%m-%d")
    daily_lookup = daily_lookup[["id", "date_key", "daily_kwh"]]

    type_lookup = customer_types[["id", label_column]].drop_duplicates().copy()
    partial_frames = []

    for chunk in _read_raw_chunks(raw_source):
        timestamps = pd.to_datetime(chunk["horodate"], errors="coerce", utc=True).dt.tz_convert(
            "Europe/Paris"
        )
        valid_mask = timestamps.notna()
        if not valid_mask.any():
            continue

        ts = timestamps.loc[valid_mask]
        step = pd.DataFrame(
            {
                "id": chunk.loc[valid_mask, "id"].astype("int64").to_numpy(),
                "date_key": ts.dt.strftime("%Y-%m-%d").to_numpy(),
                "slot": (ts.dt.hour * 2 + ts.dt.minute // 30).astype("int16").to_numpy(),
                "is_weekend": (ts.dt.dayofweek >= 5).to_numpy(),
                "step_kwh": (
                    chunk.loc[valid_mask, "valeur"].astype("float32").to_numpy()
                    * POWER_TO_KWH_FACTOR
                ),
            }
        )

        merged = step.merge(daily_lookup, on=["id", "date_key"], how="left")
        merged = merged.merge(type_lookup, on="id", how="left")
        merged = merged.loc[merged["daily_kwh"] > 0].copy()
        if merged.empty:
            continue

        merged["share"] = merged["step_kwh"] / merged["daily_kwh"]
        merged["share_sq"] = merged["share"] ** 2

        partial = (
            merged.groupby([label_column, "is_weekend", "slot"], as_index=False)
            .agg(
                share_sum=("share", "sum"),
                share_sq_sum=("share_sq", "sum"),
                step_sum=("step_kwh", "sum"),
                n_obs=("share", "size"),
            )
        )
        partial_frames.append(partial)

    templates = pd.concat(partial_frames, ignore_index=True)
    templates = (
        templates.groupby([label_column, "is_weekend", "slot"], as_index=False)
        .agg(
            share_sum=("share_sum", "sum"),
            share_sq_sum=("share_sq_sum", "sum"),
            step_sum=("step_sum", "sum"),
            n_obs=("n_obs", "sum"),
        )
        .sort_values([label_column, "is_weekend", "slot"])
        .reset_index(drop=True)
    )

    templates["mean_share"] = templates["share_sum"] / templates["n_obs"].clip(lower=1)
    templates["mean_step_kwh"] = templates["step_sum"] / templates["n_obs"].clip(lower=1)
    mean_sq = templates["share_sq_sum"] / templates["n_obs"].clip(lower=1)
    templates["std_share"] = np.sqrt(np.maximum(mean_sq - templates["mean_share"] ** 2, 0.0))
    templates["type_name"] = templates[label_column].map(LABEL_NAME_MAP).fillna("Type")
    return templates


def download_balanced_exports(
    labels: pd.DataFrame,
    n_per_class: int = 60,
    group_size: int = 20,
) -> tuple[pd.DataFrame, list[Path]]:
    # on prend le meme nombre de RP et de RS
    counts = labels["reference_label"].value_counts()
    n_per_class = int(min(n_per_class, counts.min()))

    selected_parts = []
    for label_value in sorted(labels["reference_label"].unique()):
        group = labels.loc[labels["reference_label"] == label_value]
        selected_parts.append(group.sample(n=n_per_class, random_state=42))

    selected = (
        pd.concat(selected_parts, ignore_index=True)
        .sort_values(["reference_label", "id"])
        .reset_index(drop=True)
    )
    selected.to_csv(BALANCED_LABELS_PATH, index=False)

    chunk_paths = []
    base_url = (
        "https://opendata.enedis.fr/api/explore/v2.1/catalog/datasets/"
        "courbes-de-charges-fictives-res2-6-9/exports/csv"
    )

    ids = selected["id"].astype(str).tolist()
    for group_index, start in enumerate(range(0, len(ids), group_size)):
        group_ids = ids[start : start + group_size]
        chunk_path = BALANCED_CHUNKS_DIR / f"group_{group_index:02d}.csv"
        chunk_paths.append(chunk_path)
        if chunk_path.exists():
            continue

        where = "id in ({})".format(",".join(f'"{value}"' for value in group_ids))
        params = {
            "where": where,
            "timezone": "Europe/Paris",
        }
        url = base_url + "?" + urllib.parse.urlencode(params)
        response = requests.get(url, timeout=600)
        response.raise_for_status()
        chunk_path.write_bytes(response.content)

    return selected, chunk_paths
