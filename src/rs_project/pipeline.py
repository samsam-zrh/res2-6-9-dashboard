from __future__ import annotations

import json

import joblib
import pandas as pd

from .config import (
    BEST_CLASSIFIER_PATH,
    BEST_FORECASTER_PATH,
    CLASSIFICATION_IMPORTANCE_PATH,
    CLASSIFICATION_METRICS_PATH,
    CLUSTER_PROFILE_PATH,
    CONSUMPTION_URL,
    CUSTOMER_FEATURES_PATH,
    DAILY_CONSUMPTION_PATH,
    FORECAST_METRICS_BY_TYPE_PATH,
    FORECAST_METRICS_PATH,
    FORECAST_PREDICTIONS_PATH,
    GENERATION_DAILY_PATH,
    GENERATION_METRICS_PATH,
    KMEANS_PATH,
    PCA_PROJECTION_PATH,
    PROFILE_TEMPLATES_PATH,
    RAW_LABELS_PATH,
    SCALER_PATH,
    SILHOUETTE_SCORES_PATH,
    SUMMARY_METRICS_PATH,
)
from .data import (
    copy_labels_file,
    download_balanced_exports,
    ensure_directories,
    load_labels,
    build_daily_consumption,
    build_profile_templates,
)
from .features import build_customer_features
from .generation import evaluate_generator
from .models import train_classifiers, train_clustering, train_forecasting_models


def run_pipeline() -> dict:
    # fonction principale du projet
    ensure_directories()
    copy_labels_file()

    labels = load_labels()
    selected_labels, raw_sources = download_balanced_exports(labels)
    daily = build_daily_consumption(raw_sources)
    customer_features, daily_with_flags = build_customer_features(daily)
    customer_features = customer_features.merge(selected_labels, on="id", how="left")

    (
        customer_features,
        scaler,
        kmeans,
        silhouette_df,
        cluster_profile,
        pca_df,
        clustering_summary,
    ) = train_clustering(customer_features)

    (
        classification_metrics,
        classification_importance,
        classifiers,
        best_classifier_name,
        classification_summary,
    ) = train_classifiers(customer_features)

    (
        forecast_predictions,
        forecast_metrics,
        forecast_metrics_by_type,
        forecasters,
        best_forecaster_name,
        forecasting_summary,
    ) = train_forecasting_models(daily, customer_features)

    generation_daily = daily.merge(
        selected_labels[["id", "reference_label"]],
        on="id",
        how="left",
    ).rename(columns={"reference_label": "customer_type"})
    generation_daily["date"] = pd.to_datetime(generation_daily["date"])
    generation_daily["is_weekend"] = generation_daily["date"].dt.dayofweek >= 5
    generation_daily["customer_type_name"] = generation_daily["customer_type"].map(
        {0: "RP", 1: "RS"}
    )

    profile_templates = build_profile_templates(
        raw_sources,
        daily,
        selected_labels[["id", "reference_label"]].rename(
            columns={"reference_label": "customer_type"}
        ),
        "customer_type",
    )
    generation_metrics = evaluate_generator(profile_templates, generation_daily)
    generation_metrics["customer_type_name"] = generation_metrics["customer_type"].map(
        {0: "RP", 1: "RS"}
    )

    daily_export = daily_with_flags.merge(
        customer_features[
            ["id", "reference_label", "reference_label_name", "cluster_label", "cluster_label_name"]
        ],
        on="id",
        how="left",
    )

    daily_export.to_csv(DAILY_CONSUMPTION_PATH, index=False)
    customer_features.to_csv(CUSTOMER_FEATURES_PATH, index=False)
    silhouette_df.to_csv(SILHOUETTE_SCORES_PATH, index=False)
    cluster_profile.to_csv(CLUSTER_PROFILE_PATH, index=False)
    pca_df.to_csv(PCA_PROJECTION_PATH, index=False)
    classification_metrics.to_csv(CLASSIFICATION_METRICS_PATH, index=False)
    classification_importance.to_csv(CLASSIFICATION_IMPORTANCE_PATH, index=False)
    forecast_predictions.to_csv(FORECAST_PREDICTIONS_PATH, index=False)
    forecast_metrics.to_csv(FORECAST_METRICS_PATH, index=False)
    forecast_metrics_by_type.to_csv(FORECAST_METRICS_BY_TYPE_PATH, index=False)
    profile_templates.to_csv(PROFILE_TEMPLATES_PATH, index=False)
    generation_daily.to_csv(GENERATION_DAILY_PATH, index=False)
    generation_metrics.to_csv(GENERATION_METRICS_PATH, index=False)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(kmeans, KMEANS_PATH)
    joblib.dump(classifiers[best_classifier_name], BEST_CLASSIFIER_PATH)
    if best_forecaster_name == "Seasonal naive":
        joblib.dump({"model": "lag_7"}, BEST_FORECASTER_PATH)
    else:
        joblib.dump(forecasters[best_forecaster_name], BEST_FORECASTER_PATH)

    summary = {
        "data": {
            "n_customers": int(customer_features["id"].nunique()),
            "n_daily_rows": int(daily_export.shape[0]),
            "label_balance_reference": customer_features["reference_label_name"]
            .value_counts()
            .sort_index()
            .to_dict(),
            "raw_consumption_path": CONSUMPTION_URL,
            "raw_labels_path": str(RAW_LABELS_PATH),
        },
        "clustering": clustering_summary,
        "classification": classification_summary,
        "forecasting": forecasting_summary,
        "generation": {
            "mean_profile_rmse": float(generation_metrics["profile_rmse"].mean()),
        },
    }
    SUMMARY_METRICS_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return summary
