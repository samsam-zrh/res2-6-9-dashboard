from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CUSTOMER_FEATURE_COLUMNS, FORECAST_FEATURE_COLUMNS, LABEL_NAME_MAP, RANDOM_STATE


def _classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _extract_model_importance(
    model_name: str,
    fitted_model: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    estimator = fitted_model
    if isinstance(fitted_model, Pipeline):
        estimator = fitted_model.steps[-1][1]

    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importance = np.abs(np.ravel(estimator.coef_))
    else:
        return pd.DataFrame(columns=["model", "feature", "importance"])

    return (
        pd.DataFrame(
            {
                "model": model_name,
                "feature": feature_names,
                "importance": importance,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def train_clustering(customer_features: pd.DataFrame):
    work = customer_features.copy()
    X = work[CUSTOMER_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    silhouette_rows = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=30, random_state=RANDOM_STATE)
        labels = model.fit_predict(X_scaled)
        silhouette_rows.append(
            {"k": k, "silhouette_score": float(silhouette_score(X_scaled, labels))}
        )

    kmeans = KMeans(n_clusters=2, n_init=50, random_state=RANDOM_STATE)
    clusters = kmeans.fit_predict(X_scaled)

    centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=CUSTOMER_FEATURE_COLUMNS,
    )
    rs_cluster = (
        centers.sort_values(
            ["active_day_rate", "max_gap_len", "seasonality_amp"],
            ascending=[True, False, False],
        )
        .index[0]
    )
    cluster_to_label = {int(rs_cluster): 1}
    for cluster_id in range(kmeans.n_clusters):
        cluster_to_label.setdefault(cluster_id, 0)

    work["cluster"] = clusters.astype(int)
    work["cluster_label"] = [cluster_to_label[int(cluster)] for cluster in clusters]
    work["cluster_label_name"] = work["cluster_label"].map(LABEL_NAME_MAP)

    silhouette_df = pd.DataFrame(silhouette_rows)
    cluster_profile = (
        work.groupby("cluster_label_name", as_index=False)[CUSTOMER_FEATURE_COLUMNS].mean()
    )

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    pca_df = work[
        ["id", "reference_label", "reference_label_name", "cluster_label", "cluster_label_name"]
    ].copy()
    pca_df["PC1"] = coords[:, 0]
    pca_df["PC2"] = coords[:, 1]

    clustering_summary = {
        "cluster_to_label": cluster_to_label,
        "silhouette_selected_k2": float(
            silhouette_df.loc[silhouette_df["k"] == 2, "silhouette_score"].iloc[0]
        ),
    }
    if "reference_label" in work.columns:
        clustering_summary["metrics_vs_reference"] = _classification_metrics(
            work["reference_label"], work["cluster_label"]
        )
        clustering_summary["confusion_matrix_vs_reference"] = confusion_matrix(
            work["reference_label"], work["cluster_label"]
        ).tolist()

    return work, scaler, kmeans, silhouette_df, cluster_profile, pca_df, clustering_summary


def train_classifiers(customer_features: pd.DataFrame):
    work = customer_features.copy().reset_index(drop=True)
    X = work[CUSTOMER_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_target = work["cluster_label"].astype(int)
    y_reference = work["reference_label"].astype(int)

    counts = y_reference.value_counts()
    n_test_per_class = min(max(12, int(counts.min() * 0.25)), int(counts.min() - 1))

    test_indices = []
    for _, group in work.groupby("reference_label"):
        sampled = group.sample(n=n_test_per_class, random_state=RANDOM_STATE).index.tolist()
        test_indices.extend(sampled)

    test_mask = work.index.isin(test_indices)
    train_mask = ~test_mask

    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train = y_target.loc[train_mask]
    y_test_target = y_target.loc[test_mask]
    y_test_reference = y_reference.loc[test_mask]

    models = {
        "Logistic regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2_000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(32, 16),
                        max_iter=1_500,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    metric_rows = []
    importance_frames = []
    fitted_models = {}
    best_model_name = None
    best_reference_f1 = -1.0
    best_confusion_reference = None

    for model_name, model in models.items():
        fitted = model.fit(X_train, y_train)
        preds = fitted.predict(X_test)
        fitted_models[model_name] = fitted

        target_metrics = _classification_metrics(y_test_target, preds)
        reference_metrics = _classification_metrics(y_test_reference, preds)
        metric_rows.append(
            {
                "model": model_name,
                **{f"target_{key}": value for key, value in target_metrics.items()},
                **{f"reference_{key}": value for key, value in reference_metrics.items()},
            }
        )
        importance_frames.append(
            _extract_model_importance(model_name, fitted, CUSTOMER_FEATURE_COLUMNS)
        )

        if reference_metrics["f1"] > best_reference_f1:
            best_reference_f1 = reference_metrics["f1"]
            best_model_name = model_name
            best_confusion_reference = confusion_matrix(y_test_reference, preds).tolist()

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        "reference_f1", ascending=False
    ).reset_index(drop=True)
    non_empty_importances = [frame for frame in importance_frames if not frame.empty]
    importance_df = (
        pd.concat(non_empty_importances, ignore_index=True)
        if non_empty_importances
        else pd.DataFrame(columns=["model", "feature", "importance"])
    )

    summary = {
        "best_model_name": best_model_name,
        "best_confusion_matrix_vs_reference": best_confusion_reference,
        "balanced_test_size": int(test_mask.sum()),
    }

    return metrics_df, importance_df, fitted_models, best_model_name, summary


def train_forecasting_models(daily: pd.DataFrame, customer_features: pd.DataFrame):
    work = daily[["id", "date", "daily_kwh"]].copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.merge(
        customer_features[["id", "cluster_label", "reference_label"]],
        on="id",
        how="left",
    )
    work["customer_type_name"] = work["cluster_label"].map(LABEL_NAME_MAP)
    work = work.sort_values(["id", "date"]).reset_index(drop=True)

    grouped = work.groupby("id")["daily_kwh"]
    work["lag_1"] = grouped.shift(1)
    work["lag_2"] = grouped.shift(2)
    work["lag_7"] = grouped.shift(7)
    work["lag_14"] = grouped.shift(14)
    work["rolling_mean_7"] = work.groupby("id")["daily_kwh"].transform(
        lambda s: s.shift(1).rolling(7).mean()
    )
    work["rolling_std_7"] = work.groupby("id")["daily_kwh"].transform(
        lambda s: s.shift(1).rolling(7).std()
    )
    work["rolling_mean_14"] = work.groupby("id")["daily_kwh"].transform(
        lambda s: s.shift(1).rolling(14).mean()
    )

    work["dow"] = work["date"].dt.dayofweek
    work["month"] = work["date"].dt.month
    work["is_weekend"] = (work["dow"] >= 5).astype(int)
    work["dow_sin"] = np.sin(2 * np.pi * work["dow"] / 7)
    work["dow_cos"] = np.cos(2 * np.pi * work["dow"] / 7)
    work["month_sin"] = np.sin(2 * np.pi * work["month"] / 12)
    work["month_cos"] = np.cos(2 * np.pi * work["month"] / 12)

    dataset = work.dropna(subset=FORECAST_FEATURE_COLUMNS).copy()
    unique_dates = np.sort(dataset["date"].unique())
    cutoff = unique_dates[int(len(unique_dates) * 0.8)]

    train_mask = dataset["date"] <= cutoff
    test_mask = dataset["date"] > cutoff

    X_train = dataset.loc[train_mask, FORECAST_FEATURE_COLUMNS]
    y_train = dataset.loc[train_mask, "daily_kwh"]
    X_test = dataset.loc[test_mask, FORECAST_FEATURE_COLUMNS]
    y_test = dataset.loc[test_mask, "daily_kwh"]

    models = {
        "Linear regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "Random forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    predictions = dataset.loc[
        test_mask,
        ["id", "date", "daily_kwh", "cluster_label", "reference_label", "customer_type_name"],
    ].copy()
    predictions["Seasonal naive"] = dataset.loc[test_mask, "lag_7"].to_numpy()

    metric_rows = [
        {"model": "Seasonal naive", **_regression_metrics(y_test, predictions["Seasonal naive"])}
    ]
    by_type_rows = []
    fitted_models = {}
    best_model_name = "Seasonal naive"
    best_rmse = metric_rows[0]["rmse"]

    for customer_type, group in predictions.groupby("customer_type_name"):
        by_type_rows.append(
            {
                "model": "Seasonal naive",
                "customer_type_name": customer_type,
                **_regression_metrics(group["daily_kwh"], group["Seasonal naive"]),
            }
        )

    for model_name, model in models.items():
        fitted = model.fit(X_train, y_train)
        preds = fitted.predict(X_test)
        predictions[model_name] = preds
        fitted_models[model_name] = fitted

        scores = _regression_metrics(y_test, preds)
        metric_rows.append({"model": model_name, **scores})

        if scores["rmse"] < best_rmse:
            best_rmse = scores["rmse"]
            best_model_name = model_name

        for customer_type, group in predictions.groupby("customer_type_name"):
            by_type_rows.append(
                {
                    "model": model_name,
                    "customer_type_name": customer_type,
                    **_regression_metrics(group["daily_kwh"], group[model_name]),
                }
            )

    metrics_df = pd.DataFrame(metric_rows).sort_values("rmse").reset_index(drop=True)
    by_type_df = pd.DataFrame(by_type_rows).sort_values(
        ["model", "customer_type_name"]
    ).reset_index(drop=True)

    summary = {
        "best_model_name": best_model_name,
        "cutoff_date": str(pd.Timestamp(cutoff).date()),
    }

    return predictions, metrics_df, by_type_df, fitted_models, best_model_name, summary
