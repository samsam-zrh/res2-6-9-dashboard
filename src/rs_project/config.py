from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RAW_CONSUMPTION_PATH = RAW_DIR / "RES2-6-9.csv"
RAW_LABELS_PATH = RAW_DIR / "RES2-6-9-labels.csv"
BALANCED_LABELS_PATH = RAW_DIR / "balanced_labels.csv"
BALANCED_CHUNKS_DIR = RAW_DIR / "balanced_chunks"

DAILY_CONSUMPTION_PATH = PROCESSED_DIR / "daily_consumption.csv"
CUSTOMER_FEATURES_PATH = PROCESSED_DIR / "customer_features.csv"
PCA_PROJECTION_PATH = PROCESSED_DIR / "pca_projection.csv"
SILHOUETTE_SCORES_PATH = PROCESSED_DIR / "silhouette_scores.csv"
CLUSTER_PROFILE_PATH = PROCESSED_DIR / "cluster_profiles.csv"
CLASSIFICATION_METRICS_PATH = PROCESSED_DIR / "classification_metrics.csv"
CLASSIFICATION_IMPORTANCE_PATH = PROCESSED_DIR / "classification_feature_importance.csv"
FORECAST_PREDICTIONS_PATH = PROCESSED_DIR / "forecast_predictions.csv"
FORECAST_METRICS_PATH = PROCESSED_DIR / "forecast_metrics.csv"
FORECAST_METRICS_BY_TYPE_PATH = PROCESSED_DIR / "forecast_metrics_by_type.csv"
PROFILE_TEMPLATES_PATH = PROCESSED_DIR / "profile_templates.csv"
GENERATION_DAILY_PATH = PROCESSED_DIR / "generation_daily_distribution.csv"
GENERATION_METRICS_PATH = PROCESSED_DIR / "generation_metrics.csv"

SUMMARY_METRICS_PATH = ARTIFACTS_DIR / "summary_metrics.json"
SCALER_PATH = ARTIFACTS_DIR / "clustering_scaler.joblib"
KMEANS_PATH = ARTIFACTS_DIR / "kmeans_model.joblib"
BEST_CLASSIFIER_PATH = ARTIFACTS_DIR / "best_classifier.joblib"
BEST_FORECASTER_PATH = ARTIFACTS_DIR / "best_forecaster.joblib"

LABEL_SOURCE_CANDIDATES = [
    Path(r"C:\Users\zerah\Downloads\RES2-6-9-labels - Copie.csv"),
    Path(r"C:\Users\zerah\Downloads\RES2-6-9-labels.csv"),
]

CONSUMPTION_URL = (
    "https://opendata.enedis.fr/api/explore/v2.1/catalog/datasets/"
    "courbes-de-charges-fictives-res2-6-9/exports/csv?timezone=Europe%2FParis"
)

RANDOM_STATE = 42
CHUNK_SIZE = 400_000

# Les valeurs brutes ressemblent a des watts. On convertit donc en kWh
# sur un pas de 30 minutes: W / 1000 * 0.5h.
POWER_TO_KWH_FACTOR = 0.5 / 1000.0

LABEL_NAME_MAP = {0: "RP", 1: "RS"}

CUSTOMER_FEATURE_COLUMNS = [
    "active_day_rate",
    "n_runs",
    "mean_run_len",
    "max_run_len",
    "mean_gap_len",
    "max_gap_len",
    "mean_daily_kwh",
    "p95_daily_kwh",
    "cv_daily_kwh",
    "active_rate_weekday",
    "active_rate_weekend",
    "mean_kwh_weekday",
    "mean_kwh_weekend",
    "winter_minus_summer",
    "seasonality_amp",
    "r_global",
    "r_mid",
    "r_summer",
    "r_winter",
]

FORECAST_FEATURE_COLUMNS = [
    "lag_1",
    "lag_2",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
    "is_weekend",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "cluster_label",
]
