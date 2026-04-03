import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rs_project.config import (
    CLASSIFICATION_IMPORTANCE_PATH,
    CLASSIFICATION_METRICS_PATH,
    CLUSTER_PROFILE_PATH,
    CUSTOMER_FEATURES_PATH,
    DAILY_CONSUMPTION_PATH,
    FORECAST_METRICS_BY_TYPE_PATH,
    FORECAST_METRICS_PATH,
    FORECAST_PREDICTIONS_PATH,
    GENERATION_DAILY_PATH,
    GENERATION_METRICS_PATH,
    PCA_PROJECTION_PATH,
    PROFILE_TEMPLATES_PATH,
    SILHOUETTE_SCORES_PATH,
    SUMMARY_METRICS_PATH,
)
from rs_project.generation import generate_synthetic_daily_curves

st.set_page_config(page_title="Projet RES2-6-9", page_icon="⚡", layout="wide")


def _load_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        st.error(
            "Les artefacts du projet sont absents. Lance d'abord "
            "`python scripts/train_pipeline.py`."
        )
        st.stop()
    return pd.read_csv(path, parse_dates=parse_dates)


@st.cache_data
def load_assets():
    if not SUMMARY_METRICS_PATH.exists():
        st.error(
            "Le pipeline n'a pas encore ete execute. Lance "
            "`python scripts/train_pipeline.py`."
        )
        st.stop()

    summary = json.loads(SUMMARY_METRICS_PATH.read_text(encoding="utf-8"))
    return {
        "summary": summary,
        "daily": _load_csv(DAILY_CONSUMPTION_PATH, parse_dates=["date"]),
        "features": _load_csv(CUSTOMER_FEATURES_PATH),
        "silhouette": _load_csv(SILHOUETTE_SCORES_PATH),
        "cluster_profile": _load_csv(CLUSTER_PROFILE_PATH),
        "pca": _load_csv(PCA_PROJECTION_PATH),
        "classification_metrics": _load_csv(CLASSIFICATION_METRICS_PATH),
        "classification_importance": _load_csv(CLASSIFICATION_IMPORTANCE_PATH),
        "forecast_predictions": _load_csv(FORECAST_PREDICTIONS_PATH, parse_dates=["date"]),
        "forecast_metrics": _load_csv(FORECAST_METRICS_PATH),
        "forecast_metrics_by_type": _load_csv(FORECAST_METRICS_BY_TYPE_PATH),
        "profile_templates": _load_csv(PROFILE_TEMPLATES_PATH),
        "generation_daily": _load_csv(GENERATION_DAILY_PATH, parse_dates=["date"]),
        "generation_metrics": _load_csv(GENERATION_METRICS_PATH),
    }


def render_confusion_matrix(matrix: list[list[int]], title: str):
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Pred RP", "Pred RS"],
            y=["True RP", "True RS"],
            text=matrix,
            texttemplate="%{text}",
            colorscale="Blues",
        )
    )
    fig.update_layout(title=title, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)


assets = load_assets()
summary = assets["summary"]

page = st.sidebar.radio(
    "Rubrique",
    ["Accueil", "Clustering", "Classification", "Forecasting", "Generation"],
)

st.sidebar.markdown("Projet simplifie RES2-6-9")
st.sidebar.markdown("Source: Enedis Open Data + labels de reference")

if page == "Accueil":
    st.title("Projet IA sur les courbes de charge RES2-6-9")
    st.write(
        "Petite app pour presenter le projet avec les 4 parties demandees : "
        "clustering, classification, forecasting et generation."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clients", summary["data"]["n_customers"])
    col2.metric("Jours agreges", summary["data"]["n_daily_rows"])
    col3.metric(
        "F1 clustering",
        f"{summary['clustering']['metrics_vs_reference']['f1']:.3f}",
    )
    best_forecast_rmse = float(assets["forecast_metrics"].iloc[0]["rmse"])
    col4.metric("Meilleur RMSE forecast", f"{best_forecast_rmse:.3f}")

    st.subheader("Jeu de donnees")
    label_balance = pd.Series(summary["data"]["label_balance_reference"]).rename("count")
    st.bar_chart(label_balance)

    st.subheader("Resume rapide")
    st.markdown(
        "- Clustering: KMeans a 2 clusters sur des features metier issues des consommations journalieres.\n"
        "- Classification: regression logistique, random forest et MLP pour reproduire le label issu du clustering.\n"
        "- Forecasting: baseline saisonniere, regression lineaire et random forest sur l'energie quotidienne.\n"
        "- Generation: generateur conditionnel simple base sur des profils moyens demi-horaires et une distribution empirique d'energie."
    )

if page == "Clustering":
    st.title("1. Clustering RS / RP")

    st.subheader("Score de silhouette")
    fig = px.bar(
        assets["silhouette"],
        x="k",
        y="silhouette_score",
        text_auto=".3f",
        color="silhouette_score",
        color_continuous_scale="Tealgrn",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Projection PCA")
    fig = px.scatter(
        assets["pca"],
        x="PC1",
        y="PC2",
        color="cluster_label_name",
        symbol="reference_label_name",
        hover_data=["id"],
        title="Couleurs = clustering, symboles = labels de reference",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Matrice de confusion vs reference")
    render_confusion_matrix(
        summary["clustering"]["confusion_matrix_vs_reference"],
        "Clustering vs labels de reference",
    )

    st.subheader("Profil moyen par type predit")
    st.dataframe(assets["cluster_profile"], use_container_width=True)

if page == "Classification":
    st.title("2. Classification supervisee")
    st.write(
        "Ici les modeles essayent de retrouver le type de client a partir des features."
    )

    st.subheader("Comparaison des modeles")
    st.dataframe(assets["classification_metrics"], use_container_width=True)

    fig = px.bar(
        assets["classification_metrics"],
        x="model",
        y="reference_f1",
        text_auto=".3f",
        color="reference_f1",
        color_continuous_scale="Bluered",
        title="F1-score sur le jeu de test equilibre",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Matrice de confusion du meilleur modele")
    render_confusion_matrix(
        summary["classification"]["best_confusion_matrix_vs_reference"],
        f"Meilleur modele: {summary['classification']['best_model_name']}",
    )

    st.subheader("Importance des variables")
    model_name = st.selectbox(
        "Modele pour l'importance",
        assets["classification_importance"]["model"].drop_duplicates().tolist(),
    )
    importance = assets["classification_importance"].loc[
        assets["classification_importance"]["model"] == model_name
    ].head(12)
    fig = px.bar(
        importance.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        title=f"Variables les plus influentes - {model_name}",
    )
    st.plotly_chart(fig, use_container_width=True)

if page == "Forecasting":
    st.title("3. Forecasting de la consommation quotidienne")

    st.subheader("Metriques globales")
    st.dataframe(assets["forecast_metrics"], use_container_width=True)

    st.subheader("Metriques par type de client")
    st.dataframe(assets["forecast_metrics_by_type"], use_container_width=True)

    st.subheader("Visualisation sur un client")
    customer_ids = sorted(assets["forecast_predictions"]["id"].unique().tolist())
    selected_id = st.selectbox("Client", customer_ids)

    view = assets["forecast_predictions"].loc[
        assets["forecast_predictions"]["id"] == selected_id
    ].sort_values("date")
    model_columns = [
        column
        for column in view.columns
        if column in ["Seasonal naive", "Linear regression", "Random forest"]
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=view["date"],
            y=view["daily_kwh"],
            mode="lines+markers",
            name="Reel",
        )
    )
    for column in model_columns:
        fig.add_trace(
            go.Scatter(
                x=view["date"],
                y=view[column],
                mode="lines",
                name=column,
            )
        )
    fig.update_layout(title=f"Prevision quotidienne - client {selected_id}")
    st.plotly_chart(fig, use_container_width=True)

if page == "Generation":
    st.title("4. Generation conditionnelle de courbes")
    st.write(
        "Le generateur reprend une forme de courbe moyenne et une energie journaliere plausible."
    )

    type_name = st.selectbox("Type de client", ["RP", "RS"])
    type_id = 0 if type_name == "RP" else 1
    n_days = st.slider("Horizon (jours)", min_value=7, max_value=60, value=14)
    seed = st.number_input("Seed", min_value=0, value=42, step=1)
    start_date = st.date_input("Date de depart")

    synthetic = generate_synthetic_daily_curves(
        assets["profile_templates"],
        assets["generation_daily"],
        customer_type=type_id,
        n_days=n_days,
        start_date=pd.Timestamp(start_date),
        seed=int(seed),
    )

    daily_energy = (
        synthetic.groupby("date", as_index=False)
        .agg(daily_kwh=("step_kwh", "sum"))
        .sort_values("date")
    )
    fig = px.bar(daily_energy, x="date", y="daily_kwh", title="Energie synthetique par jour")
    st.plotly_chart(fig, use_container_width=True)

    mean_profile = (
        synthetic.groupby("slot", as_index=False)
        .agg(step_kwh=("step_kwh", "mean"))
        .sort_values("slot")
    )
    fig = px.line(
        mean_profile,
        x="slot",
        y="step_kwh",
        markers=True,
        title="Profil demi-horaire moyen genere",
    )
    fig.update_xaxes(title="Creneau demi-horaire (0-47)")
    fig.update_yaxes(title="kWh par pas")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Evaluation du generateur")
    st.dataframe(assets["generation_metrics"], use_container_width=True)
