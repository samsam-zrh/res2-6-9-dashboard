# Projet RES2-6-9

Projet Python simple et fonctionnel construit autour du cas d'usage Enedis `RES2-6-9` pour distinguer les residences principales (RP) et les residences secondaires (RS), puis comparer plusieurs approches de classification, de prevision et de generation.

Pour garder un projet pedagogique, rapide a executer et coherent avec la consigne d'un jeu de test equilibre, le pipeline travaille sur un sous-jeu balance RS/RP telecharge par paquets d'IDs.

## Contenu

- `scripts/train_pipeline.py` : lance tout le pipeline.
- `app.py` : dashboard Streamlit.
- `src/rs_project/` : modules de preparation des donnees, features, modeles et generation.
- `data/raw/` : donnees sources.
- `data/processed/` : tables pretes pour l'application.
- `artifacts/` : modeles et resume des metriques.

## Methode

### 1. Clustering

On agrege les donnees demi-horaires en energie journaliere et on construit des features metier inspirees du notebook de cours :

- taux de jours actifs ;
- longueur des sequences d'activite / d'inactivite ;
- niveau moyen et variabilite de consommation ;
- comportement semaine / weekend ;
- saisonnalite hiver / ete / mi-saison.

On applique ensuite `KMeans` avec `k=2` pour obtenir une segmentation directement exploitable en RP / RS.

### 2. Classification

On entraine plusieurs modeles supervises pour reproduire le label obtenu au clustering :

- regression logistique ;
- random forest ;
- MLP.

L'evaluation est faite sur un jeu de test equilibre RP / RS, avec comparaison supplementaire aux labels de reference fournis.

### 3. Forecasting

La prevision est faite sur l'energie quotidienne a partir de l'historique recent :

- baseline saisonniere `lag_7` ;
- regression lineaire ;
- random forest.

Les features sont des retards (`lag_1`, `lag_7`, `lag_14`), moyennes glissantes et variables calendaires.

### 4. Generation

Le generateur est volontairement simple :

- on apprend un profil moyen demi-horaire par type de client ;
- on echantillonne une energie journaliere realiste depuis l'historique ;
- on ajoute un leger bruit controle a la forme de profil.

Cette approche reste pedagogique, coherente et facile a expliquer en soutenance.

## Point important sur l'unite

Le notebook HTML d'origine multiplie directement `valeur * 0.5`. En pratique, les ordres de grandeur du CSV officiel Enedis ressemblent a des watts et non a des kW. Dans ce projet, on convertit donc l'energie comme suit :

`kWh = valeur / 1000 * 0.5`

Ce choix donne des consommations journalieres realistes pour des clients residentiels 6-9 kVA.

## Installation

```powershell
python -m pip install -r requirements.txt
```

## Execution

### 1. Lancer le pipeline

```powershell
python scripts/train_pipeline.py
```

Le script :

- copie le fichier de labels depuis `Downloads` ;
- telecharge un sous-jeu equilibre RS/RP depuis l'export officiel Enedis, par paquets d'IDs ;
- construit les tables journalieres et les features ;
- entraine les modeles ;
- sauvegarde les metriques et les artefacts.

### 2. Lancer le dashboard

```powershell
streamlit run app.py
```

## Resultats attendus dans l'application

- onglet `Clustering` : PCA, silhouette, matrice de confusion ;
- onglet `Classification` : comparaison des modeles et importance des features ;
- onglet `Forecasting` : metriques et courbes reelles vs predites ;
- onglet `Generation` : courbes synthetiques conditionnelles RS / RP.
