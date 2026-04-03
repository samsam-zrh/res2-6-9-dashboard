# Projet RES2-6-9

Petit projet fait a partir du jeu de donnees Enedis `RES2-6-9`.

Le but etait de :

- separer les residences principales (RP) et secondaires (RS),
- tester quelques modeles de classification,
- faire une prevision de conso journaliere,
- generer des courbes de charge simples.

J'ai garde une version assez simple pour que le projet tourne vite et soit facile a presenter.

## Ce qu'il y a dans le repo

- `app.py` : l'app Streamlit
- `scripts/train_pipeline.py` : script qui prepare les donnees et entraine les modeles
- `src/rs_project/` : le code Python du projet
- `data/processed/` : les fichiers deja prepares pour l'app
- `artifacts/` : quelques fichiers de sortie des modeles

## Idee generale

### Clustering

Je pars des consommations demi-horaires et je les transforme en conso par jour.
Ensuite je cree quelques variables simples :

- jours actifs
- longueur des periodes d'activite / inactivite
- conso moyenne
- difference semaine / week-end
- difference hiver / ete

Puis j'applique `KMeans` pour separer les profils.

### Classification

J'ai teste :

- regression logistique
- random forest
- MLP

Le but est de retrouver le type de client a partir des features calculees.

### Forecasting

Pour la prevision j'ai garde des modeles simples :

- baseline avec `lag_7`
- regression lineaire
- random forest

La cible est la conso journaliere.

### Generation

La generation est volontairement simple :

- on prend un profil moyen par type de client
- on tire une energie journaliere plausible
- on reconstruit une courbe demi-horaire

## Remarque sur les donnees

Je n'utilise pas tout le dataset brut dans le pipeline final.
Je prends un sous-jeu equilibre RP / RS pour que le test soit plus propre et pour que le projet tourne plus vite.

Pour l'energie, j'ai utilise :

`kWh = valeur / 1000 * 0.5`

car les valeurs ressemblent plutot a des watts sur un pas de 30 minutes.

## Installation

```powershell
python -m pip install -r requirements.txt
```

## Lancer le projet

### 1. Refaire le pipeline

```powershell
python scripts/train_pipeline.py
```

### 2. Lancer l'app

```powershell
streamlit run app.py
```

## Dans l'app

Il y a 4 parties :

- `Clustering`
- `Classification`
- `Forecasting`
- `Generation`

Le but est surtout de montrer les resultats de facon simple avec quelques graphes.
