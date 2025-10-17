# Champions League — Notebook-driven Unsupervised Learning

This repository contains an exploratory, notebook-first project that clusters UEFA Champions League players and teams using unsupervised learning (k-means + PCA). The analysis is driven from Jupyter notebooks located in `Champions_League/` and `Champions_League/notebooks/`.

The notebooks include step-by-step data cleaning, feature engineering (per-90 normalization and ratio creation), clustering (elbow method + KMeans), PCA visualization, automatic cluster labeling heuristics, and exports of cluster assignments.

This repo also includes a small, reusable programmatic pipeline implemented as `Champions_League/ucl_clustering_pipeline.py` which wraps the common preprocessing, feature selection, scaling and KMeans steps into functions you can call from scripts or notebooks.

## What's in this repo

- `Champions_League/` — main analysis folder
  - `ucl_kmeans_players.ipynb` — player clustering walkthrough (merge, scale, elbow, KMeans, PCA, export)
  - `data/` — CSV inputs used by the notebooks (attacking, attempts, defending, disciplinary, goals, goalkeeping, key_stats, etc.)
  - `notebooks/` — additional focused notebooks and generated CSV outputs (examples below)

Notebooks of note (open these to follow the exact code cells):
- `Champions_League/ucl_kmeans_players.ipynb` — full player clustering pipeline in notebook form
- `Champions_League/notebooks/players.ipynb` — context, cleaning, merging on `player_name` + `club`, selecting numeric features, standardization, elbow method, KMeans, PCA, and export of `champions_league_clusters.csv`
- `Champions_League/notebooks/Axe_1.ipynb` — offensive profiles workflow: loads attacking/attempts/goals/disciplinary/key_stats, normalizes per 90 minutes, creates offensive ratios, runs elbow and KMeans, produces `ucl_offensive_clusters.csv` and interactive PCA visualizations
- `Champions_League/notebooks/goalkeeping.ipynb` — (goalkeeping-focused analysis; open to inspect goalkeeper-specific features and clustering)
- `Champions_League/notebooks/defender.ipynb` and `teams.ipynb` — defensive and team-level analyses (profiling, normalized team stats)

Pipeline script (programmatic)
--------------------------------

If you prefer running the analysis programmatically (for reproducible runs or CI), use the pipeline in `Champions_League/ucl_clustering_pipeline.py`.

What it provides:
- Functions for cleaning and splitting goalkeepers vs field players
- Numeric feature filtering, variance/correlation-based pruning
- Scaling, silhouette-based k search, KMeans fit, and centroid interpretation helpers
- A single `cluster_players_pipeline(df, ...)` entrypoint that returns artifacts (scaled X, kmeans object, labeled DataFrame, centroid descriptions, etc.)

Example: run the pipeline from a Python script or notebook

```py
from pathlib import Path
import pandas as pd
from Champions_League.ucl_clustering_pipeline import cluster_players_pipeline, plot_metric_curve

# Load your merged dataframe (or build it from CSVs in Champions_League/data/)
df = pd.read_csv(Path("Champions_League/data/merged_players.csv"))  # or merge CSVs in the notebooks

results = cluster_players_pipeline(df,
                                   id_cols=["player_name","club","position"],
                                   position_col="position",
                                   correlation_threshold=0.90,
                                   k_min=2, k_max=10)

print("Best k:", results['best_k'])
plot_metric_curve(results['silhouette_scores'], title='Silhouette vs k')
results['clusters_df'].to_csv("players_clusters_from_pipeline.csv", index=False)
```

Notes:
- The pipeline is useful when you want a deterministic, scriptable run (for batch experiments or unit tests).
- Notebooks are still the recommended place to explore feature engineering choices and visualizations interactively.

Example exported CSVs you may find in `notebooks/`:
- `champions_league_clusters.csv` — merged player dataset enriched with cluster labels and PCA coordinates
- `ucl_offensive_clusters.csv` — offensive-clustering outputs with cluster labels and centroid summaries
- `team_stats_normalized.csv` — normalized team-level features used for team clustering or comparison

## Key notebook steps (common patterns)

These are the repeated, important steps implemented across the notebooks — useful when you want to reproduce or extend the work:

- Data loading: read CSVs from `Champions_League/data/` and print shapes/columns to confirm contents.
- Clean column names and join keys: lowercase, strip, replace spaces with underscores; normalize `player_name` and `club` before merging.
- Merge strategy: outer merges on `player_name` and `club` (sometimes `position`) with feature-prefixing to avoid name collisions.
- Missing values: fill logical NaNs (for example goalkeepers not having attacking stats) with 0 or use targeted imputation.
- Feature selection: keep numeric columns for clustering (select_dtypes), optionally drop identifiers.
- Per-90 normalization: when `minutes_played` (or similar) exists, compute per-90 rates (e.g., shots_per90) to make players comparable.
- Feature engineering: create ratios like shot accuracy (shots_on_target / total_shots), goals_per_shot, discipline_per_match, and other domain-specific features.
- Scaling: StandardScaler on the numeric matrix before KMeans / PCA.
- Choosing k: elbow method (plot inertias) over a range (e.g., 2..10) and optionally silhouette scores.
- Clustering: KMeans (fixed `random_state=42`) with `n_init='auto'` or a specified `n_init` for reproducibility.
- PCA visualization: reduce to 2 components, add `pca1` and `pca2` to the merged DataFrame, and plot colored by cluster.
- Cluster profiling: group means per cluster, display transposed profiles, and pick top discriminative features using variance/centroid magnitudes.
- Automated labeling: simple heuristics map centroid patterns to human-friendly labels (e.g., "Finisher", "Creator", "Aggressive / Indisciplined", "High-volume presser").
- Export: save enriched datasets to CSV (for example `champions_league_clusters.csv` and `ucl_offensive_clusters.csv`).

## Requirements

Python 3.8+ (3.10+ recommended). Typical libraries used by the notebooks:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- plotly (optional for interactive plots)
- jupyter / jupyterlab

If you prefer to track exact dependencies, install the packages and run `pip freeze > requirements.txt`.

## How to reproduce the analysis

1. Activate the virtual environment (see setup above).
2. Start Jupyter and open the notebooks:

```pwsh
jupyter lab
# or
jupyter notebook
```

3. Recommended order to run notebooks (top → bottom within each notebook):
- `Champions_League/notebooks/Axe_1.ipynb` (offensive feature engineering, per90, K selection)
- `Champions_League/ucl_kmeans_players.ipynb` or `Champions_League/notebooks/players.ipynb` (full merge, scaling, KMeans, PCA, export)
- `Champions_League/notebooks/goalkeeping.ipynb` and `Champions_League/notebooks/defender.ipynb` for role-specific cluster slices

Run each notebook cell-by-cell; many cells include printouts and shape checks to help you validate intermediate results.