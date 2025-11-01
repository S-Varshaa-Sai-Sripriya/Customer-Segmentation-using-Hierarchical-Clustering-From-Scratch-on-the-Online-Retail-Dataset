This repository contains a from-scratch implementation and supporting tooling for hierarchical
customer segmentation using agglomerative and divisive clustering techniques. The project uses
the Online Retail dataset (CSV) to demonstrate feature engineering, clustering, evaluation, and
visualization.

The goal is to provide clear, educational implementations of hierarchical clustering algorithms
and to build a reproducible pipeline: data -> RFM features -> scaling -> clustering -> evaluation.

## Repository structure

- `data/`
    - `online_retail.csv` — place the Online Retail CSV here (not included in repository).
- `src/` — core implementations
    - `agglomerative.py` — custom agglomerative clustering (single & average linkage)
    - `divisive.py` — divisive clustering (placeholder)
    - `distance_metrics.py` — Euclidean / Manhattan / Cosine distances + pairwise matrix
    - `feature_engineering.py` — RFM extraction and scaling utilities
    - `evaluation.py` — wrappers for silhouette and Davies–Bouldin scores
    - `visualization.py` — plotting utilities (placeholder)
    - `utils.py` — helpers (placeholder)
- `notebooks/`
    - `01_EDA_and_FeatureEngineering.ipynb` — small examples and demos (RFM + clustering)
- `tests/` — unit tests covering distance metrics, clustering, feature engineering, evaluation
- `eval/` — evaluation runner and demos
- `main.py` — project entrypoint (placeholder)
- `requirements.txt` — Python package list

## Current status (summary)

- Scaffolding, tests, and environment: COMPLETE
- Implemented modules: 4 / 7
    - Implemented: `distance_metrics`, `agglomerative`, `feature_engineering`, `evaluation`
    - Remaining: `divisive`, `visualization`, `utils`
- Tests added and run: 9 unit tests across 5 test modules (all passing in the project venv)

## Results (from local run)

These numbers were produced by running the test suite and a small evaluation demo on a synthetic
dataset (see `eval/evaluate.py`). Reproduce locally using the commands in the Setup section.

- Unit tests: 9 tests — all passing (OK)
- Evaluation demo (synthetic clusters):
    - Labels: [0 0 0 1 1 1]
    - Silhouette score: 0.9733
    - Davies–Bouldin index: 0.0312

Note: the evaluation demo uses a tiny synthetic dataset to demonstrate the metrics; results
on real data will vary and depend on preprocessing and chosen features.

## Quick start (Windows PowerShell)

1. Open a PowerShell window and change to the project root (where this README lives):

```powershell
cd "D:\Concepts\ML_Algorithm_Projects\11. Hierarchical Clustering (Agglomerative, Divisive)\hierarchical_customer_segmentation"
```

2. Activate the virtual environment (created at project root as `hclustering`):

```powershell
.\hclustering\Scripts\Activate.ps1
```

3. Install the project requirements into the venv (first time only):

```powershell
pip install -r .\requirements.txt
```

4. Run the tests (recommended; run inside the activated venv or use the venv python):

```powershell
# If venv is active
python -m unittest discover -s .\tests -v

# OR call the venv python directly (no activation required)
.\hclustering\Scripts\python.exe -m unittest discover -s .\tests -v
```

5. Run the evaluation demo (prints silhouette & Davies–Bouldin on a synthetic example):

```powershell
# easiest: use the venv python and ensure project root is on sys.path
.\hclustering\Scripts\python.exe -c "import sys; sys.path.insert(0, r'.'); from eval.evaluate import run_evaluation_demo; run_evaluation_demo()"
```

## How to contribute / next work items

- Implement `divisive.py` for divisive (top-down) clustering.
- Implement `visualization.py` (dendrograms, cluster scatter plots) and add plotting cells to
    the notebook using matplotlib / seaborn.
- Harden `feature_engineering.py` to handle missing values and add more automated tests.
- Add more real-data notebooks that demonstrate the full end-to-end pipeline on
    `data/online_retail.csv` (once you add the dataset).

## Reproducibility notes

- The project includes a venv named `hclustering` at the project root. If you recreate the
    environment, use the `requirements.txt` above. If PowerShell blocks activation scripts, you
    may need to run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## License

This project is provided for educational purposes. Add an appropriate open-source license file
if you plan to publish or share publicly.

## Contact

If you want me to continue, I can implement `divisive.py` next or add visualization utilities and
plots to the notebook — tell me which you prefer and I will proceed.

