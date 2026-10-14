# Data Science & Machine Learning Projects

Three separate projects, each first written as a Colab notebook (`notebooks/`) and
now packaged as runnable code under `src/`.

| Project | Package | Notebook | Data |
|---|---|---|---|
| **Air travel vs. holidays** (DSA2101): how air travel grew by region, and how much of it follows holiday counts | `src/air_travel` | `data_janitors_DSA2101_project.ipynb` | `data/DSA2101 Project data/` |
| **Human activity recognition** (CS3244): decision tree / random forest / XGBoost on smartphone sensor features | `src/har_classification` | `RandomForest_XGB_DT.ipynb` | `data/CS3244 project data/` |
| **Firmographic segmentation** (SDS datathon): cleaning ~8.5k company records and clustering them into segments | `src/firmographics` | `Randomforestnotrees_Datathon_Code.ipynb` | `data/SDS datathon data/` |

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use .venv/bin/activate elsewhere
pip install -r requirements.txt
```

## Running

Each project is a module with a CLI (`--help` lists options). Outputs go to
`artifacts/<project>/` by default.

```bash
python main.py air-travel
python main.py har
python main.py firmographics
```

`python -m src.air_travel` (and so on) runs the same thing.

### Air travel (`src/air_travel`)
`data.py` cleans and merges the TidyTuesday holiday and passenger tables (it
reads the local CSVs and falls back to the TidyTuesday URLs), `analysis.py`
builds the regional growth series, and `plots.py` draws the three report
figures (two PNGs and an interactive Plotly HTML).

### Human activity recognition (`src/har_classification`)
- `data.py` loads the UCI HAR layout (or the notebook's `features_*.csv`
  format via `--train-csv/--test-csv`) and builds the subject-grouped CV folds.
- `models.py` defines the pipeline: correlation filter, then a searched
  reduction step (PCA / FA / ICA / random projection / LDA / tree-based
  selection), then the classifier. It also holds both search grids.
- `train.py` runs the baseline comparison, stage 1 (choose the reduction step),
  stage 2 (tune the classifier) and the test-set evaluation.
- `wavelet.py` (`--wavelet`) adds wavelet features from the raw inertial
  signals. The notebook's original wavelet extractor was not committed, so
  this is a stand-in.

The full search fits many hundreds of pipelines per model and takes hours.
Use `--models rf` to run one model, and `--n-jobs -1` to parallelise.

### Firmographic segmentation (`src/firmographics`)
`data.py` (drop columns, de-duplicate), then `cleaning.py` (handle each kind of
missingness, with `industry.py` merging the five industry-code schemes), then
`outliers.py` (move multivariate outliers to an anomalies set), then
`features.py`, then `clustering.py`, then `analysis.py` (cluster profiles and
anomaly lift). `eda.py` holds the exploratory plots.

Clustering methods are chosen with `--methods`: `kprototypes` and `kmeans` run
by default; `gower` is quick; `hdbscan` runs an 800-combination grid search
that takes over an hour.

## Tests

```bash
python -m pytest
```
