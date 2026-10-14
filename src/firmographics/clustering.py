"""Clustering methods: K-Prototypes (main model), K-Means, HDBSCAN, and
hierarchical clustering on Gower distances."""

from itertools import product

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.firmographics import config


class KPrototypesPreprocessor:
    """Builds the mixed-type matrix K-Prototypes expects.

    Numeric columns are log1p-transformed, robust-scaled and clipped;
    categorical columns are cast to ``category`` and their positions recorded.
    ``fit_transform`` on the majority set, then ``transform`` on new records
    (e.g. the anomalies) so both share columns, order and scaling.
    """

    def _base(self, feature_table: pd.DataFrame) -> pd.DataFrame:
        m = feature_table.drop(columns=config.KPROTO_DROP_COLS, errors="ignore")
        m["Ticker"] = (m["Ticker"] > 0).astype(int)  # listed vs. private
        return m

    def fit_transform(self, feature_table: pd.DataFrame) -> pd.DataFrame:
        m = self._base(feature_table)
        self.numeric_cols_ = [c for c in config.KPROTO_NUMERIC_COLS if c in m.columns]
        self.categorical_cols_ = [c for c in config.KPROTO_CATEGORICAL_COLS if c in m.columns]
        self.columns_ = list(m.columns)
        self.categorical_indices_ = [self.columns_.index(c) for c in self.categorical_cols_]
        self.scaler_ = RobustScaler().fit(np.log1p(np.maximum(m[self.numeric_cols_], 0)))
        return self._finish(m)

    def transform(self, feature_table: pd.DataFrame) -> pd.DataFrame:
        return self._finish(self._base(feature_table)[self.columns_])

    def _finish(self, m: pd.DataFrame) -> pd.DataFrame:
        m = m.copy()
        logged = np.log1p(np.maximum(m[self.numeric_cols_], 0))
        m[self.numeric_cols_] = np.clip(self.scaler_.transform(logged), -config.KPROTO_CLIP, config.KPROTO_CLIP)
        for col in self.categorical_cols_:
            m[col] = m[col].astype("category")
        return m


def fit_kprototypes(matrix: pd.DataFrame, categorical_indices, n_clusters: int = 5,
                    n_init: int = 10, verbose: int = 0):
    from kmodes.kprototypes import KPrototypes

    model = KPrototypes(n_clusters=n_clusters, init="Huang", n_init=n_init,
                        verbose=verbose, random_state=config.RANDOM_STATE)
    labels = model.fit_predict(matrix.values, categorical=categorical_indices)
    return model, labels


def scale_kmeans_features(feature_table: pd.DataFrame):
    scaler = StandardScaler()
    return scaler.fit_transform(feature_table[config.KMEANS_FEATURES]), scaler


def kmeans_inertia(X: np.ndarray, k_values) -> list[float]:
    return [
        KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10).fit(X).inertia_
        for k in k_values
    ]


def fit_kmeans(X: np.ndarray, n_clusters: int = 14) -> KMeans:
    return KMeans(n_clusters=n_clusters, random_state=config.RANDOM_STATE, n_init=10).fit(X)


def prepare_hdbscan_matrix(feature_table: pd.DataFrame):
    """Standardised continuous features alongside raw binary flags."""
    binary = [c for c in config.HDBSCAN_BINARY_COLS if c in feature_table.columns]
    continuous = [c for c in config.HDBSCAN_CONTINUOUS_COLS if c in feature_table.columns]
    df = feature_table[binary + continuous].copy()
    df[binary] = df[binary].fillna(0)
    df[continuous] = df[continuous].fillna(df[continuous].median())
    X = np.hstack([StandardScaler().fit_transform(df[continuous]), df[binary].to_numpy()])
    return df, X


def hdbscan_grid_search(X: np.ndarray, param_grid: dict = config.HDBSCAN_PARAM_GRID,
                        max_noise_ratio: float = config.HDBSCAN_MAX_NOISE_RATIO, progress: bool = True):
    """Score every parameter combination by DBCV (``relative_validity_``).

    Combinations giving fewer than two clusters or more than ``max_noise_ratio``
    noise are skipped. Returns ``(results_df, best_params, best_labels)``.
    """
    from hdbscan import HDBSCAN
    from tqdm import tqdm

    keys = list(param_grid)
    combos = list(product(*param_grid.values()))
    rows, best_score, best_params, best_labels = [], -np.inf, None, None
    for values in tqdm(combos, desc="HDBSCAN grid", disable=not progress):
        params = dict(zip(keys, values))
        if params["min_samples"] > params["min_cluster_size"]:
            continue
        try:
            clusterer = HDBSCAN(**params, core_dist_n_jobs=-1, gen_min_span_tree=True)
            labels = clusterer.fit_predict(X)
        except Exception as exc:  # some metric/parameter combinations are unsupported
            print(f"Skipping {params}: {exc}")
            continue

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = float(np.mean(labels == -1))
        if n_clusters < 2 or noise_ratio > max_noise_ratio:
            continue
        score = clusterer.relative_validity_
        rows.append({**params, "dbcv_score": score, "n_clusters": n_clusters,
                     "n_noise": int(np.sum(labels == -1)), "noise_ratio": noise_ratio,
                     "largest_cluster_size": int(np.bincount(labels[labels >= 0]).max())})
        if score > best_score:
            best_score, best_params, best_labels = score, params, labels

    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values("dbcv_score", ascending=False)
    return results, best_params, best_labels


def gower_hierarchical(feature_table: pd.DataFrame, n_sample: int = 2000, n_clusters: int = 5,
                       method: str = "average"):
    """Average-linkage clustering on Gower distances for a random sample.

    Gower handles mixed types natively but needs an O(n^2) distance matrix,
    hence the sample. Returns ``(sample_with_cluster, linkage_matrix)``.
    """
    import gower
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    df = feature_table.drop(columns=config.GOWER_DROP_COLS, errors="ignore")
    sample = df.sample(n=min(n_sample, len(df)), random_state=config.RANDOM_STATE)
    # gower only recognises numpy dtypes; pandas category/string dtypes must become object.
    for col in sample.select_dtypes(exclude="number").columns:
        sample[col] = sample[col].astype("object")

    distances = gower.gower_matrix(sample)
    Z = linkage(squareform(distances, checks=False), method=method)
    sample = sample.assign(cluster=fcluster(Z, t=n_clusters, criterion="maxclust"))
    return sample, Z
