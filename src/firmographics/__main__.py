"""Clean, engineer features for and segment the firmographic dataset.

Usage:
    python -m src.firmographics [--methods kprototypes kmeans hdbscan gower] [--no-eda]

HDBSCAN runs an 800-combination grid search (over an hour); it is opt-in.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.firmographics import analysis, cleaning, clustering, config, data, eda, outliers  # noqa: E402
from src.firmographics.features import engineer_features  # noqa: E402
from src.paths import ARTIFACTS_DIR, ensure_dir  # noqa: E402

METHODS = ("kprototypes", "kmeans", "hdbscan", "gower")


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_raw_eda(raw: pd.DataFrame, out: Path) -> None:
    """Plots behind the column-dropping decisions."""
    for col in ["Country", "Region", "Ownership Type", "Company Status (Active/Inactive)",
                "Fiscal Year End", "Registration Number Type", "Manufacturing Status"]:
        fig = eda.plot_value_counts(raw[col], f"Distribution of Companies by {col}")
        _save(fig, out / f"counts_{col.split(' (')[0].replace(' ', '_').lower()}.png")
    _save(eda.plot_column_agreement(raw["Employees Total"], raw["Employees Single Site"]),
          out / "agreement_employees_total_vs_single_site.png")
    _save(eda.plot_column_agreement(raw["Global Ultimate Company"], raw["Domestic Ultimate Company"]),
          out / "agreement_global_vs_domestic_ultimate.png")
    print(pd.crosstab(raw["Ownership Type"], raw["Ticker"].isna(), dropna=False,
                      colnames=["Ticker is Missing"]))


def run_kprototypes(feature_table, anomalies, args, out):
    prep = clustering.KPrototypesPreprocessor()
    matrix = prep.fit_transform(feature_table)
    model, labels = clustering.fit_kprototypes(matrix, prep.categorical_indices_, n_clusters=args.k_prototypes)
    feature_table["KPrototypes_Cluster"] = labels

    num = analysis.numeric_profile(feature_table, "KPrototypes_Cluster", config.KPROTO_NUMERIC_COLS)
    cat = analysis.categorical_profile(feature_table, "KPrototypes_Cluster", config.KPROTO_CATEGORICAL_COLS)
    num.to_csv(out / "kprototypes_numeric_profile.csv")
    cat.to_csv(out / "kprototypes_categorical_profile.csv")
    print("\nK-Prototypes numeric profile\n", num.round(2).to_string())
    print("\nK-Prototypes categorical profile\n", cat.to_string())

    # Which clusters do the outliers resemble?
    anomaly_matrix = prep.transform(engineer_features(anomalies))
    anomalies["KPrototypes_Cluster"] = model.predict(anomaly_matrix.values, categorical=prep.categorical_indices_)
    lift = analysis.anomaly_lift(anomalies["KPrototypes_Cluster"], feature_table["KPrototypes_Cluster"])
    lift.to_csv(out / "kprototypes_anomaly_lift.csv")
    print("\nAnomaly lift by cluster\n", lift.round(3).to_string())

    with open(out / "kprototypes_model.pkl", "wb") as f:
        pickle.dump({"model": model, "preprocessor": prep}, f)


def run_kmeans(feature_table, args, out):
    X, _ = clustering.scale_kmeans_features(feature_table)
    k_values = range(2, 32, 4)
    _save(eda.plot_elbow(k_values, clustering.kmeans_inertia(X, k_values)), out / "kmeans_elbow.png")
    feature_table["KMeans_Cluster"] = clustering.fit_kmeans(X, n_clusters=args.k_means).labels_
    profile = analysis.numeric_profile(feature_table, "KMeans_Cluster", config.KMEANS_FEATURES)
    profile.to_csv(out / "kmeans_profile.csv")
    print("\nK-Means profile\n", profile.round(2).to_string())


def run_hdbscan(feature_table, out):
    df, X = clustering.prepare_hdbscan_matrix(feature_table)
    results, best_params, best_labels = clustering.hdbscan_grid_search(X)
    results.to_csv(out / "hdbscan_grid_results.csv", index=False)
    if best_params is None:
        print("HDBSCAN: no parameter combination met the cluster/noise constraints")
        return
    _save(eda.plot_hdbscan_grid(results), out / "hdbscan_grid_results.png")
    print(f"\nHDBSCAN best params {best_params}, DBCV={results['dbcv_score'].iloc[0]:.4f}")
    feature_table["HDBSCAN_Cluster"] = best_labels
    df["HDBSCAN_Cluster"] = best_labels
    profile = analysis.numeric_profile(df, "HDBSCAN_Cluster", config.HDBSCAN_CONTINUOUS_COLS)
    profile.to_csv(out / "hdbscan_profile.csv")
    print(profile.round(2).to_string())


def run_gower(feature_table, out):
    sample, Z = clustering.gower_hierarchical(feature_table)
    _save(eda.plot_dendrogram(Z), out / "gower_dendrogram.png")
    print("\nGower hierarchical cluster sizes (sample)\n", sample["cluster"].value_counts().sort_index())


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="firmographics", description=__doc__.splitlines()[0])
    parser.add_argument("--data-path", type=Path, default=config.DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR / "firmographics")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["kprototypes", "kmeans"])
    parser.add_argument("--k-prototypes", type=int, default=5)
    parser.add_argument("--k-means", type=int, default=14)
    parser.add_argument("--no-eda", action="store_true", help="skip exploratory plots")
    args = parser.parse_args(argv)
    out = ensure_dir(args.output_dir)
    figures = ensure_dir(out / "figures")

    raw = data.load_raw(args.data_path)
    print(f"Loaded {raw.shape[0]} rows x {raw.shape[1]} columns")
    if not args.no_eda:
        run_raw_eda(raw, figures)

    df = data.deduplicate(data.drop_irrelevant(raw))
    print(f"After de-duplication: {len(df)} rows")
    print(eda.missing_summary(df).to_string())
    if not args.no_eda:
        _save(eda.plot_phi_matrix(eda.missingness_phi_matrix(df)), figures / "missingness_phi.png")

    df = cleaning.handle_missing(df)
    majority, anomalies, cutoff = outliers.split_outliers(df)
    print(f"Outlier tag cutoff {cutoff:.2f}: {len(majority)} majority, {len(anomalies)} anomalies")

    feature_table = engineer_features(majority)
    feature_table.to_csv(out / "feature_table.csv", index=False)
    if not args.no_eda:
        _save(eda.plot_numeric_histograms(feature_table, config.KPROTO_NUMERIC_COLS + config.HARDWARE_COLS),
              figures / "numeric_histograms.png")
        for col in majority.select_dtypes(include=["object", "str"]).columns:
            _save(eda.plot_top_categories(majority[col]), figures / f"top_{col.replace(' ', '_').lower()}.png")

    if "kprototypes" in args.methods:
        run_kprototypes(feature_table, anomalies, args, out)
    if "kmeans" in args.methods:
        run_kmeans(feature_table, args, out)
    if "hdbscan" in args.methods:
        run_hdbscan(feature_table, out)
    if "gower" in args.methods:
        run_gower(feature_table, out)

    feature_table.to_csv(out / "clustered_firms.csv", index=False)
    anomalies.to_csv(out / "anomalies.csv", index=False)
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
