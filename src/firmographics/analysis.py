"""Cluster profiling and the anomaly-lift risk measure."""

import pandas as pd


def numeric_profile(df: pd.DataFrame, label_col: str, numeric_cols) -> pd.DataFrame:
    """Per-cluster means plus the cluster size."""
    cols = [c for c in numeric_cols if c in df.columns]
    profile = df.groupby(label_col)[cols].mean()
    profile["Firm_Count"] = df[label_col].value_counts()
    return profile


def categorical_profile(df: pd.DataFrame, label_col: str, categorical_cols) -> pd.DataFrame:
    """Per-cluster mode (dominant value) of each categorical column."""
    cols = [c for c in categorical_cols if c in df.columns]
    return df.groupby(label_col, observed=True)[cols].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else None
    )


def anomaly_lift(anomaly_clusters: pd.Series, majority_clusters: pd.Series) -> pd.DataFrame:
    """How strongly each cluster attracts anomalies, relative to the base rate.

    ``pressure`` = anomalies assigned to the cluster / cluster size;
    ``lift`` = pressure / overall anomaly rate. A lift of 2 means firms near
    that cluster are twice as likely as average to be anomalous.
    """
    sizes = majority_clusters.value_counts()
    counts = anomaly_clusters.value_counts().reindex(sizes.index, fill_value=0)
    global_rate = len(anomaly_clusters) / (len(anomaly_clusters) + len(majority_clusters))
    report = pd.DataFrame({"cluster_size": sizes, "n_anomalies": counts})
    report["pressure"] = report["n_anomalies"] / report["cluster_size"]
    report["lift"] = report["pressure"] / global_rate
    return report.sort_index()
