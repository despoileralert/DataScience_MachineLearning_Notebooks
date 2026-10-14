"""Exploratory plots used to justify the cleaning decisions.
Every function returns a matplotlib Figure."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_value_counts(series: pd.Series, title: str) -> plt.Figure:
    counts = series.value_counts()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=counts.index.astype(str), y=counts.values, hue=counts.index.astype(str),
                palette="viridis", legend=False, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=10)
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(series.name, fontsize=12)
    ax.set_ylabel("Number of Companies", fontsize=12)
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    return fig


def plot_column_agreement(a: pd.Series, b: pd.Series) -> plt.Figure:
    """Two-circle diagram showing how often two columns hold identical values."""
    matches = int((a == b).sum())
    mismatches = len(a) - matches
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_patch(plt.Circle((-0.35, 0), 0.7, color="skyblue", alpha=0.6))
    ax.add_patch(plt.Circle((0.35, 0), 0.7, color="lightcoral", alpha=0.6))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0, 0, f"MATCHES\n(Identical Values)\n\n{matches}", ha="center", va="center",
            fontsize=12, fontweight="bold")
    for x in (-0.6, 0.6):
        ax.text(x, 0, f"MISMATCH\n{mismatches}", ha="center", va="center", fontsize=10)
    ax.text(-0.35, -0.85, a.name, ha="center")
    ax.text(0.35, -0.85, b.name, ha="center")
    ax.set_title(f"Venn Analysis: Data Redundancy Check\n{a.name} vs. {b.name}", fontsize=14, pad=20)
    return fig


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({
        "Missing Count": df.isna().sum(),
        "Percentage (%)": df.isna().mean() * 100,
    }).sort_values("Percentage (%)", ascending=False)
    return summary[summary["Missing Count"] > 0]


def missingness_phi_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Phi coefficient between the missingness indicators of every column with gaps.

    Phi on two binary variables equals their Pearson correlation.
    """
    missing = df.isna()
    return missing.loc[:, missing.any()].astype(int).corr()


def plot_phi_matrix(phi: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(phi, annot=True, fmt=".2f", cmap="vlag", center=0,
                mask=np.triu(np.ones_like(phi, dtype=bool)), square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": r"Phi Association ($\phi$)"}, ax=ax)
    ax.set_title(r"Quantitative Missingness Dependency Matrix ($\phi$)", fontsize=16)
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    return fig


def plot_numeric_histograms(df: pd.DataFrame, columns, ncols: int = 3, max_bins: int = 100) -> plt.Figure:
    """Histograms of values up to the 95th percentile so the long tails don't hide the bulk.

    Bins are computed on the plotted range only: Freedman-Diaconis over the
    full heavy-tailed range can ask for millions of bins.
    """
    columns = [c for c in columns if c in df.columns]
    nrows = -(-len(columns) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, columns):
        values = df[col].dropna()
        values = values[values <= np.percentile(values, 95)]
        n_bins = min(len(np.histogram_bin_edges(values, bins="fd")) - 1, max_bins)
        ax.hist(values, bins=max(n_bins, 1))
        ax.set_title(f"{col} Distribution")
    for ax in axes[len(columns):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_top_categories(series: pd.Series, top: int = 20) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    series.fillna("Missing").value_counts().head(top).plot(kind="bar", grid=False, ax=ax)
    ax.set_title(f"{series.name} (Top {top})")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    return fig


def plot_elbow(k_values, scores, ylabel: str = "Inertia") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_values), scores, marker="o")
    ax.set_xticks(list(k_values))
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel(ylabel)
    ax.set_title("Elbow Method for Optimal k")
    return fig


def plot_dendrogram(Z) -> plt.Figure:
    from scipy.cluster.hierarchy import dendrogram

    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, no_labels=True, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram (Gower)")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Distance")
    return fig


def plot_hdbscan_grid(results: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    sns.boxplot(data=results, x="min_cluster_size", y="dbcv_score", ax=axes[0, 0])
    axes[0, 0].set_title("DBCV Score vs Min Cluster Size")
    sns.boxplot(data=results, x="min_samples", y="dbcv_score", ax=axes[0, 1])
    axes[0, 1].set_title("DBCV Score vs Min Samples")
    sns.histplot(data=results, x="n_clusters", bins=20, ax=axes[1, 0])
    axes[1, 0].set_title("Distribution of Number of Clusters")
    sns.scatterplot(data=results, x="noise_ratio", y="dbcv_score", hue="n_clusters", ax=axes[1, 1])
    axes[1, 1].set_title("Noise Ratio vs DBCV Score")
    fig.tight_layout()
    return fig
