"""Multivariate outlier separation.

Each record collects one tag per feature on which it is extreme; records whose
tag count is far above average are moved to an anomalies set instead of being
deleted, so they can later be assigned to clusters and analysed.
"""

import pandas as pd

from src.firmographics import config


def count_outlier_tags(
    df: pd.DataFrame,
    iqr_multiplier: float = config.IQR_MULTIPLIER,
    hardware_quantile: float = config.HARDWARE_QUANTILE,
    rare_fraction: float = config.RARE_LEVEL_FRACTION,
) -> pd.Series:
    tags = pd.Series(0, index=df.index, name="Outlier_Count")

    # Quantitative: above the upper outer fence (Q3 + k * IQR).
    for col in config.OUTLIER_QUANT_COLS:
        values = pd.to_numeric(df[col], errors="coerce").fillna(0)
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        # A zero IQR (mostly-zero placeholder columns) would flag every non-zero value.
        if iqr > 0:
            tags += (values > q3 + iqr_multiplier * iqr).astype(int)

    # Ordinal hardware ranks: above the high quantile.
    for col in config.HARDWARE_COLS:
        threshold = df[col].quantile(hardware_quantile)
        if threshold > 0:
            tags += (df[col] > threshold).astype(int)

    # Nominal: belongs to a rare level.
    min_count = rare_fraction * len(df)
    for col in config.OUTLIER_NOMINAL_COLS:
        counts = df[col].value_counts()
        tags += df[col].isin(counts[counts < min_count].index).astype(int)

    return tags


def split_outliers(df: pd.DataFrame, sd_cutoff: float = config.OUTLIER_SD_CUTOFF):
    """Return ``(majority, anomalies, cutoff)``; both frames carry ``Outlier_Count``."""
    df = df.copy()
    df["Outlier_Count"] = count_outlier_tags(df)
    cutoff = df["Outlier_Count"].mean() + sd_cutoff * df["Outlier_Count"].std()
    is_extreme = df["Outlier_Count"] > cutoff
    return df[~is_extreme].copy(), df[is_extreme].copy(), cutoff
