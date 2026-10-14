"""Loading, column pruning and de-duplication."""

from pathlib import Path

import pandas as pd

from src.firmographics import config


def load_raw(path: Path = config.DEFAULT_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)


def drop_irrelevant(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=config.COLS_TO_DROP)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates, then near-duplicates on name/city/founding year,
    keeping the highest-revenue record of each group."""
    return (
        df.drop_duplicates()
        .sort_values("Revenue (USD)", ascending=False, kind="stable")
        .drop_duplicates(subset=config.DUPLICATE_SUBSET, keep="first")
        .reset_index(drop=True)
    )
