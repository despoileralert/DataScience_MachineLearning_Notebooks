"""Missing-value handling. Each block treats one kind of missingness as found by
the nullity-correlation analysis (see ``eda.missingness_phi_matrix``)."""

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (registers IterativeImputer)
from sklearn.impute import IterativeImputer

from src.firmographics import config
from src.firmographics.industry import add_master_industry


def _flag_present(series: pd.Series) -> pd.Series:
    return series.notna().astype(int)


def encode_presence_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Missing-not-at-random columns where *presence* is the signal."""
    df = df.copy()
    # Corporate-hierarchy markers: blank means standalone / not linked.
    for col, flag in config.HIERARCHY_FLAG_COLS.items():
        df[flag] = _flag_present(df.pop(col))

    # Only listed companies have tickers; 0 stands for "private".
    df["Ticker"] = pd.to_numeric(df["Ticker"], errors="coerce").fillna(0)

    # The few non-null values are all identical, so presence is the information.
    df["Is_Indonesia_Legalization_Registered"] = _flag_present(df.pop("Registration Number Type"))
    df["Is_Manufacturing_Firm"] = _flag_present(df.pop("Manufacturing Status"))

    # read_excel parses FALSE as a boolean; a few rows use "NO"/"No" instead.
    franchise = df.pop("Franchise Status").astype(str).str.upper()
    df["Is_Confirmed_Not_Franchise"] = franchise.isin(["FALSE", "NO"]).astype(int)

    df["Digital_Presence_Flag"] = _flag_present(df.pop("Website"))
    return df


def impute_hardware(df: pd.DataFrame) -> pd.DataFrame:
    """Convert binned device counts to ordinal ranks and fill the gaps.

    Routers / servers / storage: missingness tracks small standalone entities,
    so missing becomes rank 0. PCs / desktops / laptops are strongly
    inter-correlated, so they are filled with MICE.
    """
    df = df.copy()
    for col, mapping in [
        ("No. of Routers", config.ROUTER_MAP),
        ("No. of Servers", config.SERVER_MAP),
        ("No. of Storage Devices", config.STORAGE_MAP),
    ]:
        df[col] = df[col].fillna("0").astype(str).map(mapping)

    df["No. of PC"] = df["No. of PC"].map(config.PC_MAP)
    df["No. of Desktops"] = df["No. of Desktops"].map(config.LAPTOP_DESKTOP_MAP)
    df["No. of Laptops"] = df["No. of Laptops"].map(config.LAPTOP_DESKTOP_MAP)

    imputer = IterativeImputer(random_state=config.RANDOM_STATE, max_iter=10, initial_strategy="median")
    imputed = imputer.fit_transform(df[config.END_USER_DEVICE_COLS])
    df[config.END_USER_DEVICE_COLS] = np.clip(np.round(imputed, 1), 1, 5)
    return df


def fill_remaining(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Blank parent / ultimate cities mean "not linked", not unknown.
    df[config.GEOGRAPHY_COLS] = df[config.GEOGRAPHY_COLS].fillna("Standalone")
    df["Year Found"] = df["Year Found"].fillna(df["Year Found"].median())
    # Nominal code: use a sentinel rather than an imputed number.
    df["Legal Status"] = df["Legal Status"].fillna(-1)
    df["City"] = df["City"].fillna("Unknown")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Full missing-value pipeline; the result has no NaNs and no ID column."""
    df = add_master_industry(df, config.INDUSTRY_CODE_COLS)
    df = encode_presence_flags(df)
    df = impute_hardware(df)
    df = fill_remaining(df)
    return df.drop(columns=config.ID_COL)
