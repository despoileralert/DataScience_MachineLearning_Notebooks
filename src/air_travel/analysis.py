"""Aggregations behind the three report plots."""

import pandas as pd


def regional_growth(df: pd.DataFrame, smoothing_window: int = 3) -> pd.DataFrame:
    """Monthly regional passenger totals as % change from each region's first month.

    Adds ``pct_change_smooth``, a centred rolling mean of ``pct_change``.
    """
    totals = df.groupby(["Date", "Region"])["Total_OS"].sum().reset_index().sort_values("Date")
    base = totals.groupby("Region")["Total_OS"].transform("first")
    totals["base"] = base
    totals["pct_change"] = (totals["Total_OS"] - base) / base * 100
    totals["pct_change_smooth"] = totals.groupby("Region")["pct_change"].transform(
        lambda s: s.rolling(window=smoothing_window, center=True).mean()
    )
    return totals.reset_index(drop=True)


def monthly_growth_index(df: pd.DataFrame) -> pd.DataFrame:
    """Average holidays and passengers per calendar month for each region, with
    passengers indexed to January = 100."""
    monthly = (
        df.groupby(["Region", "Month"])[["HolidayCount", "Total_OS"]]
        .mean()
        .reset_index()
        .sort_values(["Region", "Month"])
    )
    monthly["base"] = monthly.groupby("Region")["Total_OS"].transform("first")
    monthly["growth_index"] = monthly["Total_OS"] / monthly["base"] * 100
    return monthly.reset_index(drop=True)
