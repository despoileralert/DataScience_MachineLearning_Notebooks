"""Feature engineering on the cleaned firmographic table."""

import pandas as pd

from src.firmographics import config


def _per(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Ratio with +1 smoothing so zero denominators stay finite."""
    return numerator / (denominator + 1)


def engineer_features(df: pd.DataFrame, reference_year: int = config.REFERENCE_YEAR) -> pd.DataFrame:
    ft = df.copy()
    ft["Entity Type"] = ft["Entity Type"].astype("category")
    ft = pd.concat([ft, pd.get_dummies(ft["Entity Type"], dtype=int)], axis=1)

    ft["Is Domestic Ultimate"] = (ft["Company Sites"] == ft["Domestic Ultimate Company"]).astype(int)
    ft["Years in operation"] = reference_year - ft["Year Found"]
    ft["City is parent city"] = (ft["City"] == ft["Parents City"]).astype(int)
    ft["City is domestic ultimate"] = (ft["City"] == ft["Domestic Ultimate City Name"]).astype(int)
    ft["IT architecture completeness"] = (ft[config.HARDWARE_COLS] > 0).sum(axis=1)

    employees = ft["Employees Total"]
    ft["Proportion of IT budget spent"] = _per(ft["IT spend"], ft["IT Budget"])
    ft["IT budget per employee"] = _per(ft["IT Budget"], employees)
    ft["Revenue per IT dollar"] = _per(ft["Revenue (USD)"], ft["IT Budget"])
    ft["Revenue per employee"] = _per(ft["Revenue (USD)"], employees)
    ft["Family members per employee"] = _per(ft["Corporate Family Members"], employees)
    ft["IT spend per employee"] = _per(ft["IT spend"], employees)
    ft["Laptop to Desktop ratio"] = _per(ft["No. of Laptops"], ft["No. of Desktops"])
    ft["Servers per 100 employees"] = _per(ft["No. of Servers"], employees / 100)
    ft["IT spend as pct of revenue"] = _per(ft["IT spend"], ft["Revenue (USD)"])
    return ft
