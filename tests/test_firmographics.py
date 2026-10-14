import numpy as np
import pandas as pd
import pytest

from src.firmographics import analysis, data, eda, features, industry, outliers


@pytest.mark.parametrize("codes, expected", [
    ({"NAICS Code": 236220.0}, "Construction"),
    ({"NAICS Code": 999999.0, "NACE Rev 2 Code": 4120.0}, "Construction"),  # unknown NAICS falls through
    ({"ISIC Rev 4 Code": 3510.0}, "Electricity, gas, steam and air conditioning supply"),
    ({"NACE Rev 2 Code": 3510.0}, "Electricity, gas, steam, and air conditioning supply"),
    ({"ISIC Rev 4 Code": 311.0}, "Agriculture, forestry and fishing"),  # pads to 0311
    ({"ANZSIC Code": 3011.0}, "Construction"),
    ({"SIC Code": 7372}, "Services"),
    ({"SIC Code": 9800}, "Unknown"),
    ({}, "Unknown"),
])
def test_master_industry(codes, expected):
    row = pd.Series({col: codes.get(col, np.nan) for col in
                     ["NAICS Code", "NACE Rev 2 Code", "ISIC Rev 4 Code", "ANZSIC Code", "SIC Code"]})
    assert industry.master_industry(row) == expected


def test_deduplicate_keeps_highest_revenue():
    df = pd.DataFrame({
        "Company Sites": ["A", "A", "B"], "City": ["X", "X", "X"],
        "Year Found": [2000, 2000, 2000], "Revenue (USD)": [5, 10, 1],
    })
    result = data.deduplicate(df)
    assert len(result) == 2
    assert result.loc[result["Company Sites"] == "A", "Revenue (USD)"].item() == 10


def test_engineer_features_ratios():
    df = pd.DataFrame({
        "Entity Type": ["Parent", "Branch"], "Company Sites": ["A", "B"],
        "Domestic Ultimate Company": ["A", "Z"], "Year Found": [2016, 2000],
        "City": ["X", "Y"], "Parents City": ["X", "Q"], "Domestic Ultimate City Name": ["X", "Y"],
        "Employees Total": [9, 0], "Revenue (USD)": [100, 0], "IT Budget": [9, 0], "IT spend": [3, 0],
        "Corporate Family Members": [1, 0],
        "No. of PC": [1, 0], "No. of Desktops": [1, 0], "No. of Laptops": [1, 0],
        "No. of Routers": [0, 0], "No. of Servers": [1, 0], "No. of Storage Devices": [0, 0],
    })
    ft = features.engineer_features(df, reference_year=2026)
    assert ft["Years in operation"].tolist() == [10, 26]
    assert ft["Revenue per employee"].tolist() == [10, 0]
    assert ft["IT architecture completeness"].tolist() == [4, 0]
    assert ft["Is Domestic Ultimate"].tolist() == [1, 0]
    assert ft["City is parent city"].tolist() == [1, 0]
    assert {"Parent", "Branch"} <= set(ft.columns)


def test_count_outlier_tags_flags_extremes_and_rare_levels():
    n = 400
    df = pd.DataFrame({col: np.ones(n) for col in
                       ["Employees Total", "Revenue (USD)", "Corporate Family Members",
                        "Market Value (USD)", "IT Budget", "IT spend", "Year Found", "Ticker",
                        "No. of PC", "No. of Desktops", "No. of Laptops",
                        "No. of Routers", "No. of Servers", "No. of Storage Devices"]})
    df["Revenue (USD)"] = np.arange(n, dtype=float)
    df.loc[0, "Revenue (USD)"] = 1e9
    for col in ["City", "Master Industry", "Entity Type", "Parents City",
                "Global Ultimate City Name", "Domestic Ultimate City Name"]:
        df[col] = "common"
    df.loc[1, "City"] = "rare"
    tags = outliers.count_outlier_tags(df)
    assert tags[0] == 1 and tags[1] == 1
    assert tags.drop([0, 1]).eq(0).all()


def test_anomaly_lift():
    majority = pd.Series([0] * 80 + [1] * 10)
    anomalies = pd.Series([1] * 10)
    report = analysis.anomaly_lift(anomalies, majority)
    assert report.loc[0, "n_anomalies"] == 0
    assert report.loc[1, "pressure"] == pytest.approx(1.0)
    assert report.loc[1, "lift"] == pytest.approx(10.0)  # global rate 10 / 100


def test_missingness_phi_matrix():
    df = pd.DataFrame({"a": [1, None, 1, None], "b": [1, None, 1, None], "c": [None, 1, 1, 1], "d": [1, 2, 3, 4]})
    phi = eda.missingness_phi_matrix(df)
    assert list(phi.columns) == ["a", "b", "c"]
    assert phi.loc["a", "b"] == pytest.approx(1.0)
