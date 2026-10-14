import pandas as pd
import pytest

from src.air_travel import analysis, data


@pytest.fixture
def raw():
    holidays = pd.DataFrame({
        "ADM_name": ["United States"] * 3 + ["Japan"],
        "ISO3": ["USA", "USA", "USA", "JPN"],
        "Date": ["2015-01-01", "2015-01-19", "2015-01-19", "2015-02-11"],  # one exact duplicate
        "Name": ["New Year", "MLK Day", "MLK Day", "Foundation Day"],
        "Type": ["Public holiday"] * 4,
    })
    passengers = pd.DataFrame({
        "ISO3": ["USA", "USA", "JPN", "JPN", "FRA"],
        "Year": [2015, 2015, 2015, 2015, 2015],
        "Month": [1, 2, 1, 2, 1],
        "Total": [None] * 5, "Domestic": [None] * 5, "International": [None] * 5,
        "Total_OS": [100.0, 90.0, 50.0, 60.0, None],
    })
    return holidays, passengers


def test_map_region():
    assert data.map_region("USA") == "America"
    assert data.map_region("SGP") == "Asia"
    assert data.map_region("ZAF") == "Other"


def test_build_dataset_counts_holidays_and_drops_holiday_free_months(raw):
    df = data.build_dataset(*raw, years=(2015,))
    assert list(df.columns) == ["ISO3", "Year", "Month", "HolidayCount", "Total_OS", "Date", "Region"]
    counts = df.set_index(["ISO3", "Month"])["HolidayCount"].to_dict()
    # Duplicate holiday removed; USA Feb and JPN Jan have no holidays so are dropped;
    # FRA has no passenger total.
    assert counts == {("USA", 1): 2, ("JPN", 2): 1}
    assert set(df["Date"]) == {pd.Timestamp("2015-01-01"), pd.Timestamp("2015-02-01")}
    assert set(df["Region"]) == {"America", "Asia"}


def test_regional_growth_is_relative_to_first_month():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2014-01-01", "2014-02-01", "2014-03-01"] * 2),
        "Region": ["Asia"] * 3 + ["Europe"] * 3,
        "Total_OS": [100, 110, 120, 50, 25, 50],
    })
    growth = analysis.regional_growth(df).set_index(["Region", "Date"])
    assert growth.loc[("Asia", pd.Timestamp("2014-03-01")), "pct_change"] == pytest.approx(20)
    assert growth.loc[("Europe", pd.Timestamp("2014-02-01")), "pct_change"] == pytest.approx(-50)
    assert growth.loc[("Asia", pd.Timestamp("2014-02-01")), "pct_change_smooth"] == pytest.approx(10)


def test_monthly_growth_index_starts_at_100():
    df = pd.DataFrame({
        "Region": ["Asia"] * 4, "Month": [1, 1, 2, 2],
        "HolidayCount": [1, 3, 2, 2], "Total_OS": [10, 30, 30, 50],
    })
    monthly = analysis.monthly_growth_index(df)
    assert monthly["growth_index"].tolist() == pytest.approx([100, 200])
    assert monthly["HolidayCount"].tolist() == pytest.approx([2, 2])
