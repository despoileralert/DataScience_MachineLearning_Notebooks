"""Loading, cleaning and merging the holiday and passenger datasets."""

from pathlib import Path

import pandas as pd

from src.paths import DATA_DIR

DEFAULT_DATA_DIR = DATA_DIR / "DSA2101 Project data"

_TIDYTUESDAY = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2024/2024-12-24"
HOLIDAYS_URL = f"{_TIDYTUESDAY}/global_holidays.csv"
PASSENGERS_URL = f"{_TIDYTUESDAY}/monthly_passengers.csv"

KEYS = ["ISO3", "Year", "Month"]
# Official-statistics columns are mostly empty, so the analysis uses the
# open-source total (``Total_OS``) instead.
OFFICIAL_COLS = ["Total", "Domestic", "International"]

REGIONS = {
    "America": ["USA", "CAN", "MEX", "BRA", "ARG", "CHL", "COL", "PER"],
    "Europe": ["GBR", "FRA", "DEU", "ESP", "ITA", "POL", "NLD", "SWE"],
    "Asia": ["CHN", "JPN", "KOR", "SGP", "IND", "THA", "MYS", "ARE"],
}
STUDY_YEARS = (2014, 2015, 2016, 2017)
STUDY_REGIONS = ("America", "Europe", "Asia")


def load_raw(data_dir: Path = DEFAULT_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(global_holidays, monthly_passengers)``.

    Reads the local copies when present and falls back to the TidyTuesday URLs.
    """
    data_dir = Path(data_dir)
    holidays_path = data_dir / "global_holidays.csv"
    passengers_path = data_dir / "monthly_passengers.csv"
    holidays = pd.read_csv(holidays_path if holidays_path.exists() else HOLIDAYS_URL)
    passengers = pd.read_csv(passengers_path if passengers_path.exists() else PASSENGERS_URL)
    return holidays, passengers


def clean_holidays(holidays: pd.DataFrame) -> pd.DataFrame:
    holidays = holidays.drop_duplicates().copy()
    holidays["Date"] = pd.to_datetime(holidays["Date"])
    holidays["Year"] = holidays["Date"].dt.year
    holidays["Month"] = holidays["Date"].dt.month
    return holidays


def clean_passengers(passengers: pd.DataFrame) -> pd.DataFrame:
    return passengers.drop_duplicates().dropna(subset=["Total_OS"])


def monthly_holiday_counts(passengers: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    """Count holidays per country-month and attach the passenger total.

    Country-months with no recorded holiday are dropped (the left merge leaves
    them with NaN holiday fields), so ``HolidayCount`` is always >= 1.
    """
    merged = (
        passengers.merge(holidays, on=KEYS, how="left")
        .drop(columns=OFFICIAL_COLS)
        .dropna()
        .drop_duplicates()
    )
    counts = (
        merged.groupby(KEYS)["Type"].count().reset_index().rename(columns={"Type": "HolidayCount"})
    )
    counts = counts.merge(passengers, on=KEYS, how="inner").drop(columns=OFFICIAL_COLS)
    counts["Date"] = pd.to_datetime(
        counts["Year"].astype(str) + counts["Month"].astype(str), format="%Y%m"
    )
    return counts


def map_region(iso3: str) -> str:
    for region, codes in REGIONS.items():
        if iso3 in codes:
            return region
    return "Other"


def build_dataset(
    holidays: pd.DataFrame,
    passengers: pd.DataFrame,
    years=STUDY_YEARS,
    regions=STUDY_REGIONS,
) -> pd.DataFrame:
    """Full cleaning pipeline: one row per country-month with
    ``ISO3, Year, Month, HolidayCount, Total_OS, Date, Region``."""
    df = monthly_holiday_counts(clean_passengers(passengers), clean_holidays(holidays))
    df["Region"] = df["ISO3"].map(map_region)
    return df[df["Year"].isin(years) & df["Region"].isin(regions)].reset_index(drop=True)
