"""Collapse the five industry-code schemes into one ``Master Industry`` label.

Schemes are tried in priority order (NAICS, NACE Rev 2, ISIC Rev 4, ANZSIC);
the first one with a recognised code wins, and SIC is the fallback.
Each table maps an inclusive ``(low, high)`` code range to a sector name.
"""

import pandas as pd

# Keyed on the first two digits of the 4-digit zero-padded code.
NAICS_SECTORS = [
    (11, 11, "Agriculture, Forestry, Fishing and Hunting"),
    (21, 21, "Mining"),
    (22, 22, "Utilities"),
    (23, 23, "Construction"),
    (31, 33, "Manufacturing"),
    (42, 42, "Wholesale Trade"),
    (44, 45, "Retail Trade"),
    (48, 49, "Transportation and Warehousing"),
    (51, 51, "Information"),
    (52, 52, "Finance and Insurance"),
    (53, 53, "Real Estate Rental and Leasing"),
    (54, 54, "Professional, Scientific, and Technical Services"),
    (55, 55, "Management of Companies and Enterprises"),
    (56, 56, "Administrative and Support and Waste Services"),
    (61, 61, "Educational Services"),
    (62, 62, "Health Care and Social Assistance"),
    (71, 71, "Arts, Entertainment, and Recreation"),
    (72, 72, "Accommodation and Food Services"),
    (81, 81, "Other Services (except Public Administration)"),
    (92, 92, "Public Administration"),
]

NACE_SECTIONS = [
    (1, 3, "Agriculture, forestry, and fishing"),
    (5, 9, "Mining and quarrying"),
    (10, 33, "Manufacturing"),
    (35, 35, "Electricity, gas, steam, and air conditioning supply"),
    (36, 39, "Water supply; sewerage, waste management"),
    (41, 43, "Construction"),
    (45, 47, "Wholesale and retail trade"),
    (49, 53, "Transportation and storage"),
    (55, 56, "Accommodation and food service activities"),
    (58, 63, "Information and communication"),
    (64, 66, "Financial and insurance activities"),
    (68, 68, "Real estate activities"),
    (69, 75, "Professional, scientific, and technical activities"),
    (77, 82, "Administrative and support service activities"),
    (84, 84, "Public administration and defence"),
    (85, 85, "Education"),
    (86, 88, "Human health and social work activities"),
    (90, 93, "Arts, entertainment, and recreation"),
    (94, 96, "Other service activities"),
]

# Same divisions as NACE; label punctuation differs (no Oxford comma).
ISIC_SECTIONS = [
    (lo, hi, label.replace(", and ", " and ")) for lo, hi, label in NACE_SECTIONS
]

# Keyed on the full 4-digit code.
ANZSIC_DIVISIONS = [
    (100, 599, "Agriculture, Forestry and Fishing"),
    (600, 1099, "Mining"),
    (1100, 2599, "Manufacturing"),
    (2600, 2999, "Electricity, Gas, Water and Waste Services"),
    (3000, 3299, "Construction"),
    (3300, 3899, "Wholesale Trade"),
    (3900, 4399, "Retail Trade"),
    (4400, 4599, "Accommodation and Food Services"),
    (4600, 5399, "Transport, Postal and Warehousing"),
    (5400, 6099, "Information Media and Telecommunications"),
    (6200, 6499, "Financial and Insurance Services"),
    (6600, 6799, "Rental, Hiring and Real Estate Services"),
    (6900, 7099, "Professional, Scientific and Technical Services"),
    (7200, 7399, "Administrative and Support Services"),
    (7500, 7799, "Public Administration and Safety"),
    (8000, 8299, "Education and Training"),
    (8400, 8799, "Health Care and Social Assistance"),
    (8900, 9299, "Arts and Recreation Services"),
    (9400, 9699, "Other Services"),
]

SIC_DIVISIONS = [
    (100, 999, "Agriculture, Forestry, Fishing"),
    (1000, 1499, "Mining"),
    (1500, 1799, "Construction"),
    (2000, 3999, "Manufacturing"),
    (4000, 4999, "Transportation, Communications, Utilities"),
    (5000, 5199, "Wholesale Trade"),
    (5200, 5999, "Retail Trade"),
    (6000, 6799, "Finance, Insurance, Real Estate"),
    (7000, 8999, "Services"),
    (9100, 9729, "Public Administration"),
    (9900, 9999, "Nonclassifiable"),
]

UNKNOWN = "Unknown"


def _pad_code(value) -> str | None:
    """Normalise a numeric code (possibly a float like 4120.0) to a 4+ digit string."""
    if pd.isna(value):
        return None
    return str(int(float(value))).zfill(4)


def _lookup(value: int, table) -> str | None:
    for low, high, label in table:
        if low <= value <= high:
            return label
    return None


def master_industry(row) -> str:
    # (column, table, use only the first two digits?)
    schemes = [
        ("NAICS Code", NAICS_SECTORS, True),
        ("NACE Rev 2 Code", NACE_SECTIONS, True),
        ("ISIC Rev 4 Code", ISIC_SECTIONS, True),
        ("ANZSIC Code", ANZSIC_DIVISIONS, False),
    ]
    for column, table, two_digit in schemes:
        code = _pad_code(row.get(column))
        if code:
            label = _lookup(int(code[:2]) if two_digit else int(code), table)
            if label:
                return label

    sic = row.get("SIC Code")
    if pd.notna(sic):
        return _lookup(int(sic), SIC_DIVISIONS) or UNKNOWN
    return UNKNOWN


def add_master_industry(df: pd.DataFrame, code_cols) -> pd.DataFrame:
    """Add ``Master Industry`` and drop the raw code columns."""
    df = df.copy()
    df["Master Industry"] = df.apply(master_industry, axis=1)
    return df.drop(columns=list(code_cols))
