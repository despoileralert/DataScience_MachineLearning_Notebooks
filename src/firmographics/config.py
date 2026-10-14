"""Column lists, category mappings and thresholds used across the pipeline."""

from src.paths import DATA_DIR

DEFAULT_DATA_PATH = DATA_DIR / "SDS datathon data" / "champions_group_data.xlsx"
RANDOM_STATE = 42
# Year used to turn "Year Found" into a company age.
REFERENCE_YEAR = 2026

ID_COL = "DUNS Number "  # trailing space is in the source file

# Dropped because they are redundant, near-constant, too granular or free text.
COLS_TO_DROP = [
    "8-Digit SIC Code", "8-Digit SIC Description", "Phone Number", "NAICS Description",
    "NACE Rev 2 Description", "Lattitude", "Longitude", "ANZSIC Description",
    "Address Line 1", "State", "State Or Province Abbreviation", "Postal Code", "Country",
    "Region", "Ownership Type", "Employees Single Site", "SIC Description",
    "Parent Street Address", "Parent State/Province", "Parent State/Province Abbreviation",
    "Parent Postal Code", "Parent Country/Region", "Global Ultimate Street Address",
    "Global Ultimate State/Province", "ISIC Rev 4 Description",
    "Ultimate State/Province Abbreviation", "Global Ultimate Postal Code",
    "Domestic Ultimate Street Address", "Company Description",
    "Domestic Ultimate State/Province Name", "Domestic Ultimate State Abbreviation",
    "Domestic Ultimate Postal Code", "Registration Number",
    "Company Status (Active/Inactive)", "Fiscal Year End",
]

# Rows sharing these are treated as the same company; the highest-revenue row is kept.
DUPLICATE_SUBSET = ["Company Sites", "City", "Year Found"]

INDUSTRY_CODE_COLS = ["NAICS Code", "NACE Rev 2 Code", "ISIC Rev 4 Code", "ANZSIC Code", "SIC Code"]
HIERARCHY_FLAG_COLS = {
    "Is Headquarters": "Is_Headquarters_Flag",
    "Is Domestic Ultimate": "Is_Domestic_Ultimate_Flag",
}
GEOGRAPHY_COLS = ["Parents City", "Global Ultimate City Name", "Domestic Ultimate City Name"]

# Binned device counts -> ordinal ranks (bins differ per column in the source data).
PC_MAP = {"1 to 10": 1, "11 to 50": 2, "51 to 100": 3, "101 to 500": 4, "501 to 1000": 5}
LAPTOP_DESKTOP_MAP = {"1 to 10": 1, "51 to 200": 3.5, "201 to 500": 4.5}
ROUTER_MAP = {"0": 0, "1 to 10": 1, "11 to 50": 2, "51 to 100": 3, "101 to 500": 4}
SERVER_MAP = {"0": 0, "1 to 10": 1, "51 to 200": 3.5, "501 to 1000": 5}
STORAGE_MAP = {"0": 0, "1 to 10": 1, "51 to 200": 3.5, "201 to 500": 4.5, "501 to 1000": 5}
END_USER_DEVICE_COLS = ["No. of PC", "No. of Desktops", "No. of Laptops"]  # MICE-imputed
INFRASTRUCTURE_COLS = ["No. of Storage Devices", "No. of Servers", "No. of Routers"]  # 0-imputed
HARDWARE_COLS = END_USER_DEVICE_COLS + ["No. of Routers", "No. of Servers", "No. of Storage Devices"]

# Outlier tagging
OUTLIER_QUANT_COLS = [
    "Employees Total", "Revenue (USD)", "Corporate Family Members", "Market Value (USD)",
    "IT Budget", "IT spend", "Year Found", "Ticker",
]
OUTLIER_NOMINAL_COLS = [
    "City", "Master Industry", "Entity Type", "Parents City",
    "Global Ultimate City Name", "Domestic Ultimate City Name",
]
IQR_MULTIPLIER = 3.0  # outer fences: firm size data is heavy-tailed
HARDWARE_QUANTILE = 0.95
RARE_LEVEL_FRACTION = 0.005
OUTLIER_SD_CUTOFF = 2.5

# K-Prototypes inputs
KPROTO_DROP_COLS = [
    "Company Sites", "Parent Company", "Global Ultimate Company", "Domestic Ultimate Company",
    "Outlier_Count", "Year Found", "Is_Domestic_Ultimate_Flag", "Branch", "Parent", "Subsidiary",
]
KPROTO_NUMERIC_COLS = [
    "Employees Total", "Revenue (USD)", "Market Value (USD)", "IT Budget", "IT spend",
    "Corporate Family Members", "Revenue per IT dollar", "Years in operation",
    "Revenue per employee", "Proportion of IT budget spent", "IT architecture completeness",
    "IT spend per employee", "IT budget per employee", "Laptop to Desktop ratio",
    "Servers per 100 employees", "IT spend as pct of revenue", "Family members per employee",
]
KPROTO_CATEGORICAL_COLS = [
    "City", "Entity Type", "Ticker", "Parents City", "Global Ultimate City Name",
    "Global Ultimate Country Name", "Domestic Ultimate City Name", "Legal Status",
    "Master Industry", "Is_Headquarters_Flag", "Is_Indonesia_Legalization_Registered",
    "Is_Confirmed_Not_Franchise", "Is_Manufacturing_Firm", "Digital_Presence_Flag",
    "City is parent city", "Is Domestic Ultimate",
    "No. of PC", "No. of Desktops", "No. of Laptops",
    "No. of Routers", "No. of Servers", "No. of Storage Devices",
]
KPROTO_CLIP = 15

# K-Means: cluster on scale and IT spend only
KMEANS_FEATURES = [
    "Employees Total", "Revenue (USD)", "Corporate Family Members",
    "Market Value (USD)", "IT spend", "IT Budget",
]

# HDBSCAN inputs and search space
HDBSCAN_BINARY_COLS = [
    "Is_Indonesia_Legalization_Registered", "Is_Confirmed_Not_Franchise",
    "Is_Manufacturing_Firm", "Digital_Presence_Flag", "Branch", "Parent", "Subsidiary",
    "Is Domestic Ultimate", "City is parent city", "City is domestic ultimate",
]
HDBSCAN_CONTINUOUS_COLS = [
    "Employees Total", "Revenue (USD)", "Market Value (USD)", "IT Budget", "IT spend",
    "No. of Routers", "Corporate Family Members", "Years in operation",
    "Proportion of IT budget spent", "IT architecture completeness", "Revenue per IT dollar",
    "Revenue per employee", "Family members per employee", "IT spend per employee",
    "Laptop to Desktop ratio", "Servers per 100 employees", "IT spend as pct of revenue",
]
HDBSCAN_PARAM_GRID = {
    "min_cluster_size": [200, 300, 500, 700, 1000],
    "min_samples": [10, 20, 30, 50, 100],
    "cluster_selection_epsilon": [0.0, 0.1, 0.2, 0.5],
    "metric": ["euclidean", "manhattan", "chebyshev", "hamming"],
    "cluster_selection_method": ["eom", "leaf"],
}
HDBSCAN_MAX_NOISE_RATIO = 0.5

# Gower + hierarchical clustering
GOWER_DROP_COLS = [
    "Company Sites", "Parent Company", "Global Ultimate Company", "Domestic Ultimate Company",
    "Parents City", "Global Ultimate City Name", "Domestic Ultimate City Name",
    "Branch", "Parent", "Subsidiary", "Is Domestic Ultimate", "Outlier_Count", "Ticker",
]
