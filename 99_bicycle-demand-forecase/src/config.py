from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

SEASONAL_PERIOD = 24   # hourly data → daily cycle
TEST_DAYS = 30
ALPHA = 2.0            # asymmetric loss under-prediction penalty
TARGET_RENT_ID = "02128"

TARGET_RENT_IDS = [
    "02111", "02112", "02116", "02122", "02128", "02129", "02130",
    "02139", "02143", "02148", "02159", "02171", "02172", "02178",
    "02179", "02185", "02186", "02191", "02198", "02199",
    "03310", "03311", "03802",
]

# Stations with long-term missing gaps (≥15 days) identified by find_long_missing.py
LONG_MISSING = {
    "02185": [("2022-09-18", "2022-10-23")],  # 36 days
    "02191": [("2022-10-05", "2022-10-21")],  # 17 days
    "02199": [("2022-01-22", "2022-02-07")],  # 17 days
}
