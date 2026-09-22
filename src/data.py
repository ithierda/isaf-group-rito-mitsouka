"""Data loading helpers shared by notebooks and the Streamlit app."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "speed_dating.csv"

TARGET = "match"
PROTECTED = "race"  # ethnicity of the participant; `race_o` = partner's

RACE_LABELS = {
    1: "Black / African American",
    2: "European / Caucasian-American",
    3: "Latino / Hispanic American",
    4: "Asian / Pacific Islander / Asian-American",
    5: "Native American",
    6: "Other",
}


def load_raw() -> pd.DataFrame:
    """Return the untouched Speed Dating dataset (8,378 rows x 195 columns)."""
    # The file contains a few Mac Roman characters (e.g. "Supérieure").
    return pd.read_csv(RAW_PATH, encoding="mac_roman")
