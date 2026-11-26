"""Loading and cleaning the Telco customer churn data."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "telco_churn.csv.gz"

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET = "Churn"


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy.

    - TotalCharges is text in the raw file. 11 rows are blank, all with
      tenure == 0 (brand-new customers who were never billed), so 0 is the
      correct value rather than an imputed average.
    - Target becomes 0/1.
    - SeniorCitizen becomes a Yes/No category like the other flags.
    - "No internet service" / "No phone service" are folded into "No" for the
      add-on columns, because InternetService / PhoneService already carry it.
    """
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"].str.strip(), errors="coerce")
    blank = out["TotalCharges"].isna()
    assert (out.loc[blank, "tenure"] == 0).all(), "unexpected blank TotalCharges"
    out.loc[blank, "TotalCharges"] = 0.0
    out[TARGET] = (out[TARGET] == "Yes").astype(int)
    out["SeniorCitizen"] = out["SeniorCitizen"].map({0: "No", 1: "Yes"})
    addons = ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]
    for col in addons:
        out[col] = out[col].replace({"No internet service": "No", "No phone service": "No"})
    out = out.drop(columns=["customerID"])
    assert out.duplicated().sum() >= 0
    return out


def feature_lists(df: pd.DataFrame):
    cats = [c for c in df.columns if c not in NUMERIC + [TARGET]]
    return NUMERIC, cats
