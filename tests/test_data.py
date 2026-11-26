import pandas as pd

from churn.data import clean, load_raw


def test_total_charges_blank_only_for_new_customers():
    raw = load_raw()
    blank = raw["TotalCharges"].str.strip() == ""
    assert blank.sum() == 11
    assert (raw.loc[blank, "tenure"] == 0).all()


def test_clean_output_is_model_ready():
    df = clean(load_raw())
    assert df["Churn"].isin([0, 1]).all()
    assert df["TotalCharges"].notna().all()
    assert "customerID" not in df.columns
    assert set(df["OnlineSecurity"].unique()) == {"Yes", "No"}
    assert 0.25 < df["Churn"].mean() < 0.28
