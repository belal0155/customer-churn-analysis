"""Pipelines, tuning and evaluation helpers."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

SEED = 42


def preprocessor(numeric, categorical, scale=True):
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", num, numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical),
    ])


def build_models(numeric, categorical):
    return {
        "Baseline (majority class)": Pipeline([
            ("prep", preprocessor(numeric, categorical)),
            ("clf", DummyClassifier(strategy="prior"))]),
        "Logistic Regression": Pipeline([
            ("prep", preprocessor(numeric, categorical)),
            ("clf", LogisticRegression(C=0.5, max_iter=2000))]),
        "Random Forest": Pipeline([
            ("prep", preprocessor(numeric, categorical, scale=False)),
            ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                           n_jobs=-1, random_state=SEED))]),
        "XGBoost": Pipeline([
            ("prep", preprocessor(numeric, categorical, scale=False)),
            ("clf", XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=3,
                                  subsample=0.8, colsample_bytree=0.8,
                                  eval_metric="logloss", random_state=SEED, n_jobs=1))]),
    }


def scores(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
    }


def retention_profit(y_true, proba, monthly, threshold,
                     offer_cost=30.0, save_rate=0.30, months_saved=12):
    """Expected profit of contacting every customer scored at or above `threshold`.

    Assumptions (all adjustable): a retention offer costs `offer_cost` per
    contacted customer; it saves `save_rate` of the customers who would
    really have left; a saved customer is worth `months_saved` x monthly bill.
    """
    contact = proba >= threshold
    gain = (contact & (y_true == 1)) * save_rate * months_saved * monthly
    return float(gain.sum() - offer_cost * contact.sum())
