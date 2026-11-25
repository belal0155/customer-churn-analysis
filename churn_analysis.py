"""Customer churn project in one file: generate data, clean (Pandas/NumPy), SQL, EDA, ML (scikit-learn).
Run: python churn_analysis.py"""
import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---- 0. Generate synthetic data ----
rng = np.random.default_rng(42)
n = 3000

df = pd.DataFrame({
    "customer_id": np.arange(1, n + 1),
    "age": rng.integers(18, 70, n),
    "tenure_months": rng.integers(1, 60, n),
    "monthly_spend": rng.gamma(4, 15, n).round(2),
    "support_tickets": rng.poisson(1.5, n),
    "plan": rng.choice(["basic", "standard", "premium"], n, p=[0.5, 0.35, 0.15]),
    "city": rng.choice(["Cairo", "Alexandria", "Giza", "Mansoura", "Tanta"], n),
})

# churn probability depends on a few features
logit = (-1.0 - 0.04 * df.tenure_months + 0.35 * df.support_tickets
         - 0.01 * df.monthly_spend + (df.plan == "basic") * 0.6)
p = 1 / (1 + np.exp(-logit))
df["churned"] = (rng.random(n) < p).astype(int)

# add some missing values to practice cleaning
df.loc[rng.choice(n, 90, replace=False), "monthly_spend"] = np.nan
df.loc[rng.choice(n, 60, replace=False), "age"] = np.nan

os.makedirs("data", exist_ok=True)
os.makedirs("reports", exist_ok=True)
df.to_csv("data/customers.csv", index=False)
print("saved", df.shape)

df = pd.read_csv("data/customers.csv")

# ---- 1. Cleaning ----
print("Missing values:\n", df.isna().sum()[lambda s: s > 0])
df["age"] = df["age"].fillna(df["age"].median())
df["monthly_spend"] = df["monthly_spend"].fillna(df["monthly_spend"].median())
df["spend_per_month_tenure"] = df["monthly_spend"] * np.log1p(df["tenure_months"])

# ---- 2. SQL (SQLite) ----
con = sqlite3.connect(":memory:")
df.to_sql("customers", con, index=False)
q = """
SELECT plan, COUNT(*) AS customers,
       ROUND(AVG(churned) * 100, 1) AS churn_rate_pct,
       ROUND(AVG(monthly_spend), 2) AS avg_spend
FROM customers GROUP BY plan ORDER BY churn_rate_pct DESC
"""
sql_result = pd.read_sql(q, con)
print("\nChurn by plan (SQL):\n", sql_result)
sql_result.to_csv("reports/churn_by_plan.csv", index=False)

# ---- 3. EDA / visualization ----
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
sns.barplot(data=df, x="plan", y="churned", ax=ax[0]); ax[0].set_title("Churn rate by plan")
sns.boxplot(data=df, x="churned", y="tenure_months", ax=ax[1]); ax[1].set_title("Tenure vs churn")
sns.barplot(data=df, x="support_tickets", y="churned", ax=ax[2]); ax[2].set_title("Support tickets vs churn")
plt.tight_layout(); plt.savefig("reports/eda.png", dpi=120); plt.close()

plt.figure(figsize=(6, 5))
sns.heatmap(df.select_dtypes("number").drop(columns="customer_id").corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation matrix"); plt.tight_layout(); plt.savefig("reports/correlation.png", dpi=120); plt.close()

# ---- 4. Machine learning ----
X = df.drop(columns=["customer_id", "churned"])
y = df["churned"]
num = ["age", "tenure_months", "monthly_spend", "support_tickets", "spend_per_month_tenure"]
cat = ["plan", "city"]
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
}
lines = []
for name, m in models.items():
    pipe = Pipeline([("pre", pre), ("model", m)])
    cv = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="roc_auc").mean()
    pipe.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
    print(f"\n{name}: CV AUC={cv:.3f}  Test AUC={auc:.3f}")
    print(classification_report(y_te, pipe.predict(X_te)))
    lines.append(f"| {name} | {cv:.3f} | {auc:.3f} |")

with open("reports/model_results.md", "w") as f:
    f.write("| Model | CV ROC-AUC | Test ROC-AUC |\n|---|---|---|\n" + "\n".join(lines) + "\n")
