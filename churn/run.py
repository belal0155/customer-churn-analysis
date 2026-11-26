"""Run the full pipeline: python -m churn.run

Writes figures to reports/figures, metrics to reports/metrics.json and the
files that power the live demo to reports/demo_model.json.
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split

from churn.data import TARGET, clean, feature_lists, load_raw
from churn.modeling import SEED, build_models, retention_profit, scores

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
PAL = {"Logistic Regression": "#2563eb", "Random Forest": "#16a34a", "XGBoost": "#dc2626",
       "Baseline (majority class)": "#9ca3af"}


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=130)
    plt.close()


def eda(df):
    rate = df[TARGET].mean()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, col in zip(ax[:2], ["Contract", "InternetService"]):
        g = df.groupby(col)[TARGET].mean().sort_values()
        sns.barplot(x=g.values * 100, y=g.index, ax=a, color="#2563eb")
        a.axvline(rate * 100, color="#dc2626", ls="--", label=f"overall {rate:.1%}")
        a.set_xlabel("Churn rate (%)"); a.set_ylabel(""); a.set_title(f"Churn by {col}"); a.legend()
    bins = pd.cut(df["tenure"], [-1, 6, 12, 24, 48, 72], labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
    g = df.groupby(bins, observed=True)[TARGET].mean() * 100
    sns.barplot(x=g.index.astype(str), y=g.values, ax=ax[2], color="#2563eb")
    ax[2].set_xlabel("Tenure (months)"); ax[2].set_ylabel("Churn rate (%)"); ax[2].set_title("Churn by tenure")
    save("01_churn_drivers_eda.png")
    return {
        "churn_rate": rate,
        "by_contract": df.groupby("Contract")[TARGET].mean().to_dict(),
        "by_internet": df.groupby("InternetService")[TARGET].mean().to_dict(),
        "first_year": float(df[df.tenure <= 12][TARGET].mean()),
        "after_year": float(df[df.tenure > 12][TARGET].mean()),
    }


def main():
    df = clean(load_raw())
    numeric, cats = feature_lists(df)
    X, y = df[numeric + cats], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    out = {"n_customers": len(df), "n_train": len(Xtr), "n_test": len(Xte), "eda": eda(df)}

    models = build_models(numeric, cats)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    # Light tuning for the boosted model, on the training set only.
    search = RandomizedSearchCV(
        models["XGBoost"],
        {"clf__max_depth": [2, 3, 4], "clf__n_estimators": [200, 300, 500],
         "clf__learning_rate": [0.02, 0.03, 0.05], "clf__min_child_weight": [1, 5, 10],
         "clf__reg_lambda": [1, 5, 10]},
        n_iter=12, scoring="roc_auc", cv=3, random_state=SEED, n_jobs=1)
    search.fit(Xtr, ytr)
    models["XGBoost"] = search.best_estimator_
    out["xgb_best_params"] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}

    results, fitted, probas = {}, {}, {}
    for name, m in models.items():
        cvr = cross_validate(m, Xtr, ytr, cv=cv, scoring=["roc_auc", "average_precision"])
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        fitted[name], probas[name] = m, p
        results[name] = {"cv_roc_auc": float(cvr["test_roc_auc"].mean()),
                         "cv_roc_auc_std": float(cvr["test_roc_auc"].std()),
                         **{"test_" + k: float(v) for k, v in scores(yte, p).items()}}
        print(name, {k: round(v, 3) for k, v in results[name].items()})
    out["models"] = results

    # ROC / PR curves
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(yte, p); pr, rc, _ = precision_recall_curve(yte, p)
        ax[0].plot(fpr, tpr, color=PAL[name], label=f"{name} ({results[name]['test_roc_auc']:.3f})")
        ax[1].plot(rc, pr, color=PAL[name], label=f"{name} ({results[name]['test_pr_auc']:.3f})")
    ax[0].plot([0, 1], [0, 1], "k:", lw=1)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test set)")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve (test set)")
    ax[0].legend(loc="lower right"); ax[1].legend(loc="upper right")
    save("02_model_comparison_roc_pr.png")

    # Pick the final model by cross-validated ROC-AUC (never by test score).
    best = max((n for n in results if not n.startswith("Baseline")), key=lambda n: results[n]["cv_roc_auc"])
    out["selected_model"] = best
    pbest = probas[best]

    # Calibration
    fig, ax = plt.subplots(figsize=(5.6, 5))
    for name in ["Logistic Regression", "XGBoost"]:
        fr, mp = calibration_curve(yte, probas[name], n_bins=8, strategy="quantile")
        ax.plot(mp, fr, "o-", color=PAL[name], label=name)
    ax.plot([0, 1], [0, 1], "k:", label="perfectly calibrated")
    ax.set(xlabel="Predicted churn probability", ylabel="Observed churn rate", title="Calibration (test set)")
    ax.legend(); save("03_calibration.png")

    # Decision threshold by expected business value
    monthly = Xte["MonthlyCharges"].to_numpy(); yt = yte.to_numpy()
    ths = np.linspace(0.05, 0.9, 86)
    prof = np.array([retention_profit(yt, pbest, monthly, t) for t in ths])
    t_best = float(ths[prof.argmax()])
    contact_all = retention_profit(yt, np.ones_like(pbest), monthly, 0.0)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(ths, prof, color="#2563eb", lw=2)
    ax.axvline(t_best, color="#dc2626", ls="--", label=f"best threshold = {t_best:.2f}")
    ax.axhline(0, color="k", lw=1); ax.axhline(contact_all, color="#9ca3af", ls=":", label="contact everyone")
    ax.set(xlabel="Contact customers with churn probability above", ylabel="Expected profit on test set ($)",
           title=f"Choosing a threshold by business value ({best})")
    ax.legend(); save("04_threshold_profit.png")
    sc = scores(yte, pbest, t_best)
    out["threshold"] = {"value": t_best, "profit": float(prof.max()), "contact_everyone_profit": contact_all,
                        "precision": sc["precision"], "recall": sc["recall"],
                        "assumptions": {"offer_cost_usd": 30, "save_rate": 0.30, "months_value": 12}}
    ConfusionMatrixDisplay.from_predictions(yte, (pbest >= t_best).astype(int),
                                            display_labels=["Stayed", "Churned"], cmap="Blues", colorbar=False)
    plt.title(f"Confusion matrix at threshold {t_best:.2f}"); save("05_confusion_matrix.png")

    # Explanations: SHAP for the boosted model, permutation importance for the final model
    xgb = fitted["XGBoost"]
    prep = xgb.named_steps["prep"]
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    Xt = pd.DataFrame(prep.transform(Xte), columns=names)
    import xgboost as xgb_lib  # native TreeSHAP values (last column is the bias term)
    sv = xgb.named_steps["clf"].get_booster().predict(xgb_lib.DMatrix(Xt), pred_contribs=True)[:, :-1]
    plt.figure(); shap.summary_plot(sv, Xt, show=False, max_display=12, plot_size=(8.5, 5.2))
    plt.title("What pushes a customer toward leaving (SHAP, XGBoost)"); save("06_shap_summary.png")
    imp = permutation_importance(fitted[best], Xte, yte, scoring="roc_auc", n_repeats=10, random_state=SEED)
    order = imp.importances_mean.argsort()[::-1][:10]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh([Xte.columns[i] for i in order][::-1], imp.importances_mean[order][::-1],
            xerr=imp.importances_std[order][::-1], color="#2563eb")
    ax.set(xlabel="Drop in ROC-AUC when the column is shuffled", title=f"Permutation importance ({best})")
    save("07_permutation_importance.png")
    out["top_features"] = [{"feature": Xte.columns[i], "auc_drop": float(imp.importances_mean[i])} for i in order[:6]]

    # Export the logistic regression for the in-browser demo
    lr = fitted["Logistic Regression"]
    lp = lr.named_steps["prep"]; coef = lr.named_steps["clf"].coef_[0]
    sc_ = lp.named_transformers_["num"]; oh = lp.named_transformers_["cat"]
    cat_info = []
    fi = len(numeric)
    for col, cs, drop in zip(cats, oh.categories_, oh.drop_idx_):
        if drop is not None:  # binary column encoded as one 0/1 feature; 1 = the non-dropped category
            cat_info.append({"name": col, "values": [str(v) for v in cs],
                             "weights": {str(cs[drop]): 0.0, **{str(v): float(coef[fi]) for j, v in enumerate(cs) if j != drop}}})
            fi += 1
        else:
            cat_info.append({"name": col, "values": [str(v) for v in cs],
                             "weights": {str(v): float(coef[fi + j]) for j, v in enumerate(cs)}})
            fi += len(cs)
    for ci in cat_info:
        ci["freq"] = {str(k): float(v) for k, v in df[ci["name"]].value_counts(normalize=True).items()}
    demo = {"intercept": float(lr.named_steps["clf"].intercept_[0]),
            "numeric": [{"name": n, "mean": float(m), "scale": float(s), "coef": float(c),
                         "min": float(df[n].min()), "max": float(df[n].max()), "median": float(df[n].median())}
                        for n, m, s, c in zip(numeric, sc_.mean_, sc_.scale_, coef[:len(numeric)])],
            "categorical": cat_info,
            "threshold_note": "Probabilities come from the Logistic Regression model.",
            "test_example": {"row": Xte.iloc[0].to_dict(), "proba": float(lr.predict_proba(Xte.iloc[[0]])[0, 1])}}
    demo["test_example"]["row"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in demo["test_example"]["row"].items()}
    (ROOT / "reports" / "demo_model.json").write_text(json.dumps(demo, indent=1))
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out["threshold"], indent=1), best)


if __name__ == "__main__":
    main()
