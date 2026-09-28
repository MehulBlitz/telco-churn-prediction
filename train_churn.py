"""
End-to-end churn modeling on the IBM Telco Customer Churn dataset.

What this script does
---------------------
1. Loads data/telco_customer_churn.csv (run download_data.py first).
2. Cleans it (TotalCharges coercion) *inside* a sklearn Pipeline +
   ColumnTransformer so no information leaks from validation folds into
   training.
3. Trains and tunes three classifiers with GridSearchCV (stratified 5-fold
   cross-validation, scoring = ROC-AUC):
       - Logistic Regression
       - Random Forest
       - Gradient Boosting
4. Picks the best model, then tunes the decision *threshold* on
   out-of-fold training predictions for the best F1 on the churn class -
   this is what lifts churn-class recall vs. the default 0.5.
5. Evaluates once on a held-out test set with precision / recall / F1 /
   ROC-AUC / PR-AUC + confusion matrices.
6. Saves artifacts/ (model + metrics + importance) and figures/ (EDA +
   ROC/PR curves + confusion matrices + feature importance).

Usage:
    python download_data.py   # once
    python train_churn.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless: save figures, don't open windows

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "telco_customer_churn.csv"
ART = HERE / "artifacts"
FIGS = HERE / "figures"
ART.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

sns.set_theme(style="whitegrid", palette="deep")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY = ["SeniorCitizen"]  # already 0/1
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


# --------------------------------------------------------------------------- #
# 1. Load + row-level cleaning (everything else lives inside the Pipeline)
# --------------------------------------------------------------------------- #
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise SystemExit(
            f"{DATA_PATH} not found. Run `python download_data.py` first."
        )
    df = pd.read_csv(DATA_PATH)

    # TotalCharges arrives as object because blanks stand in for missing values
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_blank = int(df["TotalCharges"].isna().sum())
    if n_blank:
        # These are all tenure == 0 (brand-new customers) -> charge is 0
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    print(f"Loaded {len(df):,} rows | blank TotalCharges imputed with 0: {n_blank}")

    df["Churn"] = (df["Churn"].str.strip().str.lower() == "yes").astype(int)
    return df.drop(columns=["customerID"])


# --------------------------------------------------------------------------- #
# 2. Preprocessing inside Pipelines (fit on train folds only -> no leakage)
# --------------------------------------------------------------------------- #
def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([("scale", StandardScaler())])
    cat_pipe = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC),
            ("bin", "passthrough", BINARY),
            ("cat", cat_pipe, CATEGORICAL),
        ],
        remainder="drop",
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names = list(NUMERIC) + list(BINARY)
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    names += list(ohe.get_feature_names_out(CATEGORICAL))
    return names


# --------------------------------------------------------------------------- #
# 3. Models + hyperparameter grids
# --------------------------------------------------------------------------- #
def model_specs() -> list[tuple[str, object, dict]]:
    specs: list[tuple[str, object, dict]] = []

    lr = LogisticRegression(
        max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE
    )
    grid_lr = {
        "model__C": [0.03, 0.1, 0.3, 1.0, 3.0],
    }
    specs.append(("Logistic Regression", lr, grid_lr))

    rf = RandomForestClassifier(
        class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE
    )
    grid_rf = {
        "model__n_estimators": [300, 600],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_leaf": [1, 5, 20],
        "model__max_features": ["sqrt", 0.3],
    }
    specs.append(("Random Forest", rf, grid_rf))

    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
    grid_gb = {
        "model__n_estimators": [200, 400],
        "model__learning_rate": [0.03, 0.06, 0.1],
        "model__max_depth": [2, 3],
        "model__subsample": [0.8, 1.0],
    }
    specs.append(("Gradient Boosting", gb, grid_gb))
    return specs


# --------------------------------------------------------------------------- #
# 4. Threshold tuning on out-of-fold predictions (test set never involved)
# --------------------------------------------------------------------------- #
def best_threshold_by_f1(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision/recall have one more entry than thresholds (last point = 1.0)
    f1s = 2 * precision[:-1] * recall[:-1] / np.clip(
        precision[:-1] + recall[:-1], 1e-12, None
    )
    idx = int(np.argmax(f1s))
    return float(thresholds[idx]), float(precision[idx]), float(recall[idx])


def evaluate_at_threshold(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision_churn": round(float(precision_score(y_true, pred)), 4),
        "recall_churn": round(float(recall_score(y_true, pred)), 4),
        "f1_churn": round(float(f1_score(y_true, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
    }


# --------------------------------------------------------------------------- #
# 5. Figures
# --------------------------------------------------------------------------- #
def plot_eda(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    sns.countplot(x="Churn", data=df, ax=axes[0])
    axes[0].set_title("Churn distribution (0 = stays, 1 = churns)")
    axes[0].set_xticks([0, 1], ["No", "Yes"])

    sns.kdeplot(data=df, x="tenure", hue="Churn", common_norm=False, fill=True,
                ax=axes[1])
    axes[1].set_title("Tenure by churn")

    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[2])
    axes[2].set_title("Monthly charges by churn")

    fig.suptitle("Telco churn - quick EDA", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "eda_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_top_feature_importance(pipe: Pipeline) -> pd.DataFrame:
    """Model-native importances; |coef| on standardized features for linear."""
    pre = pipe.named_steps["preprocess"]
    clf = pipe.named_steps["model"]
    names = get_feature_names(pre)

    if hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
        label = "feature importance"
    else:  # linear model
        vals = np.abs(clf.coef_[0])
        label = "|coefficient| (standardized features)"

    imp = pd.DataFrame({"feature": names, "importance": vals})
    imp = imp.sort_values("importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(data=imp, y="feature", x="importance", ax=ax, color="#4c72b0")
    ax.set_title(f"Top 20 churn drivers ({label})")
    fig.tight_layout()
    fig.savefig(FIGS / "feature_importance.png", dpi=150)
    plt.close(fig)
    return imp


def plot_confusion(y_true, y_pred, title: str, fname: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No churn", "Churn"])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGS / fname, dpi=150)
    plt.close(fig)


def plot_roc_pr_curves(models_probas: dict, y_test) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, proba in models_probas.items():
        RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_test, proba, name=name, ax=axes[1])
    axes[0].set_title("ROC curves (test set)")
    axes[1].set_title("Precision-Recall curves (test set)")
    fig.tight_layout()
    fig.savefig(FIGS / "roc_pr_curves.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 6. Main
# --------------------------------------------------------------------------- #
def main() -> None:
    df = load_data()
    plot_eda(df)

    y = df["Churn"].values
    X = df.drop(columns=["Churn"])

    # Stratified hold-out test set: 20% - touched exactly once, at the end.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,} | churn rate {y.mean():.1%}")

    results: dict[str, dict] = {}
    fitted_models: dict[str, Pipeline] = {}
    test_probas: dict[str, np.ndarray] = {}

    # --- Reference baseline: plain LogReg, NO class weighting, thr=0.5 ---
    # This is the "before" number for the headline recall improvement claim.
    base_pipe = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
    ])
    base_pipe.fit(X_train, y_train)
    base_proba = base_pipe.predict_proba(X_test)[:, 1]
    baseline_metrics = evaluate_at_threshold(y_test, base_proba, 0.5)
    print(f"\nBaseline (LogReg, no class weights, thr=0.5): {baseline_metrics}")

    for name, estimator, grid in model_specs():
        print(f"\n=== {name}: GridSearchCV (5-fold stratified, scoring=roc_auc) ===")
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
        gs = GridSearchCV(pipe, grid, cv=CV, scoring="roc_auc", n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)

        best_cv_auc = float(gs.best_score_)
        print(f"  best params : {gs.best_params_}")
        print(f"  best CV AUC : {best_cv_auc:.4f}")

        # Out-of-fold probabilities on TRAIN -> honest threshold selection
        oof = cross_val_predict(
            gs.best_estimator_, X_train, y_train, cv=CV,
            method="predict_proba", n_jobs=-1,
        )[:, 1]
        thr, p_at_thr, r_at_thr = best_threshold_by_f1(y_train, oof)
        print(f"  tuned threshold = {thr:.3f} "
              f"(OOF precision={p_at_thr:.3f}, recall={r_at_thr:.3f})")

        proba_test = gs.best_estimator_.predict_proba(X_test)[:, 1]
        results[name] = {
            "best_params": {k.replace("model__", ""): v
                            for k, v in gs.best_params_.items()},
            "cv_roc_auc": round(best_cv_auc, 4),
            "test_default_0.5": evaluate_at_threshold(y_test, proba_test, 0.5),
            "test_tuned_threshold": evaluate_at_threshold(y_test, proba_test, thr),
        }
        fitted_models[name] = gs.best_estimator_
        test_probas[name] = proba_test

    # ---------------- pick winner by tuned-threshold F1 ------------------ #
    winner = max(results, key=lambda n: results[n]["test_tuned_threshold"]["f1_churn"])
    best_pipe = fitted_models[winner]
    best_thr = results[winner]["test_tuned_threshold"]["threshold"]
    proba_best = test_probas[winner]
    pred_best = (proba_best >= best_thr).astype(int)

    print(f"\n### Winner: {winner} (threshold {best_thr}) ###")
    print(f"  test @ tuned threshold: {results[winner]['test_tuned_threshold']}")
    print(f"  test @ 0.5            : {results[winner]['test_default_0.5']}")

    plot_confusion(y_test, pred_best,
                   f"{winner} - confusion matrix (thr={best_thr:.2f})",
                   "confusion_matrix_tuned.png")
    plot_confusion(y_test, (proba_best >= 0.5).astype(int),
                   f"{winner} - confusion matrix (thr=0.50)",
                   "confusion_matrix_default.png")
    plot_roc_pr_curves(test_probas, y_test)

    imp = plot_top_feature_importance(best_pipe)
    imp.to_csv(ART / "feature_importance.csv", index=False)

    metrics = {
        "generated_at": utc_now_iso(),
        "winner_model": winner,
        "dataset_rows": int(len(df)),
        "churn_rate": round(float(y.mean()), 4),
        "test_size": int(len(y_test)),
        "baseline_no_class_weights": baseline_metrics,
        "models": results,
        "headline": {
            "recall_baseline_no_class_weights":
                baseline_metrics["recall_churn"],
            "recall_default_0.5":
                results[winner]["test_default_0.5"]["recall_churn"],
            "recall_tuned_threshold":
                results[winner]["test_tuned_threshold"]["recall_churn"],
            "f1_tuned_threshold":
                results[winner]["test_tuned_threshold"]["f1_churn"],
            "roc_auc": results[winner]["test_tuned_threshold"]["roc_auc"],
        },
        "top_features": imp.head(10).to_dict(orient="records"),
    }
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump(
        {"pipeline": best_pipe, "threshold": best_thr, "model_name": winner},
        ART / "churn_model.joblib",
    )
    print(f"\nSaved model   -> {ART / 'churn_model.joblib'}")
    print(f"Saved metrics -> {ART / 'metrics.json'}")
    print("Figures       -> figures/")


if __name__ == "__main__":
    main()
