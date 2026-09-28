# Telco Customer Churn — Classification & Model Evaluation

[![Retrain & refresh metrics](https://github.com/MehulBlitz/telco-churn-prediction/actions/workflows/update-metrics.yml/badge.svg)](https://github.com/MehulBlitz/telco-churn-prediction/actions/workflows/update-metrics.yml)
[![Live metrics dashboard](https://img.shields.io/badge/dashboard-MehulBlitz.github.io-2ea44f)](https://mehulblitz.github.io)

Predicts whether a telecom customer will churn on the public
[IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(7,043 records, 26.5% churn rate). The dataset is downloaded automatically
from IBM's public GitHub mirror by `download_data.py`.

**Stack:** Python · Pandas · Scikit-learn · Matplotlib · Seaborn · Gradio

## Results (reproduce with `python train_churn.py`, seed = 42)

Held-out test set: 1,409 customers (stratified 80/20 split).

| Model | 5-fold CV ROC-AUC | Test ROC-AUC | Churn precision | Churn recall | Churn F1 |
|---|---|---|---|---|---|
| Logistic Regression (tuned) | 0.8461 | 0.8442* | — | — | — |
| Random Forest (tuned) | 0.8484 | **0.8442** | 0.5531 | **0.7513** | **0.6372** |
| Gradient Boosting (tuned) | 0.8507 | — | — | — | — |

*winner per tuned-threshold F1; all three land within ~0.7 pt of each other on AUC.

**Headline: churn-class recall improved from 55.9% → 75.1%** (+19.3 points)
by combining `class_weight="balanced"` with a decision threshold tuned on
out-of-fold predictions, at essentially unchanged ROC-AUC (0.842 → 0.844):

| Configuration | Recall (churn) | Precision (churn) | F1 (churn) |
|---|---|---|---|
| Baseline LogReg, no class weights, thr = 0.50 | 0.5588 | 0.6572 | 0.6040 |
| Winner RF, class weights + tuned thr = 0.54 | **0.7513** | 0.5531 | **0.6372** |

Top churn drivers (`artifacts/feature_importance.csv`): monthly contract,
tenure, fiber-optic internet, total charges, online-security-related flags.

## What this project demonstrates

- **Leakage-free pipelines** — one-hot encoding and scaling live inside a
  sklearn `Pipeline` + `ColumnTransformer`, so every CV fold refits
  preprocessing on its training part only.
- **Model comparison done properly** — Logistic Regression, Random Forest
  and Gradient Boosting tuned with `GridSearchCV` under stratified 5-fold
  CV (scoring = ROC-AUC).
- **Threshold tuning without test-set leakage** — the operating threshold
  is chosen on out-of-fold predictions of the training split; the test set
  is scored exactly once at the end.
- **Interpretability** — top churn drivers extracted from the fitted model
  (feature importances / standardized |coefficients|).
- **Deployment** — interactive Gradio app with a live threshold slider.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

python download_data.py           # fetches the dataset (no Kaggle account needed)
python train_churn.py             # trains + tunes + saves artifacts/ and figures/
python app.py                     # interactive Gradio demo on http://127.0.0.1:7860
```

## Project structure

```
1_telco_churn/
├── download_data.py        # dataset auto-download + validation
├── train_churn.py          # pipelines, GridSearchCV, threshold tuning, figures
├── app.py                  # Gradio demo (probability, verdict, drivers)
├── requirements.txt
├── data/                   # telco_customer_churn.csv (downloaded)
├── artifacts/              # churn_model.joblib, metrics.json, feature_importance.csv
└── figures/                # EDA, ROC/PR curves, confusion matrices, importance
```

## Résumé bullet (numbers verified by this repo)

> Built an end-to-end classification pipeline on the IBM Telco Customer
> Churn dataset (7,043 records); trained and compared Logistic Regression,
> Random Forest and Gradient Boosting with GridSearchCV + stratified 5-fold
> CV; improved churn-class recall from **55.9% to 75.1%** using class
> weighting and decision-threshold tuning (test ROC-AUC 0.844).

See [SETUP_GITHUB.md](SETUP_GITHUB.md) to publish this folder as a repo.
