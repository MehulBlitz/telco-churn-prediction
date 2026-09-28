"""
Download the IBM Telco Customer Churn dataset (7,043 rows) automatically.

Tries, in order:
  1. A local file already present in data/ (any of the common names).
  2. Downloading from IBM's public GitHub repo
     (https://github.com/IBM/telco-customer-churn-on-icp4d), which hosts the
     exact CSV (7,043 rows, 21 columns) used in IBM's official churn tutorial.

The resulting file is saved to data/telco_customer_churn.csv with the canonical
column layout:
    customerID, gender, SeniorCitizen, Partner, Dependents, tenure,
    PhoneService, MultipleLines, InternetService, OnlineSecurity,
    OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
    StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
    MonthlyCharges, TotalCharges, Churn

Usage:
    python download_data.py
"""

from __future__ import annotations

import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_PATH = DATA_DIR / "telco_customer_churn.csv"

EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]

RAW_URLS = [
    # IBM's official tutorial repo (Telco Churn, 7,043 rows)
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    # Well-known mirror used in many ML courses
    "https://raw.githubusercontent.com/dsrscientist/DSData/master/Telco%20Customer%20Churn.csv",
]


def looks_like_telco_csv(path: Path) -> bool:
    """Return True if `path` is a readable CSV with the canonical columns."""
    try:
        df = pd.read_csv(path, nrows=5)
    except Exception:
        return False
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    return not missing


def find_existing_file() -> Path | None:
    """Look for a previously downloaded copy inside data/."""
    if not DATA_DIR.exists():
        return None
    candidates = sorted(DATA_DIR.glob("*.csv"))
    for cand in candidates:
        if looks_like_telco_csv(cand):
            return cand
    return None


def download_from(url: str) -> pd.DataFrame:
    print(f"  Trying {url} ...")
    with urllib.request.urlopen(url, timeout=60) as resp:
        raw = resp.read()
    return pd.read_csv(io.BytesIO(raw))


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    existing = find_existing_file()
    if existing is not None:
        if existing.resolve() == OUT_PATH.resolve():
            print(f"[ok] Dataset already present: {existing}")
        else:
            df = pd.read_csv(existing)
            df.to_csv(OUT_PATH, index=False)
            print(f"[ok] Found local copy at {existing}; copied to {OUT_PATH}")
        return 0

    print("Dataset not found locally. Downloading IBM Telco Customer Churn ...")
    last_error: Exception | None = None
    for url in RAW_URLS:
        try:
            df = download_from(url)
            missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
            if missing:
                raise ValueError(f"Downloaded CSV is missing columns: {missing}")
            if len(df) < 7000:
                raise ValueError(f"Unexpected row count: {len(df)} (expected 7,043)")
            break
        except Exception as exc:  # noqa: BLE001 - try every mirror
            last_error = exc
            print(f"  [warn] failed: {exc}")
    else:
        print(
            "\n[error] Could not download the dataset automatically.\n"
            f"Last error: {last_error}\n\n"
            "Please download it manually:\n"
            "  1. Go to https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
            "  2. Download 'WA_Fn-UseC_-Telco-Customer-Churn.csv'\n"
            f"  3. Place it in {OUT_PATH} (rename if you like - any .csv in\n"
            "     data/ with the right columns will be picked up).\n"
        )
        return 1

    df.to_csv(OUT_PATH, index=False)
    print(f"[ok] Saved {len(df):,} rows x {len(df.columns)} columns -> {OUT_PATH}")
    print(f"     Churn distribution:\n{df['Churn'].value_counts().to_string()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
