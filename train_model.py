"""
train_model.py
--------------
Trains a RandomForestClassifier DIRECTLY from
"Disease and symptoms dataset.csv" (must be in the project root).

Artifacts saved:
  model.pkl           - Trained RandomForest model
  label_encoder.pkl   - LabelEncoder for disease names
  symptoms_list.pkl   - Full ordered list of symptom feature columns (for prediction)
  ui_symptoms.pkl     - Top 40 most-common symptoms (for dashboard dropdown UI)

Run once before starting the Flask app:
    python train_model.py
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
DATASET_PATH = "Disease and symptoms dataset.csv"
MAX_ROWS     = 15_000    # Balanced: good accuracy + manageable model size
N_TREES      = 100
MAX_DEPTH    = 20        # Prevents over-fitting & keeps .pkl small
RANDOM_STATE = 42
TEST_SIZE    = 0.20
TOP_UI_SYMPTOMS = 50     # Number of symptoms shown in dashboard dropdown
# ─────────────────────────────────────────────────────────────────────────────


def load_and_clean(path, max_rows):
    """Load CSV, normalize columns, remove bad rows."""
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Dataset not found: '{path}'\n"
                 "Place 'Disease and symptoms dataset.csv' in the project root.")

    print(f"[1/4] Loading '{path}' (up to {max_rows:,} rows) ...")
    df = pd.read_csv(path, nrows=max_rows)

    # Normalize column names
    df.columns = (df.columns
                  .str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace(r"[^\w]", "_", regex=True))

    # Clean the disease label
    df["diseases"] = df["diseases"].astype(str).str.strip().str.lower()
    df = df[df["diseases"].notna() & (df["diseases"] != "") & (df["diseases"] != "nan")]

    print(f"      Rows: {len(df):,} | Unique diseases: {df['diseases'].nunique():,}")
    return df


def prepare_features(df):
    """Build X, y; drop rare classes; re-encode labels."""
    symptom_cols = [c for c in df.columns if c != "diseases"]

    # Force all symptom values to numeric 0/1
    X = df[symptom_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.int8)
    y_raw = df["diseases"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)

    # Remove disease classes with fewer than 2 samples (cannot stratify)
    counts = np.bincount(y_enc)
    valid_mask = counts[y_enc] >= 2
    X, y_enc = X[valid_mask], y_enc[valid_mask]

    # Re-fit encoder so class indices stay contiguous after masking
    le = LabelEncoder()
    le.fit(y_raw[valid_mask])
    y_final = le.transform(y_raw[valid_mask])

    print(f"[2/4] Features: {X.shape[1]} | Classes kept: {len(le.classes_):,} | "
          f"Training samples: {len(X):,}")
    return X, y_final, le, symptom_cols


def select_ui_symptoms(df, symptom_cols, n=TOP_UI_SYMPTOMS):
    """Return the N most-common symptoms for the dropdown UI."""
    X_all = df[symptom_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    top = X_all.sum().sort_values(ascending=False).head(n).index.tolist()
    return top


def train(X, y):
    """Fit RandomForestClassifier, return model + accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"[3/4] Training RandomForest (n_estimators={N_TREES}, "
          f"max_depth={MAX_DEPTH}, n_jobs=-1) ...")

    clf = RandomForestClassifier(
        n_estimators=N_TREES,
        max_depth=MAX_DEPTH,
        n_jobs=-1,            # use all CPU cores
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(f"      Test Accuracy: {acc:.4f}  ({acc * 100:.2f}%)")
    return clf


def save_artifacts(clf, le, symptom_cols, ui_symptoms):
    """Persist all model artifacts."""
    print("[4/4] Saving artifacts ...")
    joblib.dump(clf,          "model.pkl",          compress=3)
    joblib.dump(le,           "label_encoder.pkl")
    joblib.dump(symptom_cols, "symptoms_list.pkl")
    joblib.dump(ui_symptoms,  "ui_symptoms.pkl")
    print("      Saved: model.pkl | label_encoder.pkl | "
          "symptoms_list.pkl | ui_symptoms.pkl")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df               = load_and_clean(DATASET_PATH, MAX_ROWS)
    X, y, le, cols   = prepare_features(df)
    ui_syms          = select_ui_symptoms(df, cols)
    clf              = train(X, y)
    save_artifacts(clf, le, cols, ui_syms)
    print("\nDone! Run: python app.py")
