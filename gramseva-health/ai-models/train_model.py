"""
GramSeva Health — AI Triage Model Training Script
==================================================
Robust ML pipeline with cross-validation, feature importance,
and comprehensive metrics export.

Dataset: 132 binary symptom features → 41 disease classes
Model:   RandomForestClassifier (300 trees)

Run:
    python train_model.py --dataset-dir /path/to/ml_training_dir
    # or set the GRAMSEVA_DATASET_DIR environment variable instead of --dataset-dir
    # (the directory must contain Training.csv and Testing.csv)
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_dataset_dir() -> str:
    """
    Resolve the directory containing Training.csv / Testing.csv from
    (in priority order):
      1. --dataset-dir CLI argument
      2. GRAMSEVA_DATASET_DIR environment variable
      3. ../../ml training relative to this script (original fallback default)
    """
    parser = argparse.ArgumentParser(description="Train GramSeva triage RandomForest model")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to the directory containing Training.csv and Testing.csv "
             "(overrides GRAMSEVA_DATASET_DIR env var)",
    )
    args, _ = parser.parse_known_args()

    if args.dataset_dir:
        return os.path.abspath(args.dataset_dir)

    env_dir = os.environ.get("GRAMSEVA_DATASET_DIR")
    if env_dir:
        return os.path.abspath(env_dir)

    return os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "ml training")


DATA_DIR   = resolve_dataset_dir()
TRAIN_CSV  = os.path.join(DATA_DIR, "Training.csv")
TEST_CSV   = os.path.join(DATA_DIR, "Testing.csv")

# Output artifacts
MODEL_OUT       = os.path.join(BASE_DIR, "triage_model.pkl")
LE_OUT          = os.path.join(BASE_DIR, "label_encoder.pkl")
COLS_OUT        = os.path.join(BASE_DIR, "symptom_columns.json")
SPECIALIST_OUT  = os.path.join(BASE_DIR, "specialist_map.json")
METRICS_OUT     = os.path.join(BASE_DIR, "model_metrics.json")

# ─── VALIDATE FILES ──────────────────────────────────────────────────────────
for path, name in [(TRAIN_CSV, "Training.csv"), (TEST_CSV, "Testing.csv")]:
    if not os.path.exists(path):
        print(f"❌ ERROR: {name} not found at {path}")
        print("   Pass --dataset-dir, or set GRAMSEVA_DATASET_DIR, to point at the "
              "directory containing Training.csv and Testing.csv.")
        sys.exit(1)

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
print("=" * 60)
print("  GramSeva Health — AI Triage Model Training")
print("=" * 60)

print("\n📂 Loading datasets...")
train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)

# Drop unnamed garbage columns
train.drop(columns=[c for c in train.columns if "Unnamed" in c], inplace=True)
test.drop(columns=[c for c in test.columns if "Unnamed" in c],  inplace=True)

# Strip whitespace from column names
train.columns = [c.strip() for c in train.columns]
test.columns  = [c.strip() for c in test.columns]

# Strip whitespace from prognosis values
train["prognosis"] = train["prognosis"].str.strip()
test["prognosis"]  = test["prognosis"].str.strip()

print(f"   Train : {train.shape[0]} rows, {train.shape[1]-1} symptom features")
print(f"   Test  : {test.shape[0]} rows")
print(f"   Diseases: {train['prognosis'].nunique()}")

# ─── FEATURES / LABELS ──────────────────────────────────────────────────────
X_train = train.drop("prognosis", axis=1)
y_train = train["prognosis"]
X_test  = test.drop("prognosis", axis=1)
y_test  = test["prognosis"]

SYMPTOM_COLUMNS = list(X_train.columns)

# ─── DIAGNOSTIC: DUPLICATE / TRAIN-TEST OVERLAP CHECK ────────────────────────
# A perfect (or near-perfect) test accuracy combined with zero CV variance is a
# classic symptom of leakage. There are two distinct ways it can happen here:
#   1. Training.csv itself contains many exact-duplicate rows per disease, so a
#      CV fold split can put a row's twin in the training half and the
#      original in the validation half of the *same* fold.
#   2. Testing.csv's "held-out" rows are themselves duplicates of rows already
#      present in Training.csv, so the model has already seen them.
# Both are checked explicitly below rather than assumed.
print("\n🔍 Diagnostic: checking for duplication and train/test overlap...")

train_full = pd.concat([X_train, y_train], axis=1)
test_full  = pd.concat([X_test, y_test], axis=1)

n_train_total = len(train_full)
n_train_dupes = int(train_full.duplicated().sum())
n_train_unique = int(train_full.drop_duplicates().shape[0])
print(f"   Training.csv total rows        : {n_train_total}")
print(f"   Training.csv duplicate rows    : {n_train_dupes}  ({n_train_dupes / n_train_total:.1%} of total)")
print(f"   Training.csv unique combos     : {n_train_unique}")

print("\n   Unique symptom-combinations per disease (Training.csv):")
per_disease_unique = train_full.groupby("prognosis").apply(lambda g: g.drop_duplicates().shape[0])
for disease, n_unique in per_disease_unique.sort_values().items():
    total_for_disease = (train_full["prognosis"] == disease).sum()
    print(f"     {disease:<45s} {n_unique:>4d} unique / {total_for_disease:>4d} total rows")

train_row_set = set(map(tuple, train_full.values.tolist()))
test_row_set  = set(map(tuple, test_full.values.tolist()))
overlap = train_row_set & test_row_set
n_overlap = len(overlap)
print(f"\n   Testing.csv rows also present in Training.csv: {n_overlap} / {len(test_row_set)}")

if n_train_dupes / n_train_total > 0.5:
    print(
        "\n   ⚠️  Over half of Training.csv rows are exact duplicates. Any CV split on "
        "this data risks a row's duplicate landing in the opposite fold, inflating CV "
        "accuracy independent of true generalization."
    )
if n_overlap > 0:
    print(
        f"\n   ⚠️  {n_overlap} of the {len(test_row_set)} 'held-out' Testing.csv rows are "
        "identical to rows already in Training.csv. Test accuracy on those rows reflects "
        "memorization, not generalization to unseen cases."
    )
if n_train_dupes / n_train_total <= 0.5 and n_overlap == 0:
    print("\n   ✅ No major duplication or train/test overlap detected.")

# ─── LABEL ENCODE ─────────────────────────────────────────────────────────────
le = LabelEncoder()
le.fit(pd.concat([y_train, y_test]).unique())
y_train_enc = le.transform(y_train)
y_test_enc  = le.transform(y_test)

print(f"\n🏷️  Classes ({len(le.classes_)}): {list(le.classes_[:5])} ...")

# ─── TRAIN ────────────────────────────────────────────────────────────────────
print("\n🌲 Training RandomForest (300 trees, max_depth=20)...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=2,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
model.fit(X_train, y_train_enc)
print("   ✅ Training complete!")

# ─── CROSS-VALIDATION ────────────────────────────────────────────────────────
print("\n📊 Running 5-fold cross-validation...")
cv_scores = cross_val_score(model, X_train, y_train_enc, cv=5, scoring="accuracy")
print(f"   CV Scores : {[f'{s:.4f}' for s in cv_scores]}")
print(f"   CV Mean   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─── EVALUATE ON TEST SET ────────────────────────────────────────────────────
y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test_enc, y_pred)

print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")
print("\n📋 Classification Report:")
report_str = classification_report(y_test_enc, y_pred, target_names=le.classes_)
print(report_str)

# Parse classification report into dict
report_dict = classification_report(y_test_enc, y_pred, target_names=le.classes_, output_dict=True)

# ─── FEATURE IMPORTANCE ──────────────────────────────────────────────────────
print("\n🔬 Top 20 Most Important Symptoms:")
importances = model.feature_importances_
feat_imp = sorted(zip(SYMPTOM_COLUMNS, importances), key=lambda x: -x[1])
top_20 = feat_imp[:20]
for i, (feat, imp) in enumerate(top_20, 1):
    bar = "█" * int(imp * 200)
    print(f"   {i:2d}. {feat:<35s} {imp:.4f} {bar}")

# ─── SPECIALIST MAPPING ─────────────────────────────────────────────────────
SPECIALIST_MAP = {
    "Fungal infection":             "Dermatologist",
    "Allergy":                      "Allergist / Immunologist",
    "GERD":                         "Gastroenterologist",
    "Chronic cholestasis":          "Gastroenterologist",
    "Drug Reaction":                "Dermatologist",
    "Peptic ulcer diseae":          "Gastroenterologist",
    "AIDS":                         "Infectious Disease",
    "Diabetes":                     "Endocrinologist",
    "Gastroenteritis":              "Gastroenterologist",
    "Bronchial Asthma":             "Pulmonologist",
    "Hypertension":                 "Cardiologist",
    "Migraine":                     "Neurologist",
    "Cervical spondylosis":         "Orthopedic",
    "Paralysis (brain hemorrhage)": "Neurologist",
    "Jaundice":                     "Hepatologist",
    "Malaria":                      "Infectious Disease",
    "Chicken pox":                  "General Physician",
    "Dengue":                       "Infectious Disease",
    "Typhoid":                      "Infectious Disease",
    "hepatitis A":                  "Hepatologist",
    "Hepatitis B":                  "Hepatologist",
    "Hepatitis C":                  "Hepatologist",
    "Hepatitis D":                  "Hepatologist",
    "Hepatitis E":                  "Hepatologist",
    "Alcoholic hepatitis":          "Hepatologist",
    "Tuberculosis":                 "Pulmonologist",
    "Common Cold":                  "General Physician",
    "Pneumonia":                    "Pulmonologist",
    "Dimorphic hemmorhoids(piles)": "Proctologist",
    "Heart attack":                 "Cardiologist",
    "Varicose veins":               "Vascular Surgeon",
    "Hypothyroidism":               "Endocrinologist",
    "Hyperthyroidism":              "Endocrinologist",
    "Hypoglycemia":                 "Endocrinologist",
    "Osteoarthristis":              "Orthopedic",
    "Arthritis":                    "Rheumatologist",
    "(vertigo) Paroymsal  Positional Vertigo": "ENT / Neurologist",
    "Acne":                         "Dermatologist",
    "Urinary tract infection":      "Urologist",
    "Psoriasis":                    "Dermatologist",
    "Impetigo":                     "Dermatologist",
}

# ─── URGENCY MAPPING ─────────────────────────────────────────────────────────
URGENCY_MAP = {
    "Heart attack":                 "critical",
    "Paralysis (brain hemorrhage)": "critical",
    "Dengue":                       "high",
    "Malaria":                      "high",
    "Tuberculosis":                 "high",
    "Pneumonia":                    "high",
    "Hepatitis B":                  "high",
    "Hepatitis C":                  "high",
    "AIDS":                         "high",
    "Typhoid":                      "medium",
    "Diabetes":                     "medium",
    "Hypertension":                 "medium",
    "Bronchial Asthma":             "medium",
    "Jaundice":                     "medium",
    "Chicken pox":                  "medium",
    "GERD":                         "low",
    "Common Cold":                  "low",
    "Acne":                         "low",
    "Fungal infection":             "low",
    "Allergy":                      "low",
    "Migraine":                     "medium",
    "Urinary tract infection":      "medium",
    "Psoriasis":                    "low",
    "Impetigo":                     "low",
}

# ─── SAVE ARTIFACTS ──────────────────────────────────────────────────────────
print("\n💾 Saving model artifacts...")

joblib.dump(model, MODEL_OUT)
print(f"   Model         → {MODEL_OUT}")

joblib.dump(le, LE_OUT)
print(f"   LabelEncoder  → {LE_OUT}")

with open(COLS_OUT, "w") as f:
    json.dump(SYMPTOM_COLUMNS, f, indent=2)
print(f"   Symptom cols  → {COLS_OUT}")

with open(SPECIALIST_OUT, "w") as f:
    json.dump(SPECIALIST_MAP, f, indent=2)
print(f"   Specialist map→ {SPECIALIST_OUT}")

# Save comprehensive metrics
metrics = {
    "test_accuracy": round(test_acc, 4),
    "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
    "cv_std": round(float(cv_scores.std()), 4),
    "cv_scores": [round(float(s), 4) for s in cv_scores],
    "n_diseases": int(len(le.classes_)),
    "n_symptoms": len(SYMPTOM_COLUMNS),
    "n_train_samples": int(X_train.shape[0]),
    "n_test_samples": int(X_test.shape[0]),
    "n_train_duplicate_rows": n_train_dupes,
    "n_train_unique_combinations": n_train_unique,
    "n_test_rows_overlapping_train": n_overlap,
    "model_params": {
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_leaf": 2,
        "min_samples_split": 5,
    },
    "top_20_features": [{"symptom": f, "importance": round(float(i), 4)} for f, i in top_20],
    "diseases": list(le.classes_),
    "urgency_map": URGENCY_MAP,
    "per_class_report": {
        k: {
            "precision": round(v["precision"], 4),
            "recall": round(v["recall"], 4),
            "f1_score": round(v["f1-score"], 4),
        }
        for k, v in report_dict.items()
        if k not in ("accuracy", "macro avg", "weighted avg")
    },
}

with open(METRICS_OUT, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"   Metrics       → {METRICS_OUT}")

print("\n" + "=" * 60)
print(f"  🎉 Training complete! Test accuracy: {test_acc*100:.2f}%")
print("=" * 60)
