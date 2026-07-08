"""
GramSeva Health — AI Triage Training Script using dataset.csv
=============================================================
This script parses the raw `dataset.csv` (disease,symptom1,symptom2,...)
into a one-hot encoded dataframe, trains the RandomForest model,
and exports all required artifacts for server.py.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "ml training")
DATASET_CSV = os.path.join(DATA_DIR, "dataset.csv")

MODEL_OUT       = os.path.join(BASE_DIR, "triage_model.pkl")
LE_OUT          = os.path.join(BASE_DIR, "label_encoder.pkl")
COLS_OUT        = os.path.join(BASE_DIR, "symptom_columns.json")
SPECIALIST_OUT  = os.path.join(BASE_DIR, "specialist_map.json")
METRICS_OUT     = os.path.join(BASE_DIR, "model_metrics.json")

# ─── SPECIALIST & URGENCY MAPPING ─────────────────────────────────────────────
SPECIALIST_MAP = {
    "Fungal infection":             "Dermatologist",
    "Allergy":                      "Allergist / Immunologist",
    "GERD":                         "Gastroenterologist",
    "Chronic cholestasis":          "Gastroenterologist",
    "Drug Reaction":                "Dermatologist",
    "Peptic ulcer diseae":          "Gastroenterologist",
    "AIDS":                         "Infectious Disease",
    "Diabetes ":                    "Endocrinologist",
    "Gastroenteritis":              "Gastroenterologist",
    "Bronchial Asthma":             "Pulmonologist",
    "Hypertension ":                "Cardiologist",
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
    "Diabetes ":                    "medium",
    "Hypertension ":                "medium",
    "Bronchial Asthma":             "medium",
    "Jaundice":                     "medium",
    "Chicken pox":                  "medium",
    "Migraine":                     "medium",
    "Urinary tract infection":      "medium",
    "GERD":                         "low",
    "Common Cold":                  "low",
    "Acne":                         "low",
    "Fungal infection":             "low",
    "Allergy":                      "low",
    "Psoriasis":                    "low",
    "Impetigo":                     "low",
}

# ─── DATA LOADING & PARSING ───────────────────────────────────────────────────
print(f"Reading {DATASET_CSV}...")
if not os.path.exists(DATASET_CSV):
    print(f"❌ ERROR: {DATASET_CSV} not found!")
    sys.exit(1)

with open(DATASET_CSV, "r") as f:
    lines = f.readlines()

parsed_data = []
all_symptoms = set()

for line in lines:
    line = line.strip()
    # Skip empty lines or headers consisting of only commas
    if not line or line.replace(",", "").strip() == "":
        continue
        
    parts = [p.strip() for p in line.split(",")]
    parts = [p for p in parts if p] # Remove empty parts
    if len(parts) < 2:
        continue
    
    disease = parts[0]
    symptoms = parts[1:]
    
    # Normalize symptom names 
    norm_symptoms = [s.strip().replace(" ", "_") for s in symptoms]
    
    parsed_data.append({"disease": disease, "symptoms": norm_symptoms})
    all_symptoms.update(norm_symptoms)

SYMPTOM_COLUMNS = sorted(list(all_symptoms))
print(f"   Parsed {len(parsed_data)} valid rows.")
print(f"   Found {len(SYMPTOM_COLUMNS)} unique symptoms.")

# Build one-hot encoded dataset
print("\nBuilding one-hot encoded dataset...")
df_data = []
for row in parsed_data:
    entry = {"prognosis": row["disease"]}
    for sym in SYMPTOM_COLUMNS:
        entry[sym] = 1 if sym in row["symptoms"] else 0
    df_data.append(entry)

df = pd.DataFrame(df_data)

print(f"   Shape (before dedup): {df.shape}")
print(f"   Unique Diseases: {df['prognosis'].nunique()}")

# ─── DIAGNOSTIC: DUPLICATE / LEAKAGE CHECK ────────────────────────────────────
# A perfect (or near-perfect) test accuracy combined with zero CV variance is a
# classic symptom of train/test leakage via duplicate rows. This dataset is
# known to contain many exact-duplicate symptom combinations per disease, so
# we check for that explicitly before splitting.
print("\n🔍 Diagnostic: checking for duplicate rows (train/test leakage risk)...")
n_total = len(df)
n_duplicate_rows = int(df.duplicated().sum())
n_unique_rows = int(df.drop_duplicates().shape[0])
print(f"   Total rows                 : {n_total}")
print(f"   Duplicate rows             : {n_duplicate_rows}  ({n_duplicate_rows / n_total:.1%} of total)")
print(f"   Unique symptom combinations: {n_unique_rows}")

print("\n   Unique symptom-combinations per disease:")
per_disease_unique = df.groupby("prognosis").apply(lambda g: g.drop_duplicates().shape[0])
for disease, n_unique in per_disease_unique.sort_values().items():
    total_for_disease = (df["prognosis"] == disease).sum()
    print(f"     {disease:<45s} {n_unique:>4d} unique / {total_for_disease:>4d} total rows")

if n_duplicate_rows / n_total > 0.5:
    print(
        "\n   ⚠️  Over half of all rows are exact duplicates. A train/test split on the "
        "raw data will almost certainly leak identical rows across the split boundary, "
        "inflating test accuracy. Proceeding with de-duplication before splitting."
    )

# ─── DEDUPLICATE BEFORE SPLITTING ─────────────────────────────────────────────
# This is the actual fix: remove exact-duplicate rows so the same symptom
# combination cannot appear in both the train and test sets.
df = df.drop_duplicates().reset_index(drop=True)
print(f"\n   Shape (after dedup): {df.shape}")

X = df.drop("prognosis", axis=1)
y = df["prognosis"]

print(f"\n   Final deduplicated dataset:")
print(f"   Shape: {df.shape}")
print(f"   Unique Diseases: {y.nunique()}")

# ─── LABEL ENCODE & SPLIT ─────────────────────────────────────────────────────
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Warn if any class has too few unique rows left for a stratified split
class_counts = pd.Series(y_enc).value_counts()
too_small = class_counts[class_counts < 2]
if not too_small.empty:
    small_diseases = [le.classes_[i] for i in too_small.index]
    print(
        f"\n   ⚠️  {len(small_diseases)} disease(s) have fewer than 2 unique rows after "
        f"dedup, so a stratified split isn't possible for them: {small_diseases}"
    )

stratify_arg = y_enc if too_small.empty else None
if stratify_arg is None:
    print("   Falling back to a non-stratified split.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=stratify_arg
)

# ─── MODEL TRAINING ───────────────────────────────────────────────────────────
print("\n🌲 Training RandomForest (300 trees)...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=2,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# ─── EVALUATION ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy (post-dedup): {test_acc * 100:.2f}%")

cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print(f"📊 CV Mean Accuracy (post-dedup): {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)

# Feature Importance
importances = model.feature_importances_
feat_imp = sorted(zip(SYMPTOM_COLUMNS, importances), key=lambda x: -x[1])
top_20 = feat_imp[:20]

# ─── SAVE ARTIFACTS ───────────────────────────────────────────────────────────
print("\n💾 Saving artifacts to allow server.py to use the new model...")
joblib.dump(model, MODEL_OUT)
joblib.dump(le, LE_OUT)

with open(COLS_OUT, "w") as f:
    json.dump(SYMPTOM_COLUMNS, f, indent=2)

with open(SPECIALIST_OUT, "w") as f:
    json.dump(SPECIALIST_MAP, f, indent=2)

metrics = {
    "test_accuracy": round(test_acc, 4),
    "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
    "cv_std": round(float(cv_scores.std()), 4),
    "n_diseases": len(le.classes_),
    "n_symptoms": len(SYMPTOM_COLUMNS),
    "n_rows_before_dedup": n_total,
    "n_duplicate_rows_removed": n_duplicate_rows,
    "n_rows_after_dedup": n_unique_rows,
    "top_20_features": [{"symptom": f, "importance": round(float(i), 4)} for f, i in top_20],
    "diseases": list(le.classes_)
}

with open(METRICS_OUT, "w") as f:
    json.dump(metrics, f, indent=2)

print("\n🎉 Training using dataset.csv complete!")
