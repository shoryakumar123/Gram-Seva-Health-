#!/usr/bin/env python3
"""
GramSeva Health — Skin Disease Detection Model Training
========================================================
Pipeline: Images → Feature extraction → PCA → RandomForest

Usage:
    # Activate the virtual environment first:
    source .venv/bin/activate
    python train_skin_model.py --dataset-dir /path/to/dataset
    # or set the GRAMSEVA_DATASET_DIR environment variable instead of --dataset-dir

Requirements:
    pip install -r requirements.txt
    # or: pip install scikit-learn pillow numpy joblib
"""

from __future__ import annotations  # enables tuple[...] syntax on Python < 3.10

import argparse
import json
import logging
import os
import sys
import time

# ─── DEPENDENCY CHECK ─────────────────────────────────────────────────────────
_REQUIRED = {
    "numpy":        "numpy",
    "PIL":          "Pillow",
    "sklearn":      "scikit-learn",
    "joblib":       "joblib",
}

_missing: list[str] = []
for _mod, _pkg in _REQUIRED.items():
    try:
        __import__(_mod)
    except ImportError:
        _missing.append(_pkg)

if _missing:
    print(
        "❌  Missing required packages: " + ", ".join(_missing) + "\n"
        "\n"
        "Install them with:\n"
        "    pip install " + " ".join(_missing) + "\n"
        "\n"
        "Or, if using the project's virtual environment:\n"
        "    source .venv/bin/activate\n"
        "    pip install -r requirements.txt\n",
        file=sys.stderr,
    )
    sys.exit(1)

import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("gramseva.train")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_dataset_dir() -> str:
    """
    Resolve the dataset directory from (in priority order):
      1. --dataset-dir CLI argument
      2. GRAMSEVA_DATASET_DIR environment variable
      3. ./dataset relative to this script (fallback default)
    """
    parser = argparse.ArgumentParser(description="Train GramSeva skin-disease PCA+RandomForest model")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to the class-subfolder dataset directory "
             "(overrides GRAMSEVA_DATASET_DIR env var)",
    )
    args, _ = parser.parse_known_args()

    if args.dataset_dir:
        return os.path.abspath(args.dataset_dir)

    env_dir = os.environ.get("GRAMSEVA_DATASET_DIR")
    if env_dir:
        return os.path.abspath(env_dir)

    return os.path.join(BASE_DIR, "dataset")


DATASET_DIR = resolve_dataset_dir()

IMG_SIZE     = 128   # Resize all images to IMG_SIZE × IMG_SIZE
HIST_BINS    = 16    # Colour histogram bins per channel
N_PCA_COMP   = 200   # Max PCA components (auto-capped to dataset size)
N_ESTIMATORS = 500   # RandomForest trees
MAX_DEPTH    = 30
TEST_SIZE    = 0.2
RANDOM_STATE = 42
CV_FOLDS     = 5
MIN_CV_FOLD_SAMPLES = 2  # warn if a class has fewer samples than CV_FOLDS

MODEL_OUT   = os.path.join(BASE_DIR, "skin_model.pkl")
LE_OUT      = os.path.join(BASE_DIR, "skin_label_encoder.pkl")
CLASSES_OUT = os.path.join(BASE_DIR, "skin_classes.json")
METRICS_OUT = os.path.join(BASE_DIR, "skin_metrics.json")

# ─── IMAGE VALID EXTENSIONS ───────────────────────────────────────────────────
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ─── FEATURE EXTRACTION ───────────────────────────────────────────────────────
def extract_features(arr: np.ndarray) -> list[float]:
    """
    Build a feature vector from a normalised (H × W × 3) float32 array.

    Features (in order):
      1. Raw flattened pixel values  [H*W*3 values]
      2. Per-channel mean, std, median  [9 values]
      3. Normalised colour histogram per channel  [HIST_BINS * 3 values]
    """
    features: list[float] = []

    # 1. Raw pixels
    features.extend(arr.flatten().tolist())

    # 2. Per-channel statistics
    for c in range(3):
        ch = arr[:, :, c]
        features.append(float(ch.mean()))
        features.append(float(ch.std()))
        features.append(float(np.median(ch)))

    # 3. Colour histograms — guard against all-zero bins (divide-by-zero)
    for c in range(3):
        hist, _ = np.histogram(arr[:, :, c], bins=HIST_BINS, range=(0.0, 1.0))
        total = hist.sum()
        norm_hist = (hist / total).tolist() if total > 0 else [0.0] * HIST_BINS
        features.extend(norm_hist)

    return features


# ─── DATA LOADING ─────────────────────────────────────────────────────────────
def load_dataset(dataset_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Walk *dataset_dir*, treating each sub-folder as a class label.

    Returns:
        X : float32 ndarray, shape (n_samples, n_features)
        y : str ndarray,     shape (n_samples,)

    Raises:
        FileNotFoundError  – dataset_dir does not exist
        ValueError         – no valid images were found
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            "Please check DATASET_DIR in the config section, pass --dataset-dir, "
            "or set the GRAMSEVA_DATASET_DIR environment variable."
        )

    X: list[list[float]] = []
    y: list[str] = []
    skipped = 0

    class_dirs = sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
        and not d.startswith(".")
        and d != "__pycache__"
    )

    if not class_dirs:
        raise ValueError(f"No class sub-directories found in {dataset_dir}")

    for class_name in class_dirs:
        class_dir = os.path.join(dataset_dir, class_name)
        count = 0

        for img_name in sorted(os.listdir(class_dir)):
            # Skip non-image files early
            if os.path.splitext(img_name)[1].lower() not in VALID_EXT:
                continue
            img_path = os.path.join(class_dir, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                arr = np.array(img, dtype=np.float32) / 255.0
                X.append(extract_features(arr))
                y.append(class_name)
                count += 1
            except (UnidentifiedImageError, OSError) as exc:
                log.debug("Skipping %s — %s", img_path, exc)
                skipped += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("Unexpected error reading %s — %s", img_path, exc)
                skipped += 1

        log.info("  ✅ %-30s %4d images", class_name, count)

    if skipped:
        log.warning("  ⚠️  Skipped %d corrupt / unreadable images", skipped)

    if not X:
        raise ValueError(
            "No images were loaded. Make sure the dataset directory contains "
            "class sub-folders with valid image files."
        )

    return np.array(X, dtype=np.float32), np.array(y)


# ─── PIPELINE ─────────────────────────────────────────────────────────────────
def build_pipeline(n_samples: int, n_features: int) -> Pipeline:
    """Return a PCA + RandomForest sklearn Pipeline."""
    n_components = min(N_PCA_COMP, n_samples, n_features)
    log.info("  PCA components : %d", n_components)

    return Pipeline([
        ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
        ("clf", RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=2,
            min_samples_split=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


# ─── SAVE ARTIFACTS ───────────────────────────────────────────────────────────
def save_artifacts(
    model: Pipeline,
    le: LabelEncoder,
    class_names: list[str],
    metrics: dict,
) -> None:
    """Persist model, label encoder, class list, and metrics to disk."""
    joblib.dump(model, MODEL_OUT)
    log.info("  Model    → %s", MODEL_OUT)

    joblib.dump(le, LE_OUT)
    log.info("  Encoder  → %s", LE_OUT)

    with open(CLASSES_OUT, "w", encoding="utf-8") as fh:
        json.dump(class_names, fh, indent=2)
    log.info("  Classes  → %s", CLASSES_OUT)

    with open(METRICS_OUT, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    log.info("  Metrics  → %s", METRICS_OUT)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main() -> None:
    t_start = time.perf_counter()
    sep = "=" * 60
    print(f"\n{sep}")
    print("  GramSeva — Skin Disease Model Training")
    print(sep)
    print(f"  Dataset : {DATASET_DIR}")
    print(f"  Img size: {IMG_SIZE}×{IMG_SIZE}\n")

    # ── Load ─────────────────────────────────────────────────────────────────
    log.info("📂 Loading images…")
    X, y = load_dataset(DATASET_DIR)
    log.info("  Total samples  : %d", len(X))
    log.info("  Feature vector : %d dims", X.shape[1])

    # ── Encode labels ────────────────────────────────────────────────────────
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names: list[str] = [str(c) for c in le.classes_]
    log.info("  Classes        : %s", class_names)

    # Warn if any class has very few samples (may break CV)
    unique, counts = np.unique(y_enc, return_counts=True)
    for cls_idx, cnt in zip(unique, counts):
        if cnt < CV_FOLDS:
            log.warning(
                "Class '%s' has only %d sample(s) — fewer than CV_FOLDS=%d. "
                "CV results may be unreliable.",
                class_names[cls_idx], cnt, CV_FOLDS,
            )

    # ── Train / test split ───────────────────────────────────────────────────
    # Use stratify only when every class has at least 2 samples (required)
    min_count = int(counts.min())
    stratify = y_enc if min_count >= 2 else None
    if stratify is None:
        log.warning("Disabling stratified split (some classes have only 1 sample).")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )
    log.info("  Train / Test   : %d / %d", len(X_train), len(X_test))

    # ── Build & train ────────────────────────────────────────────────────────
    log.info("\n🏗️  Building pipeline…")
    model = build_pipeline(X_train.shape[0], X_train.shape[1])

    log.info("🚀 Training…")
    model.fit(X_train, y_train)
    log.info("  ✅ Training complete!")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    log.info("\n📊 Evaluating…")
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))
    log.info("  Train accuracy : %.2f%%", train_acc * 100)
    log.info("  Test  accuracy : %.2f%%", test_acc  * 100)

    y_pred = model.predict(X_test)
    print("\n" + classification_report(y_test, y_pred, target_names=class_names))

    # ── Cross-validation ─────────────────────────────────────────────────────
    log.info("📊 Running %d-fold cross-validation…", CV_FOLDS)
    cv_scores = cross_val_score(model, X, y_enc, cv=CV_FOLDS, scoring="accuracy", n_jobs=-1)
    log.info(
        "  CV accuracy    : %.2f%% (±%.2f%%)",
        cv_scores.mean() * 100,
        cv_scores.std()  * 100,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    n_components: int = model.named_steps["pca"].n_components_
    metrics = {
        "train_accuracy":   round(float(train_acc),        4),
        "test_accuracy":    round(float(test_acc),         4),
        "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
        "cv_std":           round(float(cv_scores.std()),  4),
        "num_classes":      len(class_names),
        "classes":          class_names,
        "total_samples":    int(len(X)),
        "train_samples":    int(len(X_train)),
        "test_samples":     int(len(X_test)),
        "img_size":         IMG_SIZE,
        "n_pca_components": n_components,
        "architecture":     f"PCA({n_components}) + RandomForest({N_ESTIMATORS} trees)",
        "training_time_s":  round(float(elapsed), 2),
    }

    log.info("\n💾 Saving artifacts…")
    save_artifacts(model, le, class_names, metrics)

    print(f"\n{sep}")
    print(f"  🎉 Done!  Test accuracy: {test_acc * 100:.2f}%")
    print(f"  ⏱️  Total time: {elapsed:.1f}s")
    print(sep + "\n")


if __name__ == "__main__":
    main()
