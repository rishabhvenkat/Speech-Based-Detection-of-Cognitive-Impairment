"""
train.py - Train the cognitive impairment detection ensemble model.

Pipeline:
  1. Load data/speech_features.csv
  2. 80/20 stratified train/test split
  3. Apply SMOTE (or manual oversampling) if classes are imbalanced
  4. Train ensemble pipeline
  5. 5-fold stratified cross-validation
  6. Save model -> models/cognitive_impairment_detector.pkl
  7. Save test split -> data/test_data.csv
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# Allow running as standalone script or imported from parent dir
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from model import build_model, save_model  # noqa: E402


# ─── SMOTE / oversampling ─────────────────────────────────────────────────────

def _apply_smote(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Apply SMOTE if imbalanced-learn is available, otherwise manual oversampling.

    Args:
        X: Feature matrix.
        y: Label vector.

    Returns:
        (X_resampled, y_resampled)
    """
    counts = np.bincount(y)
    if len(counts) < 2 or counts[0] == counts[1]:
        return X, y  # Already balanced

    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        print(f"[train] SMOTE applied: {dict(zip(*np.unique(y_res, return_counts=True)))}")
        return X_res, y_res
    except ImportError:
        print("[train] imbalanced-learn not available -> using manual oversampling.")

    # Manual oversampling: duplicate minority class with tiny jitter
    minority = int(np.argmin(counts))
    majority_count = counts.max()
    minority_idx = np.where(y == minority)[0]
    rng = np.random.default_rng(42)
    n_needed = majority_count - counts[minority]
    chosen = rng.choice(minority_idx, size=n_needed, replace=True)
    noise = rng.normal(0, 1e-4, (n_needed, X.shape[1]))
    X_res = np.vstack([X, X[chosen] + noise])
    y_res = np.concatenate([y, np.full(n_needed, minority)])
    print(f"[train] Manual oversample: {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res


# ─── Main training function ───────────────────────────────────────────────────

def train(
    csv_path: str = "data/speech_features.csv",
    model_path: str = "models/cognitive_impairment_detector.pkl",
    test_csv_path: str = "data/test_data.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
) -> dict:
    """
    Full training pipeline.

    Args:
        csv_path: Path to features CSV.
        model_path: Where to save the trained model.
        test_csv_path: Where to save test split.
        test_size: Fraction of data for testing.
        random_state: Random seed.
        cv_folds: Number of cross-validation folds.

    Returns:
        Dict with keys: model, cv_scores, cv_mean, cv_std, train_accuracy, test_accuracy
    """
    print(f"\n{'='*60}")
    print("  TRAINING PIPELINE")
    print(f"{'='*60}")

    # ── 1. Load data ─────────────────────────────────────────────────────────
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Run data_generator.py first."
        )
    df = pd.read_csv(csv_path)
    print(f"[train] Loaded {len(df)} samples from {csv_path}")
    print(f"[train] Class distribution: {dict(df['label'].value_counts().sort_index())}")

    # ── 2. Prepare X, y ──────────────────────────────────────────────────────
    drop_cols = ["subject_id", "label"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(int)

    print(f"[train] Feature matrix shape: {X.shape}")

    # ── 3. Train/test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    print(f"[train] Train set: {len(X_train)} samples  |  Test set: {len(X_test)} samples")

    # ── 4. Save test split ────────────────────────────────────────────────────
    test_df = pd.DataFrame(X_test, columns=feature_cols)
    test_df.insert(0, "label", y_test)
    Path(test_csv_path).parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(test_csv_path, index=False)
    print(f"[train] Test data saved -> {test_csv_path}")

    # ── 5. Apply SMOTE / balancing ────────────────────────────────────────────
    X_train_bal, y_train_bal = _apply_smote(X_train, y_train)

    # ── 6. Build & train model ────────────────────────────────────────────────
    print("\n[train] Building ensemble model...")
    model = build_model()

    print("[train] Fitting model on training data...")
    model.fit(X_train_bal, y_train_bal)
    print("[train] Training complete.")

    # ── 7. Training accuracy ──────────────────────────────────────────────────
    train_preds = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    print(f"[train] Train accuracy (unbalanced original): {train_acc * 100:.2f}%")

    # ── 8. Test accuracy ───────────────────────────────────────────────────────
    test_preds = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    print(f"[train] Test accuracy                       : {test_acc * 100:.2f}%")

    # ── 9. 5-fold cross-validation ────────────────────────────────────────────
    print(f"\n[train] Running {cv_folds}-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

    print(f"\n[train] Cross-Validation Results ({cv_folds} folds):")
    print(f"  {'Fold':<8} {'Accuracy':>10}")
    print(f"  {'-'*20}")
    for fold_idx, score in enumerate(cv_scores, 1):
        print(f"  {fold_idx:<8} {score * 100:>9.2f}%")
    print(f"  {'-'*20}")
    print(f"  {'Mean':<8} {cv_scores.mean() * 100:>9.2f}%")
    print(f"  {'Std':<8} {cv_scores.std() * 100:>9.2f}%")

    # ── 10. Save model ────────────────────────────────────────────────────────
    save_model(model, model_path)

    result = {
        "model":          model,
        "cv_scores":      cv_scores,
        "cv_mean":        cv_scores.mean(),
        "cv_std":         cv_scores.std(),
        "train_accuracy": train_acc,
        "test_accuracy":  test_acc,
        "feature_cols":   feature_cols,
        "X_test":         X_test,
        "y_test":         y_test,
    }

    print(f"\n[train] {'='*58}")
    print(f"[train]  Training DONE  |  Test Accuracy: {test_acc * 100:.2f}%")
    print(f"[train] {'='*58}\n")

    return result


if __name__ == "__main__":
    train()
