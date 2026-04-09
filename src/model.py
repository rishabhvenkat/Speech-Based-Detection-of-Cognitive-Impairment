"""
model.py - Ensemble classifier for cognitive impairment detection.

Architecture:
  Pipeline: StandardScaler -> VotingClassifier (soft voting)
    Base classifiers:
      1. RandomForestClassifier       (300 trees)
      2. GradientBoostingClassifier
      3. SVC                          (RBF kernel, probability=True)
      4. XGBClassifier                (fallback: ExtraTreesClassifier)

Model is saved/loaded via joblib.
"""

import os
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


def _build_xgboost() -> Optional[object]:
    """Try to import and return an XGBClassifier; return None if unavailable."""
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    except ImportError:
        return None


def build_model() -> Pipeline:
    """
    Build and return the full sklearn Pipeline:
      StandardScaler -> VotingClassifier (soft voting).

    Returns:
        Unfitted sklearn Pipeline.
    """
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        random_state=42,
    )

    svm = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42,
    )

    xgb = _build_xgboost()
    if xgb is not None:
        fourth_name = "xgboost"
        fourth_clf  = xgb
        print("[model] XGBoost available -> included in ensemble.")
    else:
        fourth_clf = ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        fourth_name = "extratrees"
        print("[model] XGBoost not available -> using ExtraTreesClassifier as fallback.")

    ensemble = VotingClassifier(
        estimators=[
            ("random_forest",       rf),
            ("gradient_boosting",   gb),
            ("svm",                 svm),
            (fourth_name,           fourth_clf),
        ],
        voting="soft",
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler",   StandardScaler()),
            ("ensemble", ensemble),
        ]
    )

    return pipeline


def save_model(model: Pipeline, path: str = "models/cognitive_impairment_detector.pkl") -> None:
    """
    Persist the trained pipeline to disk using joblib.

    Args:
        model: Fitted sklearn Pipeline.
        path: Output file path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"[model] Model saved -> {path}  ({size_mb:.1f} MB)")


def load_model(path: str = "models/cognitive_impairment_detector.pkl") -> Pipeline:
    """
    Load a previously saved pipeline from disk.

    Args:
        path: Path to the .pkl model file.

    Returns:
        Loaded sklearn Pipeline.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Run train.py first to train and save the model."
        )
    model = joblib.load(path)
    print(f"[model] Model loaded <- {path}")
    return model


if __name__ == "__main__":
    print("[model] Building model architecture (dry run)...")
    pipeline = build_model()
    print(f"[model] Pipeline steps: {[name for name, _ in pipeline.steps]}")
    print("[model] Done.")
