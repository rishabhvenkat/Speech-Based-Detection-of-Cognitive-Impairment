"""
predict.py - Inference module for cognitive impairment detection.

Usage:
  Single file  : predict_single("path/to/audio.wav")
  Batch folder : batch_predict("path/to/audio_folder/")
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from model import load_model                      # noqa: E402
from feature_extractor import (                    # noqa: E402
    extract_features,
    features_to_array,
    FEATURE_NAMES,
)

CLASS_LABELS = {0: "Healthy", 1: "Cognitively Impaired"}


def predict_single(
    audio_path: str,
    model_path: str = "models/cognitive_impairment_detector.pkl",
    verbose: bool = True,
) -> Optional[dict]:
    """
    Predict cognitive status from a single .wav file.

    Args:
        audio_path: Path to the .wav audio file.
        model_path: Path to the saved model .pkl file.
        verbose: If True, print the prediction result.

    Returns:
        Dict with keys: file, prediction, label, confidence, features
        or None if feature extraction failed.
    """
    model = load_model(model_path)

    features = extract_features(audio_path)
    if features is None:
        print(f"[predict] ERROR: Could not extract features from {audio_path}")
        return None

    X = features_to_array(features).reshape(1, -1)
    label_idx = int(model.predict(X)[0])
    proba     = model.predict_proba(X)[0]
    confidence = float(proba[label_idx]) * 100.0

    result = {
        "file":       os.path.basename(audio_path),
        "prediction": CLASS_LABELS[label_idx],
        "label":      label_idx,
        "confidence": confidence,
        "features":   features,
    }

    if verbose:
        print(f"  File       : {result['file']}")
        print(f"  Prediction : {result['prediction']}")
        print(f"  Confidence : {confidence:.1f}%")
        print(f"  Healthy prob   : {proba[0]*100:.1f}%  |  Impaired prob: {proba[1]*100:.1f}%")

    return result


def batch_predict(
    audio_folder: str,
    model_path: str = "models/cognitive_impairment_detector.pkl",
    extensions: tuple = (".wav", ".flac", ".mp3", ".ogg"),
) -> pd.DataFrame:
    """
    Run prediction on all audio files in a folder.

    Args:
        audio_folder: Directory containing audio files.
        model_path: Path to the saved model .pkl file.
        extensions: File extensions to include.

    Returns:
        DataFrame with columns: file, prediction, label, confidence
        Returns empty DataFrame if no valid files found.
    """
    folder = Path(audio_folder)
    if not folder.exists():
        print(f"[predict] ERROR: Folder not found: {audio_folder}")
        return pd.DataFrame()

    audio_files = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in extensions
    )

    if not audio_files:
        print(f"[predict] No audio files found in {audio_folder}")
        return pd.DataFrame()

    print(f"\n[predict] Batch prediction on {len(audio_files)} files in {audio_folder}")
    print(f"  {'File':<30} {'Prediction':<22} {'Confidence':>12}")
    print(f"  {'-'*66}")

    model = load_model(model_path)
    rows = []

    for path in audio_files:
        features = extract_features(str(path))
        if features is None:
            print(f"  {path.name:<30} {'[EXTRACTION FAILED]':<22} {'N/A':>12}")
            continue

        X = features_to_array(features).reshape(1, -1)
        label_idx  = int(model.predict(X)[0])
        proba      = model.predict_proba(X)[0]
        confidence = float(proba[label_idx]) * 100.0

        rows.append({
            "file":       path.name,
            "prediction": CLASS_LABELS[label_idx],
            "label":      label_idx,
            "confidence": round(confidence, 1),
        })
        print(f"  {path.name:<30} {CLASS_LABELS[label_idx]:<22} {confidence:>10.1f}%")

    if not rows:
        return pd.DataFrame(columns=["file", "prediction", "label", "confidence"])

    df = pd.DataFrame(rows)
    healthy_count  = (df["label"] == 0).sum()
    impaired_count = (df["label"] == 1).sum()
    print(f"\n[predict] Results: {healthy_count} Healthy | {impaired_count} Cognitively Impaired")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cognitive Impairment Prediction from Speech Audio"
    )
    parser.add_argument("input", help="Path to .wav file or folder")
    parser.add_argument(
        "--model",
        default="models/cognitive_impairment_detector.pkl",
        help="Path to model .pkl file",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        results = batch_predict(str(input_path), model_path=args.model)
        if not results.empty:
            print(f"\n{results.to_string(index=False)}")
    elif input_path.is_file():
        result = predict_single(str(input_path), model_path=args.model)
    else:
        print(f"ERROR: Input path does not exist: {args.input}")
        sys.exit(1)
