"""
evaluate.py - Model evaluation for cognitive impairment detection.

Outputs:
  - Classification report (precision, recall, F1)
  - Confusion matrix (text + heatmap image)
  - ROC-AUC score + curve plot
  - Feature importance bar chart
  All plots saved to results/
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from model import load_model  # noqa: E402


# ─── Text helpers ─────────────────────────────────────────────────────────────

def _print_confusion_matrix(cm: np.ndarray, class_names=("Healthy", "Impaired")) -> None:
    """Print a formatted text confusion matrix."""
    print("\n[evaluate] Confusion Matrix:")
    print(f"  {'':>12}", end="")
    for name in class_names:
        print(f"  {name:>12}", end="")
    print()
    print(f"  {'-'*40}")
    for i, row_name in enumerate(class_names):
        print(f"  {row_name:>12}", end="")
        for val in cm[i]:
            print(f"  {val:>12}", end="")
        print()


# ─── Plot helpers ─────────────────────────────────────────────────────────────

def _plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str = "results/confusion_matrix.png",
    class_names=("Healthy", "Impaired"),
) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"[evaluate] Confusion matrix plot saved -> {save_path}")
    except Exception as e:
        print(f"[evaluate] Warning: Could not save confusion matrix plot: {e}")


def _plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    auc: float,
    save_path: str = "results/roc_curve.png",
) -> None:
    try:
        import matplotlib.pyplot as plt

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"[evaluate] ROC curve plot saved -> {save_path}")
    except Exception as e:
        print(f"[evaluate] Warning: Could not save ROC curve plot: {e}")


def _plot_feature_importance(
    model,
    feature_names,
    save_path: str = "results/feature_importance.png",
    top_n: int = 20,
) -> None:
    """
    Attempt to extract and plot feature importances from the ensemble.
    Uses RandomForest or GradientBoosting sub-estimator if available.
    """
    try:
        import matplotlib.pyplot as plt

        importances = None
        src_name = "unknown"
        try:
            # pipeline.named_steps['ensemble'] is the VotingClassifier
            ensemble = model.named_steps["ensemble"]
            # Use estimators_ (fitted) not estimators (unfitted config)
            fitted_estimators = getattr(ensemble, "estimators_", None)
            named_estimators  = getattr(ensemble, "named_estimators_", {})
            if fitted_estimators is not None:
                for est_name, est in named_estimators.items():
                    if hasattr(est, "feature_importances_"):
                        importances = est.feature_importances_
                        src_name = est_name
                        break
            if importances is None and fitted_estimators:
                for est in fitted_estimators:
                    if hasattr(est, "feature_importances_"):
                        importances = est.feature_importances_
                        src_name = type(est).__name__
                        break
        except Exception:
            pass

        if importances is None:
            print("[evaluate] Warning: No feature importances available from ensemble.")
            return

        indices = np.argsort(importances)[::-1][:top_n]
        top_importances = importances[indices]
        top_names = [feature_names[i] for i in indices]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(range(top_n), top_importances[::-1], color="steelblue")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names[::-1], fontsize=8)
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Top {top_n} Feature Importances (from {src_name})")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"[evaluate] Feature importance plot saved -> {save_path}")
    except Exception as e:
        print(f"[evaluate] Warning: Could not save feature importance plot: {e}")


# ─── Main evaluation function ─────────────────────────────────────────────────

def evaluate(
    model_path: str = "models/cognitive_impairment_detector.pkl",
    test_csv_path: str = "data/test_data.csv",
    results_dir: str = "results",
) -> dict:
    """
    Load model + test data, compute and display all evaluation metrics.

    Args:
        model_path: Path to saved .pkl model.
        test_csv_path: Path to test split CSV.
        results_dir: Directory for saving plots.

    Returns:
        Dict with keys: accuracy, auc, report_dict, confusion_matrix
    """
    print(f"\n{'='*60}")
    print("  EVALUATION PIPELINE")
    print(f"{'='*60}")

    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(model_path)

    # ── Load test data ────────────────────────────────────────────────────────
    if not Path(test_csv_path).exists():
        raise FileNotFoundError(
            f"Test data not found: {test_csv_path}\n"
            "Run train.py first."
        )
    test_df = pd.read_csv(test_csv_path)
    y_test = test_df["label"].values.astype(int)
    X_test = test_df.drop(columns=["label"]).values.astype(np.float32)
    feature_names = [c for c in test_df.columns if c != "label"]
    print(f"[evaluate] Loaded test set: {len(y_test)} samples")

    # ── Predictions ───────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ── Accuracy ──────────────────────────────────────────────────────────────
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[evaluate] *** Test Accuracy: {accuracy * 100:.2f}% ***")

    # ── Classification report ─────────────────────────────────────────────────
    target_names = ["Healthy (0)", "Impaired (1)"]
    report_str = classification_report(y_test, y_pred, target_names=target_names)
    report_dict = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True
    )
    print(f"\n[evaluate] Classification Report:\n{report_str}")

    # ── Confusion matrix (text) ────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    _print_confusion_matrix(cm)

    # ── ROC-AUC ───────────────────────────────────────────────────────────────
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n[evaluate] ROC-AUC Score: {auc:.4f}")

    # ── Save plots ────────────────────────────────────────────────────────────
    _plot_confusion_matrix(cm, save_path=f"{results_dir}/confusion_matrix.png")
    _plot_roc_curve(y_test, y_prob, auc, save_path=f"{results_dir}/roc_curve.png")
    _plot_feature_importance(
        model,
        feature_names,
        save_path=f"{results_dir}/feature_importance.png",
    )

    print(f"\n[evaluate] {'='*58}")
    print(f"[evaluate]  Evaluation DONE  |  Accuracy: {accuracy*100:.2f}%  |  AUC: {auc:.4f}")
    print(f"[evaluate] {'='*58}\n")

    return {
        "accuracy":         accuracy,
        "auc":              auc,
        "report_dict":      report_dict,
        "confusion_matrix": cm,
        "y_pred":           y_pred,
        "y_prob":           y_prob,
        "y_test":           y_test,
    }


if __name__ == "__main__":
    evaluate()
