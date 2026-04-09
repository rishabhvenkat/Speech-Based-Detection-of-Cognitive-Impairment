"""
utils.py - Helper utilities for the Cognitive Impairment Detection system.

Provides directory setup, visualization, and display helpers.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional


def setup_directories() -> None:
    """Create required project directories if they don't exist."""
    dirs = [
        "data",
        "data/sample_audio",
        "models",
        "results",
        "src",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("[setup] Directories ready: data/, data/sample_audio/, models/, results/")


def print_banner() -> None:
    """Print ASCII art banner for the project."""
    banner = """
+======================================================================+
|                                                                      |
|     ____  ____  ____  ____   ____  ____                             |
|    / ___||  _ \\| __ )|  _ \\ |  _ \\|  _ \\                            |
|    \\___ \\| |_) |  _ \\| | | || |_) | |_) |                           |
|     ___) |  __/| |_) | |_| ||  _ <|  _ <                            |
|    |____/|_|   |____/|____/ |_| \\_\\_| \\_\\                           |
|                                                                      |
|       Speech-Based Cognitive Impairment Detection System            |
|             Alzheimer's & MCI Detection via Audio ML                |
|                                                                      |
+======================================================================+
    """
    print(banner)


def plot_feature_distributions(
    df: pd.DataFrame,
    features: List[str],
    save_path: str = "results/feature_distributions.png",
    max_features: int = 12,
) -> None:
    """
    Plot healthy vs impaired feature distributions side by side.

    Args:
        df: DataFrame with features and 'label' column (0=healthy, 1=impaired).
        features: List of feature column names to plot.
        save_path: Output path for the saved figure.
        max_features: Maximum number of features to plot.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        features_to_plot = features[:max_features]
        n = len(features_to_plot)
        cols = 4
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes = axes.flatten() if n > 1 else [axes]

        healthy = df[df["label"] == 0]
        impaired = df[df["label"] == 1]

        for i, feat in enumerate(features_to_plot):
            ax = axes[i]
            ax.hist(
                healthy[feat].dropna(),
                bins=30,
                alpha=0.6,
                color="steelblue",
                label="Healthy",
                density=True,
            )
            ax.hist(
                impaired[feat].dropna(),
                bins=30,
                alpha=0.6,
                color="salmon",
                label="Impaired",
                density=True,
            )
            ax.set_title(feat, fontsize=9)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)

        # Hide unused axes
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Feature Distributions: Healthy vs Cognitively Impaired", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
        plt.close()
        print(f"[utils] Feature distribution plot saved -> {save_path}")

    except Exception as e:
        print(f"[utils] Warning: Could not plot feature distributions: {e}")


def print_summary_table(
    accuracy: float,
    auc: float,
    cv_mean: float,
    cv_std: float,
    f1_healthy: float,
    f1_impaired: float,
) -> None:
    """
    Print a formatted final summary table.

    Args:
        accuracy: Test set accuracy (0-1).
        auc: ROC-AUC score (0-1).
        cv_mean: Mean cross-validation accuracy (0-1).
        cv_std: Std of cross-validation accuracy (0-1).
        f1_healthy: F1 score for healthy class.
        f1_impaired: F1 score for impaired class.
    """
    line = "-" * 52
    print(f"\n{'='*52}")
    print(f"  FINAL PERFORMANCE SUMMARY")
    print(f"{'='*52}")
    print(f"  {'Metric':<30} {'Value':>16}")
    print(f"  {line}")
    print(f"  {'Test Accuracy':<30} {accuracy*100:>14.2f}%")
    print(f"  {'ROC-AUC Score':<30} {auc:>16.4f}")
    print(f"  {'CV Mean Accuracy (5-fold)':<30} {cv_mean*100:>14.2f}%")
    print(f"  {'CV Std':<30} {cv_std*100:>14.2f}%")
    print(f"  {'F1 Score  - Healthy (class 0)':<30} {f1_healthy:>16.4f}")
    print(f"  {'F1 Score  - Impaired (class 1)':<30} {f1_impaired:>16.4f}")
    print(f"{'='*52}\n")
