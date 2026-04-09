"""
main.py - Full pipeline runner for Cognitive Impairment Detection from Speech.

Steps:
  1. Setup directories
  2. Generate synthetic dataset
  3. Train ensemble model
  4. Evaluate model
  5. Batch predict on sample audio
  6. Print final summary table
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Ensure src/ is on the path
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

_SEP = "-" * 60


def main() -> None:
    # Imports
    from src.utils import setup_directories, print_banner, print_summary_table
    from src.data_generator import generate_dataset
    from src.train import train
    from src.evaluate import evaluate
    from src.predict import batch_predict

    # Banner
    print_banner()

    # Step 1: Setup directories
    print("\n" + _SEP)
    print("  STEP 1: Setup Directories")
    print(_SEP)
    setup_directories()

    # Step 2: Generate dataset
    print("\n" + _SEP)
    print("  STEP 2: Generate Synthetic Dataset")
    print(_SEP)
    df = generate_dataset(
        csv_path="data/speech_features.csv",
        audio_dir="data/sample_audio",
        n_healthy=1000,
        n_impaired=1000,
        random_state=42,
    )

    # Step 3: Train model
    print("\n" + _SEP)
    print("  STEP 3: Train Ensemble Model")
    print(_SEP)
    train_result = train(
        csv_path="data/speech_features.csv",
        model_path="models/cognitive_impairment_detector.pkl",
        test_csv_path="data/test_data.csv",
        test_size=0.2,
        random_state=42,
        cv_folds=5,
    )

    # Step 4: Evaluate model
    print("\n" + _SEP)
    print("  STEP 4: Evaluate Model")
    print(_SEP)
    eval_result = evaluate(
        model_path="models/cognitive_impairment_detector.pkl",
        test_csv_path="data/test_data.csv",
        results_dir="results",
    )

    # Step 5: Batch prediction on sample audio
    print("\n" + _SEP)
    print("  STEP 5: Batch Prediction on Sample Audio")
    print(_SEP)
    predictions = batch_predict(
        audio_folder="data/sample_audio",
        model_path="models/cognitive_impairment_detector.pkl",
    )

    # Step 6: Final summary table
    print("\n" + _SEP)
    print("  STEP 6: Final Summary")
    print(_SEP)

    report = eval_result["report_dict"]
    f1_healthy  = report.get("Healthy (0)",  {}).get("f1-score", 0.0)
    f1_impaired = report.get("Impaired (1)", {}).get("f1-score", 0.0)

    print_summary_table(
        accuracy    = eval_result["accuracy"],
        auc         = eval_result["auc"],
        cv_mean     = train_result["cv_mean"],
        cv_std      = train_result["cv_std"],
        f1_healthy  = f1_healthy,
        f1_impaired = f1_impaired,
    )

    # Status line
    acc_pct = eval_result["accuracy"] * 100
    target  = 90.0
    status  = "PASS" if acc_pct >= target else "FAIL"
    print(f"  Achieved accuracy: {acc_pct:.2f}%  (realistic target ~93-95%)  [{status}]\n")

    # Feature distribution plot
    try:
        import pandas as pd
        from src.utils import plot_feature_distributions

        data_df = pd.read_csv("data/speech_features.csv")
        feat_cols = [
            c for c in data_df.columns
            if c not in ("subject_id", "label")
        ]
        plot_feature_distributions(
            data_df,
            feat_cols,
            save_path="results/feature_distributions.png",
        )
    except Exception as e:
        print(f"[main] Warning: Feature distribution plot failed: {e}")

    print("\n[main] Pipeline complete. Results saved to results/")
    print("[main] Saved artifacts:")
    print("         models/cognitive_impairment_detector.pkl")
    print("         data/speech_features.csv")
    print("         data/test_data.csv")
    print("         results/confusion_matrix.png")
    print("         results/roc_curve.png")
    print("         results/feature_importance.png")
    print("         results/feature_distributions.png")


if __name__ == "__main__":
    main()
