"""
VERITAS Evaluator
Day 4: Runs pipeline against FEVER dev set and computes metrics.
Produces Precision, Recall, F1 per class and Macro-F1 overall.
"""

import sys
sys.path.insert(0, ".")

import pandas as pd
import json
import os
import time
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix


# Map FEVER labels to VERITAS verdicts
FEVER_TO_VERITAS = {
    "SUPPORTS": "SUPPORTED",
    "REFUTES": "REFUTED",
    "NOT ENOUGH INFO": "INSUFFICIENT_EVIDENCE"
}

# Map VERITAS verdicts back to 3-class for evaluation
VERITAS_TO_3CLASS = {
    "SUPPORTED": "SUPPORTS",
    "REFUTED": "REFUTES",
    "PARTIALLY_SUPPORTED": "SUPPORTS",  # conservative mapping
    "CONFLICTING": "REFUTES",           # conservative mapping
    "INSUFFICIENT_EVIDENCE": "NOT ENOUGH INFO",
    "NOT_FACTUAL": "NOT ENOUGH INFO"
}

EVAL_PATH = "./logs/eval_results"


class Evaluator:

    def __init__(self, pipeline):
        self.pipeline = pipeline
        os.makedirs(EVAL_PATH, exist_ok=True)
        print("[Evaluator] Ready.")

    def evaluate_fever(
        self,
        fever_path: str = "./data/fever/dev.csv",
        n_samples: int = 100,
        save_results: bool = True
    ) -> dict:
        """
        Run pipeline on FEVER dev set and compute metrics.
        n_samples: how many claims to evaluate (full dev = 19K, slow)
        """
        print(f"\n[Evaluator] Loading FEVER dev set from {fever_path}...")
        df = pd.read_csv(fever_path)

        # Deduplicate — FEVER has multiple evidence rows per claim
        df_claims = df.drop_duplicates(subset=["claim"]).copy()
        df_claims = df_claims[df_claims["label"].isin(["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"])]

        # Sample evenly across classes for balanced evaluation
        samples_per_class = n_samples // 3
        sampled = pd.concat([
            df_claims[df_claims["label"] == "SUPPORTS"].sample(
                min(samples_per_class, len(df_claims[df_claims["label"] == "SUPPORTS"])),
                random_state=42
            ),
            df_claims[df_claims["label"] == "REFUTES"].sample(
                min(samples_per_class, len(df_claims[df_claims["label"] == "REFUTES"])),
                random_state=42
            ),
            df_claims[df_claims["label"] == "NOT ENOUGH INFO"].sample(
                min(samples_per_class, len(df_claims[df_claims["label"] == "NOT ENOUGH INFO"])),
                random_state=42
            ),
        ]).reset_index(drop=True)

        print(f"[Evaluator] Evaluating {len(sampled)} claims...")
        print(f"  Class distribution: {sampled['label'].value_counts().to_dict()}")

        y_true = []
        y_pred = []
        results = []
        errors = 0

        for i, row in sampled.iterrows():
            claim = row["claim"]
            true_label = row["label"]

            try:
                result = self.pipeline.run(claim)
                pred_veritas = result["verdict"]
                pred_3class = VERITAS_TO_3CLASS.get(pred_veritas, "NOT ENOUGH INFO")

                y_true.append(true_label)
                y_pred.append(pred_3class)

                results.append({
                    "claim": claim,
                    "true_label": true_label,
                    "predicted": pred_3class,
                    "veritas_verdict": pred_veritas,
                    "confidence": result["confidence"],
                    "latency": result["latency"],
                    "correct": true_label == pred_3class
                })

                if (i + 1) % 10 == 0:
                    correct_so_far = sum(r["correct"] for r in results)
                    print(f"  [{i+1}/{len(sampled)}] Running accuracy: {correct_so_far/(i+1):.3f}")

            except Exception as e:
                errors += 1
                y_true.append(true_label)
                y_pred.append("NOT ENOUGH INFO")
                results.append({
                    "claim": claim,
                    "true_label": true_label,
                    "predicted": "NOT ENOUGH INFO",
                    "veritas_verdict": "ERROR",
                    "confidence": 0.0,
                    "latency": 0.0,
                    "correct": False,
                    "error": str(e)
                })

        # Compute metrics
        labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        report = classification_report(
            y_true, y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0
        )

        macro_f1 = report["macro avg"]["f1-score"]
        accuracy = report["accuracy"]

        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "n_samples": len(sampled),
            "errors": errors,
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "per_class": {
                label: {
                    "precision": round(report[label]["precision"], 4),
                    "recall": round(report[label]["recall"], 4),
                    "f1": round(report[label]["f1-score"], 4),
                }
                for label in labels
            },
            "avg_latency": round(
                sum(r["latency"] for r in results) / len(results), 2
            )
        }

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"  Accuracy:  {metrics['accuracy']}")
        print(f"  Macro-F1:  {metrics['macro_f1']}")
        print(f"  Avg latency: {metrics['avg_latency']}s")
        print("\n  Per-class:")
        for label, scores in metrics["per_class"].items():
            print(f"    {label:20s} P={scores['precision']} R={scores['recall']} F1={scores['f1']}")

        if save_results:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            metrics_path = f"{EVAL_PATH}/metrics_{ts}.json"
            results_path = f"{EVAL_PATH}/predictions_{ts}.json"

            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

            print(f"\n  Saved to {metrics_path}")

        return metrics


if __name__ == "__main__":
    from src.pipeline import VERITASPipeline

    pipeline = VERITASPipeline()
    evaluator = Evaluator(pipeline)

    # Start small — 30 claims to verify the harness works
    metrics = evaluator.evaluate_fever(n_samples=30)