"""
VERITAS Kaggle Baseline Evaluator
==================================
Self-contained evaluation script for Kaggle T4 GPU environment.
Uses cross-encoder/nli-deberta-v3-small for NLI verdicts (no Ollama).
Builds ChromaDB index from FEVER train.csv.
Evaluates 300 balanced claims (100 per class) from FEVER dev.csv.
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "veritas_fever_baseline"
FEVER_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

VERITAS_TO_3CLASS = {
    "SUPPORTED": "SUPPORTS",
    "REFUTED": "REFUTES",
    "INSUFFICIENT_EVIDENCE": "NOT ENOUGH INFO",
}


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def build_index(fever_train_path: str, chroma_path: str, max_passages: int = 50000):
    import chromadb
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path=chroma_path)

    try:
        col = client.get_collection(COLLECTION_NAME)
        count = col.count()
        if count > 0:
            print(f"[Index] Existing index with {count:,} passages. Skipping rebuild.")
            return col
    except Exception:
        pass

    print("[Index] Building ChromaDB index from train.csv...")
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    col = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    df = pd.read_csv(fever_train_path)
    df = df.dropna(subset=["evidence_wiki_url"])
    df = df[df["label"] != "NOT ENOUGH INFO"]

    passages, seen = [], set()
    for _, row in df.iterrows():
        text = f"{row['claim']} [Source: {row['evidence_wiki_url']}]"
        if text not in seen:
            seen.add(text)
            passages.append(text)
        if len(passages) >= max_passages:
            break

    print(f"[Index] {len(passages)} unique passages extracted.")

    device = "cuda" if _cuda_available() else "cpu"
    embedder = SentenceTransformer(EMBED_MODEL, device=device)

    BATCH = 500
    for i in range(0, len(passages), BATCH):
        batch = passages[i:i+BATCH]
        embeddings = embedder.encode(batch, batch_size=64, show_progress_bar=False).tolist()
        col.add(
            documents=batch,
            embeddings=embeddings,
            ids=[f"p_{i+j}" for j in range(len(batch))],
            metadatas=[{"source": "fever"} for _ in batch]
        )
        if (i // BATCH) % 10 == 0:
            print(f"  Indexed {i+len(batch)}/{len(passages)}")

    del embedder
    print(f"[Index] Done. {col.count():,} passages indexed.")
    return col


class KaggleRetriever:
    def __init__(self, collection, device="cpu"):
        from sentence_transformers import SentenceTransformer
        self.collection = collection
        self.embedder = SentenceTransformer(EMBED_MODEL, device=device)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if not query.strip():
            return []
        embedding = self.embedder.encode(query).tolist()
        results = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        return [
            {"text": results["documents"][0][i], "score": 1.0 - results["distances"][0][i]}
            for i in range(len(results["documents"][0]))
        ]


class NLIVerifier:
    def __init__(self, device="cpu"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        print(f"[NLIVerifier] Loading {NLI_MODEL} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        self.model.eval()
        self.device = device
        self.model.to(device)
        self.torch = torch
        print("[NLIVerifier] Ready.")

    def verify(self, claim: str, passages: list[dict]) -> dict:
        if not passages:
            return {"verdict": "INSUFFICIENT_EVIDENCE", "confidence": 0.0,
                    "entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}

        import torch
        entailment_scores, contradiction_scores, neutral_scores = [], [], []

        for p in passages:
            enc = self.tokenizer(
                claim, p["text"],
                max_length=512, truncation=True, padding=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**enc).logits[0]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            contradiction_scores.append(float(probs[0]))
            entailment_scores.append(float(probs[1]))
            neutral_scores.append(float(probs[2]))

        max_entailment = max(entailment_scores)
        max_contradiction = max(contradiction_scores)
        mean_neutral = float(np.mean(neutral_scores))

        if max_entailment > 0.5 and max_entailment > max_contradiction:
            verdict, confidence = "SUPPORTED", max_entailment
        elif max_contradiction > 0.5 and max_contradiction > max_entailment:
            verdict, confidence = "REFUTED", max_contradiction
        else:
            verdict, confidence = "INSUFFICIENT_EVIDENCE", mean_neutral

        return {
            "verdict": verdict,
            "confidence": round(float(confidence), 4),
            "entailment": round(max_entailment, 4),
            "contradiction": round(max_contradiction, 4),
            "neutral": round(mean_neutral, 4),
        }


def load_fever_sample(fever_dev_path: str, n_per_class: int = 100, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(fever_dev_path)
    df = df.drop_duplicates(subset=["claim"]).copy()
    df = df[df["label"].isin(FEVER_LABELS)]

    sampled = pd.concat([
        df[df["label"] == label].sample(
            min(n_per_class, len(df[df["label"] == label])), random_state=seed
        )
        for label in FEVER_LABELS
    ]).reset_index(drop=True)

    print(f"[Eval] Sampled {len(sampled)} claims: {sampled['label'].value_counts().to_dict()}")
    return sampled


def evaluate(fever_train_path, fever_dev_path, n_samples=300,
             chroma_path="./chroma_index", output_path="./baseline_results.json", top_k=5):
    from sklearn.metrics import classification_report

    device = "cuda" if _cuda_available() else "cpu"
    print(f"[Eval] Device: {device}")

    collection = build_index(fever_train_path, chroma_path)
    retriever = KaggleRetriever(collection, device=device)
    verifier = NLIVerifier(device=device)

    n_per_class = n_samples // 3
    df = load_fever_sample(fever_dev_path, n_per_class=n_per_class)

    y_true, y_pred, records, errors = [], [], [], 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        claim, true_label = row["claim"], row["label"]
        t0 = time.time()
        try:
            passages = retriever.retrieve(claim, top_k=top_k)
            result = verifier.verify(claim, passages)
            pred = VERITAS_TO_3CLASS.get(result["verdict"], "NOT ENOUGH INFO")
            y_true.append(true_label)
            y_pred.append(pred)
            records.append({
                "claim": claim, "true_label": true_label, "predicted": pred,
                "confidence": result["confidence"], "latency": round(time.time() - t0, 3),
                "correct": true_label == pred
            })
        except Exception as e:
            errors += 1
            y_true.append(true_label)
            y_pred.append("NOT ENOUGH INFO")
            records.append({"claim": claim, "true_label": true_label,
                            "predicted": "NOT ENOUGH INFO", "correct": False, "error": str(e)})

    report = classification_report(y_true, y_pred, labels=FEVER_LABELS,
                                   output_dict=True, zero_division=0)
    accuracy = sum(r["correct"] for r in records) / len(records)
    macro_f1 = report["macro avg"]["f1-score"]

    metrics = {
        "model": NLI_MODEL, "n_samples": len(df), "errors": errors,
        "accuracy": round(accuracy, 4), "macro_f1": round(macro_f1, 4),
        "per_class": {
            label: {
                "precision": round(report[label]["precision"], 4),
                "recall": round(report[label]["recall"], 4),
                "f1": round(report[label]["f1-score"], 4),
            }
            for label in FEVER_LABELS
        },
        "avg_latency_s": round(sum(r["latency"] for r in records) / len(records), 3),
    }

    print("\n" + "=" * 60)
    print("BASELINE EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy:  {metrics['accuracy']}")
    print(f"  Macro-F1:  {metrics['macro_f1']}")
    for label, s in metrics["per_class"].items():
        print(f"    {label:20s} P={s['precision']}  R={s['recall']}  F1={s['f1']}")
    print("=" * 60)

    with open(output_path, "w") as f:
        json.dump({"metrics": metrics, "predictions": records}, f, indent=2)
    print(f"\n[Eval] Saved to {output_path}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fever_train_path", default="./data/fever/train.csv")
    parser.add_argument("--fever_dev_path", default="./data/fever/dev.csv")
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--chroma_path", default="/kaggle/working/chroma_index")
    parser.add_argument("--output_path", default="/kaggle/working/baseline_results.json")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    evaluate(
        fever_train_path=args.fever_train_path,
        fever_dev_path=args.fever_dev_path,
        n_samples=args.n_samples,
        chroma_path=args.chroma_path,
        output_path=args.output_path,
        top_k=args.top_k,
    )