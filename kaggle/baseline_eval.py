"""
VERITAS Kaggle Baseline Evaluator
==================================
Self-contained evaluation script for Kaggle T4 GPU environment.

Differences from local pipeline:
- No Ollama: uses cross-encoder/nli-deberta-v3-small for NLI verdicts
- Rebuilds ChromaDB index from FEVER wiki-pages on first run
- Evaluates 300 balanced claims (100 per class) from FEVER dev set
- Outputs Macro-F1, Precision, Recall per class

Install deps (Kaggle notebook cell):
    !pip install chromadb sentence-transformers scikit-learn pandas tqdm

Usage:
    python kaggle/baseline_eval.py \
        --fever_dir /kaggle/input/fever \
        --n_samples 300 \
        --chroma_path /kaggle/working/chroma_index \
        --output_path /kaggle/working/baseline_results.json
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "veritas_fever_baseline"

# NLI label mapping: cross-encoder outputs [contradiction, entailment, neutral]
NLI_IDX = {"contradiction": 0, "entailment": 1, "neutral": 2}

# FEVER label → 3-class canonical
FEVER_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

VERITAS_TO_3CLASS = {
    "SUPPORTED": "SUPPORTS",
    "REFUTED": "REFUTES",
    "INSUFFICIENT_EVIDENCE": "NOT ENOUGH INFO",
}


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index(fever_dir: str, chroma_path: str, max_passages: int = 500_000):
    """
    Build ChromaDB index from FEVER wiki-pages corpus.
    Skips if index already exists and has documents.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path=chroma_path)

    # Check if already built
    try:
        col = client.get_collection(COLLECTION_NAME)
        count = col.count()
        if count > 0:
            print(f"[Index] Existing index found with {count:,} passages. Skipping rebuild.")
            return col
    except Exception:
        pass

    print("[Index] Building ChromaDB index from FEVER wiki-pages...")
    col = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    embedder = SentenceTransformer(EMBED_MODEL, device="cuda" if _cuda_available() else "cpu")

    wiki_dir = Path(fever_dir) / "wiki-pages"
    if not wiki_dir.exists():
        # Try flat structure
        wiki_dir = Path(fever_dir)

    jsonl_files = sorted(wiki_dir.glob("wiki-*.jsonl"))
    if not jsonl_files:
        jsonl_files = sorted(wiki_dir.glob("*.jsonl"))

    if not jsonl_files:
        raise FileNotFoundError(
            f"No wiki JSONL files found in {wiki_dir}. "
            "Expected wiki-pages/wiki-NNN.jsonl from FEVER dataset."
        )

    print(f"[Index] Found {len(jsonl_files)} wiki JSONL files.")

    ids, texts, metas = [], [], []
    total_added = 0
    BATCH = 512

    for jf in tqdm(jsonl_files, desc="Indexing wiki pages"):
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                doc = json.loads(line)
                page_id = str(doc.get("id", ""))
                lines_text = doc.get("lines", "")

                # lines_text is tab-separated: "0\tsentence\t\t1\tsentence\t..."
                sentences = []
                for seg in lines_text.split("\n"):
                    parts = seg.split("\t")
                    if len(parts) >= 2 and parts[1].strip():
                        sentences.append(parts[1].strip())

                for sent_idx, sentence in enumerate(sentences):
                    if len(sentence) < 20:
                        continue
                    uid = f"{page_id}_{sent_idx}"
                    ids.append(uid)
                    texts.append(sentence[:512])
                    metas.append({"page": page_id, "sent_idx": sent_idx})

                    if len(ids) >= BATCH:
                        embeddings = embedder.encode(
                            texts, batch_size=64, show_progress_bar=False
                        ).tolist()
                        col.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)
                        ids, texts, metas = [], [], []
                        total_added += BATCH

                if total_added >= max_passages:
                    break

        if total_added >= max_passages:
            print(f"[Index] Reached {max_passages:,} passage limit.")
            break

    # Flush remaining
    if ids:
        embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=False).tolist()
        col.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)
        total_added += len(ids)

    del embedder
    print(f"[Index] Done. {total_added:,} passages indexed.")
    return col


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class KaggleRetriever:
    """Embed query → ChromaDB cosine search."""

    def __init__(self, collection, device="cpu"):
        from sentence_transformers import SentenceTransformer
        self.collection = collection
        self.embedder = SentenceTransformer(EMBED_MODEL, device=device)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if not query.strip():
            return []
        embedding = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )
        passages = []
        for i in range(len(results["documents"][0])):
            passages.append({
                "text": results["documents"][0][i],
                "score": 1.0 - results["distances"][0][i],
            })
        return passages


# ---------------------------------------------------------------------------
# NLI-based verifier (replaces Ollama)
# ---------------------------------------------------------------------------

class NLIVerifier:
    """
    Uses cross-encoder/nli-deberta-v3-small to score each evidence passage
    against the claim, then aggregates to a verdict.

    The cross-encoder outputs logits for [contradiction, entailment, neutral].
    We take softmax and aggregate across top-k passages.
    """

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
        """
        Score all passages vs claim, aggregate into SUPPORTED/REFUTED/INSUFFICIENT_EVIDENCE.
        Returns dict with verdict, confidence, and per-passage scores.
        """
        if not passages:
            return {
                "verdict": "INSUFFICIENT_EVIDENCE",
                "confidence": 0.0,
                "entailment": 0.0,
                "contradiction": 0.0,
                "neutral": 0.0,
            }

        import torch

        entailment_scores = []
        contradiction_scores = []
        neutral_scores = []

        for p in passages:
            enc = self.tokenizer(
                claim,
                p["text"],
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**enc).logits[0]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            # cross-encoder/nli-deberta-v3-small label order: contradiction=0, entailment=1, neutral=2
            contradiction_scores.append(float(probs[0]))
            entailment_scores.append(float(probs[1]))
            neutral_scores.append(float(probs[2]))

        max_entailment = max(entailment_scores)
        max_contradiction = max(contradiction_scores)
        mean_neutral = float(np.mean(neutral_scores))

        # Decision rule: take the dominant signal
        if max_entailment > 0.5 and max_entailment > max_contradiction:
            verdict = "SUPPORTED"
            confidence = max_entailment
        elif max_contradiction > 0.5 and max_contradiction > max_entailment:
            verdict = "REFUTED"
            confidence = max_contradiction
        else:
            verdict = "INSUFFICIENT_EVIDENCE"
            confidence = mean_neutral

        return {
            "verdict": verdict,
            "confidence": round(float(confidence), 4),
            "entailment": round(max_entailment, 4),
            "contradiction": round(max_contradiction, 4),
            "neutral": round(mean_neutral, 4),
        }


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def load_fever_sample(fever_dir: str, n_per_class: int = 100, seed: int = 42) -> pd.DataFrame:
    """Load balanced sample of FEVER dev claims (100 per class)."""
    dev_path = Path(fever_dir) / "dev.csv"
    if not dev_path.exists():
        # Try JSONL format
        dev_path = Path(fever_dir) / "dev.jsonl"
        if dev_path.exists():
            rows = []
            with open(dev_path) as f:
                for line in f:
                    d = json.loads(line)
                    rows.append({"claim": d["claim"], "label": d["label"]})
            df = pd.DataFrame(rows)
        else:
            raise FileNotFoundError(f"dev.csv or dev.jsonl not found in {fever_dir}")
    else:
        df = pd.read_csv(dev_path)

    df = df.drop_duplicates(subset=["claim"]).copy()
    df = df[df["label"].isin(FEVER_LABELS)]

    sampled = pd.concat([
        df[df["label"] == label].sample(
            min(n_per_class, len(df[df["label"] == label])),
            random_state=seed
        )
        for label in FEVER_LABELS
    ]).reset_index(drop=True)

    print(f"[Eval] Sampled {len(sampled)} claims: {sampled['label'].value_counts().to_dict()}")
    return sampled


def evaluate(
    fever_dir: str,
    n_samples: int = 300,
    chroma_path: str = "./chroma_index",
    output_path: str = "./baseline_results.json",
    top_k: int = 5,
):
    from sklearn.metrics import classification_report

    device = "cuda" if _cuda_available() else "cpu"
    print(f"[Eval] Device: {device}")

    # 1. Build / load index
    collection = build_index(fever_dir, chroma_path)

    # 2. Init retriever and verifier
    retriever = KaggleRetriever(collection, device=device)
    verifier = NLIVerifier(device=device)

    # 3. Load sample
    n_per_class = n_samples // 3
    df = load_fever_sample(fever_dir, n_per_class=n_per_class)

    # 4. Evaluation loop
    y_true, y_pred = [], []
    records = []
    errors = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        claim = row["claim"]
        true_label = row["label"]
        t0 = time.time()

        try:
            passages = retriever.retrieve(claim, top_k=top_k)
            result = verifier.verify(claim, passages)

            pred_veritas = result["verdict"]
            pred_3class = VERITAS_TO_3CLASS.get(pred_veritas, "NOT ENOUGH INFO")

            y_true.append(true_label)
            y_pred.append(pred_3class)

            records.append({
                "claim": claim,
                "true_label": true_label,
                "predicted": pred_3class,
                "veritas_verdict": pred_veritas,
                "confidence": result["confidence"],
                "entailment": result["entailment"],
                "contradiction": result["contradiction"],
                "latency": round(time.time() - t0, 3),
                "correct": true_label == pred_3class,
            })

        except Exception as e:
            errors += 1
            y_true.append(true_label)
            y_pred.append("NOT ENOUGH INFO")
            records.append({
                "claim": claim,
                "true_label": true_label,
                "predicted": "NOT ENOUGH INFO",
                "veritas_verdict": "ERROR",
                "confidence": 0.0,
                "latency": round(time.time() - t0, 3),
                "correct": False,
                "error": str(e),
            })

    # 5. Compute metrics
    report = classification_report(
        y_true, y_pred,
        labels=FEVER_LABELS,
        output_dict=True,
        zero_division=0
    )

    macro_f1 = report["macro avg"]["f1-score"]
    accuracy = report.get("accuracy", sum(r["correct"] for r in records) / len(records))

    metrics = {
        "model": NLI_MODEL,
        "embed_model": EMBED_MODEL,
        "n_samples": len(df),
        "errors": errors,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": {
            label: {
                "precision": round(report[label]["precision"], 4),
                "recall": round(report[label]["recall"], 4),
                "f1": round(report[label]["f1-score"], 4),
                "support": int(report[label]["support"]),
            }
            for label in FEVER_LABELS
        },
        "avg_latency_s": round(
            sum(r["latency"] for r in records) / len(records), 3
        ),
    }

    # 6. Print results
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Model:     {NLI_MODEL}")
    print(f"  Samples:   {metrics['n_samples']}  (errors: {errors})")
    print(f"  Accuracy:  {metrics['accuracy']}")
    print(f"  Macro-F1:  {metrics['macro_f1']}")
    print(f"  Avg latency: {metrics['avg_latency_s']}s/claim")
    print("\n  Per-class:")
    for label, s in metrics["per_class"].items():
        print(f"    {label:20s} P={s['precision']}  R={s['recall']}  F1={s['f1']}  n={s['support']}")
    print("=" * 60)

    # 7. Save
    output = {"metrics": metrics, "predictions": records}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Eval] Results saved to {output_path}")

    return metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VERITAS Kaggle baseline evaluator")
    parser.add_argument(
        "--fever_dir",
        default="/kaggle/input/fever",
        help="Directory containing FEVER dev.csv (or dev.jsonl) and wiki-pages/",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=300,
        help="Total claims to evaluate (balanced across 3 classes, default: 300)",
    )
    parser.add_argument(
        "--chroma_path",
        default="/kaggle/working/chroma_index",
        help="Path to store/load ChromaDB index",
    )
    parser.add_argument(
        "--output_path",
        default="/kaggle/working/baseline_results.json",
        help="Output JSON path for metrics and predictions",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of passages to retrieve per claim (default: 5)",
    )

    args = parser.parse_args()

    evaluate(
        fever_dir=args.fever_dir,
        n_samples=args.n_samples,
        chroma_path=args.chroma_path,
        output_path=args.output_path,
        top_k=args.top_k,
    )
