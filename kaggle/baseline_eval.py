"""
VERITAS Kaggle Baseline Evaluator
==================================
Self-contained evaluation script for Kaggle T4 GPU environment.

Differences from local pipeline:
- No Ollama: uses cross-encoder/nli-deberta-v3-small for NLI verdicts
- Builds ChromaDB index from train.csv (evidence_wiki_url column)
  Passage format: "{claim} [Source: {evidence_wiki_url}]"  (matches build_index.py)
- Evaluates 300 balanced claims (100 per class) from dev.csv
- Outputs Macro-F1, Precision, Recall per class

Install deps (Kaggle notebook cell):
    !pip install chromadb sentence-transformers transformers scikit-learn pandas tqdm

Usage:
    python kaggle/baseline_eval.py \
        --fever_train_path ./data/fever/train.csv \
        --fever_dev_path   ./data/fever/dev.csv \
        --n_samples        300 \
        --chroma_path      /kaggle/working/chroma_index \
        --output_path      /kaggle/working/baseline_results.json
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NLI_MODEL   = "cross-encoder/nli-deberta-v3-small"
EMBED_MODEL  = "all-MiniLM-L6-v2"
COLLECTION_NAME = "veritas_fever_baseline"
BATCH_SIZE   = 500
MAX_PASSAGES = 50_000

FEVER_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

# Internal verdict → FEVER 3-class
VERITAS_TO_3CLASS = {
    "SUPPORTED":             "SUPPORTS",
    "REFUTED":               "REFUTES",
    "INSUFFICIENT_EVIDENCE": "NOT ENOUGH INFO",
}


# ---------------------------------------------------------------------------
# Index builder  (mirrors data/build_index.py logic)
# ---------------------------------------------------------------------------

def build_index(train_path: str, chroma_path: str):
    """
    Build ChromaDB index from train.csv using evidence_wiki_url.
    Passage format: "{claim} [Source: {evidence_wiki_url}]"
    Skips rebuild if the collection already has documents.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path=chroma_path)

    # Skip rebuild if index already populated
    try:
        col = client.get_collection(COLLECTION_NAME)
        if col.count() > 0:
            print(f"[Index] Existing index found with {col.count():,} passages. Skipping rebuild.")
            return col
    except Exception:
        pass

    print(f"[Index] Loading train.csv from {train_path}...")
    df = pd.read_csv(train_path)
    print(f"[Index] Total rows: {len(df):,}  Columns: {df.columns.tolist()}")

    # Mirror build_index.py: drop rows without evidence, exclude NOT ENOUGH INFO
    df = df.dropna(subset=["evidence_wiki_url"]).copy()
    df = df[df["label"] != "NOT ENOUGH INFO"].copy()
    print(f"[Index] Rows with evidence: {len(df):,}")

    passages, ids, metadatas = [], [], []
    seen = set()

    for idx, row in df.iterrows():
        claim    = str(row["claim"]).strip()
        wiki_url = str(row["evidence_wiki_url"]).strip()
        text     = f"{claim} [Source: {wiki_url}]"

        if text not in seen and len(text) > 20:
            seen.add(text)
            passages.append(text)
            ids.append(f"fever_{idx}")
            metadatas.append({
                "source":   "fever",
                "wiki_url": wiki_url[:200],
                "label":    str(row["label"]),
                "claim":    claim[:200],
            })

        if len(passages) >= MAX_PASSAGES:
            break

    print(f"[Index] Extracted {len(passages):,} unique passages. Building ChromaDB collection...")

    col = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    device = "cuda" if _cuda_available() else "cpu"
    embedder = SentenceTransformer(EMBED_MODEL, device=device)

    for i in tqdm(range(0, len(passages), BATCH_SIZE), desc="Indexing"):
        batch_p   = passages[i : i + BATCH_SIZE]
        batch_ids = ids[i : i + BATCH_SIZE]
        batch_m   = metadatas[i : i + BATCH_SIZE]

        embeddings = embedder.encode(batch_p, show_progress_bar=False).tolist()
        col.add(documents=batch_p, embeddings=embeddings, ids=batch_ids, metadatas=batch_m)

    del embedder
    print(f"[Index] Done. {col.count():,} passages indexed.")
    return col


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class KaggleRetriever:
    """Embed query → ChromaDB cosine search."""

    def __init__(self, collection, device="cpu"):
        from sentence_transformers import SentenceTransformer
        self.collection = collection
        self.embedder   = SentenceTransformer(EMBED_MODEL, device=device)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if not query.strip():
            return []
        embedding = self.embedder.encode(query).tolist()
        results   = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        return [
            {
                "text":  results["documents"][0][i],
                "score": 1.0 - results["distances"][0][i],
            }
            for i in range(len(results["documents"][0]))
        ]


# ---------------------------------------------------------------------------
# NLI-based verifier  (replaces Ollama)
# ---------------------------------------------------------------------------

class NLIVerifier:
    """
    Uses cross-encoder/nli-deberta-v3-small to score each evidence passage
    against the claim, then aggregates to a verdict.

    Label order for this checkpoint: contradiction=0, entailment=1, neutral=2
    Decision rule: take max signal across passages; threshold at 0.5.
    """

    def __init__(self, device="cpu"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        print(f"[NLIVerifier] Loading {NLI_MODEL} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.model     = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.torch  = torch
        print("[NLIVerifier] Ready.")

    def verify(self, claim: str, passages: list[dict]) -> dict:
        if not passages:
            return {
                "verdict":       "INSUFFICIENT_EVIDENCE",
                "confidence":    0.0,
                "entailment":    0.0,
                "contradiction": 0.0,
                "neutral":       0.0,
            }

        import torch

        ent_scores, con_scores, neu_scores = [], [], []

        for p in passages:
            enc = self.tokenizer(
                claim, p["text"],
                max_length=512, truncation=True, padding=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                probs = torch.softmax(self.model(**enc).logits[0], dim=-1).cpu().numpy()

            con_scores.append(float(probs[0]))
            ent_scores.append(float(probs[1]))
            neu_scores.append(float(probs[2]))

        max_ent  = max(ent_scores)
        max_con  = max(con_scores)
        mean_neu = float(np.mean(neu_scores))

        if max_ent > 0.5 and max_ent > max_con:
            verdict    = "SUPPORTED"
            confidence = max_ent
        elif max_con > 0.5 and max_con > max_ent:
            verdict    = "REFUTED"
            confidence = max_con
        else:
            verdict    = "INSUFFICIENT_EVIDENCE"
            confidence = mean_neu

        return {
            "verdict":       verdict,
            "confidence":    round(float(confidence), 4),
            "entailment":    round(max_ent, 4),
            "contradiction": round(max_con, 4),
            "neutral":       round(mean_neu, 4),
        }


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_fever_dev_sample(dev_path: str, n_per_class: int = 100, seed: int = 42) -> pd.DataFrame:
    """Load balanced sample of FEVER dev claims (n_per_class per label)."""
    print(f"[Eval] Loading dev set from {dev_path}...")
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


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    fever_train_path: str = "./data/fever/train.csv",
    fever_dev_path:   str = "./data/fever/dev.csv",
    n_samples:        int = 300,
    chroma_path:      str = "./chroma_index",
    output_path:      str = "./baseline_results.json",
    top_k:            int = 5,
):
    from sklearn.metrics import classification_report

    device = "cuda" if _cuda_available() else "cpu"
    print(f"[Eval] Device: {device}")

    # 1. Build / load index
    collection = build_index(fever_train_path, chroma_path)

    # 2. Init retriever and verifier
    retriever = KaggleRetriever(collection, device=device)
    verifier  = NLIVerifier(device=device)

    # 3. Load balanced dev sample
    n_per_class = n_samples // 3
    df = load_fever_dev_sample(fever_dev_path, n_per_class=n_per_class)

    # 4. Evaluation loop
    y_true, y_pred = [], []
    records = []
    errors  = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        claim      = row["claim"]
        true_label = row["label"]
        t0 = time.time()

        try:
            passages     = retriever.retrieve(claim, top_k=top_k)
            result       = verifier.verify(claim, passages)
            pred_veritas = result["verdict"]
            pred_3class  = VERITAS_TO_3CLASS.get(pred_veritas, "NOT ENOUGH INFO")

            y_true.append(true_label)
            y_pred.append(pred_3class)
            records.append({
                "claim":         claim,
                "true_label":    true_label,
                "predicted":     pred_3class,
                "veritas_verdict": pred_veritas,
                "confidence":    result["confidence"],
                "entailment":    result["entailment"],
                "contradiction": result["contradiction"],
                "latency":       round(time.time() - t0, 3),
                "correct":       true_label == pred_3class,
            })

        except Exception as e:
            errors += 1
            y_true.append(true_label)
            y_pred.append("NOT ENOUGH INFO")
            records.append({
                "claim":           claim,
                "true_label":      true_label,
                "predicted":       "NOT ENOUGH INFO",
                "veritas_verdict": "ERROR",
                "confidence":      0.0,
                "latency":         round(time.time() - t0, 3),
                "correct":         False,
                "error":           str(e),
            })

    # 5. Compute metrics
    report   = classification_report(
        y_true, y_pred,
        labels=FEVER_LABELS,
        output_dict=True,
        zero_division=0
    )
    macro_f1 = report["macro avg"]["f1-score"]
    accuracy = report.get(
        "accuracy",
        sum(r["correct"] for r in records) / len(records)
    )

    metrics = {
        "model":        NLI_MODEL,
        "embed_model":  EMBED_MODEL,
        "n_samples":    len(df),
        "errors":       errors,
        "accuracy":     round(accuracy, 4),
        "macro_f1":     round(macro_f1, 4),
        "per_class": {
            label: {
                "precision": round(report[label]["precision"], 4),
                "recall":    round(report[label]["recall"], 4),
                "f1":        round(report[label]["f1-score"], 4),
                "support":   int(report[label]["support"]),
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
    print(f"  Model:       {NLI_MODEL}")
    print(f"  Samples:     {metrics['n_samples']}  (errors: {errors})")
    print(f"  Accuracy:    {metrics['accuracy']}")
    print(f"  Macro-F1:    {metrics['macro_f1']}")
    print(f"  Avg latency: {metrics['avg_latency_s']}s/claim")
    print("\n  Per-class:")
    for label, s in metrics["per_class"].items():
        print(f"    {label:20s}  P={s['precision']}  R={s['recall']}  F1={s['f1']}  n={s['support']}")
    print("=" * 60)

    # 7. Save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"metrics": metrics, "predictions": records}, f, indent=2)
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
        "--fever_train_path",
        default="./data/fever/train.csv",
        help="Path to FEVER train.csv (used to build ChromaDB index)",
    )
    parser.add_argument(
        "--fever_dev_path",
        default="./data/fever/dev.csv",
        help="Path to FEVER dev.csv (used for evaluation)",
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
        fever_train_path=args.fever_train_path,
        fever_dev_path=args.fever_dev_path,
        n_samples=args.n_samples,
        chroma_path=args.chroma_path,
        output_path=args.output_path,
        top_k=args.top_k,
    )
