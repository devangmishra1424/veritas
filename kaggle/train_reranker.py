"""
VERITAS Reranker Training Script
=================================
Trains a DistilBERT cross-encoder to rerank retrieved passages.

Task: Binary relevance classification.
  Positive (label=1): (claim, "{claim} [Source: {evidence_wiki_url}]") — gold evidence
  Negative (label=0): (claim, "{other_claim} [Source: {other_wiki_url}]") — 3 per positive

Saves best checkpoint (by val loss) to output_dir/.
Download from Kaggle and place in models/reranker/ for local use in retriever.py.

Install deps (Kaggle cell):
    !pip install transformers==4.36.0 torch pandas scikit-learn tqdm

Usage:
    python kaggle/train_reranker.py \
        --fever_train_path ./data/fever/train.csv \
        --output_dir /kaggle/working/reranker \
        --epochs 3 --batch_size 32 --max_pairs 50000
"""

import argparse
import json
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

MODEL_NAME = "distilbert-base-uncased"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_pairs(fever_train_path: str, max_pairs: int = 50_000, seed: int = 42) -> pd.DataFrame:
    """
    Build (claim, passage, label) training pairs from FEVER train.csv.

    Positive: (claim, "{claim} [Source: {wiki_url}]") → label=1
    Negative: 3 random passages from other claims → label=0
    Total pairs capped at max_pairs.
    """
    rng = random.Random(seed)

    print(f"[Data] Loading {fever_train_path}...")
    df = pd.read_csv(fever_train_path)
    df = df.dropna(subset=["evidence_wiki_url"])
    df = df[df["label"] != "NOT ENOUGH INFO"].copy()
    print(f"[Data] {len(df):,} evidence rows after filtering.")

    # Build passage text column
    df["passage"] = (
        df["claim"].str.strip()
        + " [Source: "
        + df["evidence_wiki_url"].str.strip()
        + "]"
    )

    # Map each claim → its set of gold passages (used to filter negatives)
    claim_to_gold: dict[str, set] = {}
    for row in df.itertuples(index=False):
        claim = str(row.claim).strip()
        claim_to_gold.setdefault(claim, set()).add(row.passage)

    # Unique positive pairs
    df_pos = (
        df[["claim", "passage"]]
        .drop_duplicates()
        .copy()
        .reset_index(drop=True)
    )
    df_pos["claim"] = df_pos["claim"].str.strip()
    df_pos["label"] = 1

    # Pool of all unique passages for negative sampling
    all_passages = list(df["passage"].unique())
    print(f"[Data] {len(all_passages):,} unique passages in pool.")

    # Cap positives so total pairs fit within max_pairs (1 pos + 3 neg = 4 each)
    max_pos = max_pairs // 4
    if len(df_pos) > max_pos:
        df_pos = df_pos.sample(max_pos, random_state=seed).reset_index(drop=True)
    print(f"[Data] {len(df_pos):,} positive pairs.")

    # Sample 3 negatives per positive from passages of other claims
    neg_claims, neg_passages = [], []
    for row in tqdm(df_pos.itertuples(index=False), total=len(df_pos), desc="Sampling negatives"):
        claim    = row.claim
        gold_set = claim_to_gold.get(claim, set())

        # Oversample then filter; deduplicate within this anchor
        candidates = rng.choices(all_passages, k=min(40, len(all_passages)))
        chosen: list[str] = []
        seen_neg: set[str] = set()
        for p in candidates:
            if p not in gold_set and p not in seen_neg:
                chosen.append(p)
                seen_neg.add(p)
            if len(chosen) == 3:
                break

        for neg in chosen:
            neg_claims.append(claim)
            neg_passages.append(neg)

    df_neg = pd.DataFrame({"claim": neg_claims, "passage": neg_passages, "label": 0})
    df_pairs = (
        pd.concat([df_pos, df_neg], ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    pos_n = int(df_pairs["label"].sum())
    print(f"[Data] Final pairs: {len(df_pairs):,}  pos={pos_n:,}  neg={len(df_pairs) - pos_n:,}")
    return df_pairs


class RerankerDataset(Dataset):
    def __init__(self, claims: list, passages: list, labels: list, tokenizer, max_length: int = 256):
        self.claims    = claims
        self.passages  = passages
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.claims[idx],
            self.passages[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
    from sklearn.metrics import accuracy_score, roc_auc_score

    set_seed(42)
    device = _device()
    print(f"[Train] Device: {device}  Model: {MODEL_NAME}")

    # Build dataset
    df_pairs = build_pairs(args.fever_train_path, max_pairs=args.max_pairs)
    val_size  = max(200, len(df_pairs) // 10)
    df_val    = df_pairs.iloc[:val_size].reset_index(drop=True)
    df_train  = df_pairs.iloc[val_size:].reset_index(drop=True)
    print(f"[Train] Train: {len(df_train):,}  Val: {len(df_val):,}")

    # Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    # DataLoaders
    train_ds = RerankerDataset(
        df_train["claim"].tolist(), df_train["passage"].tolist(), df_train["label"].tolist(), tokenizer
    )
    val_ds = RerankerDataset(
        df_val["claim"].tolist(), df_val["passage"].tolist(), df_val["label"].tolist(), tokenizer
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=(device == "cuda"))

    # Optimizer + linear warmup scheduler
    optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(0.10 * total_steps)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    history       = []

    for epoch in range(1, args.epochs + 1):

        # --- Train ---
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += outputs.loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        all_probs, all_preds, all_labels = [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]  "):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["label"].to(device)

                outputs  = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

                probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        avg_val_loss = val_loss / len(val_loader)
        auc = roc_auc_score(all_labels, all_probs)
        acc = accuracy_score(all_labels, all_preds)

        print(
            f"\n[Epoch {epoch}] train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f} | AUC={auc:.4f} | Acc={acc:.4f}"
        )
        history.append({
            "epoch":      epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss":   round(avg_val_loss, 4),
            "auc":        round(auc, 4),
            "accuracy":   round(acc, 4),
        })

        # Save best checkpoint by validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"  ✓ Best checkpoint saved (val_loss={best_val_loss:.4f})")

    # Save training summary
    summary = {
        "model":         MODEL_NAME,
        "task":          "passage_reranking",
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "lr":            args.lr,
        "max_pairs":     args.max_pairs,
        "best_val_loss": round(best_val_loss, 4),
        "history":       history,
        "output_dir":    args.output_dir,
        "usage":         "Place output_dir/ contents in models/reranker/ for local use",
    }
    summary_path = os.path.join(args.output_dir, "train_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Done] Best val_loss={best_val_loss:.4f}")
    print(f"       Model + tokenizer → {args.output_dir}")
    print(f"       Summary           → {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train VERITAS passage reranker (DistilBERT cross-encoder)"
    )
    parser.add_argument("--fever_train_path", default="./data/fever/train.csv",
                        help="Path to FEVER train.csv")
    parser.add_argument("--output_dir", default="/kaggle/working/reranker",
                        help="Directory to save best model checkpoint and tokenizer")
    parser.add_argument("--epochs",     type=int,   default=3,      help="Training epochs")
    parser.add_argument("--batch_size", type=int,   default=32,     help="Training batch size")
    parser.add_argument("--lr",         type=float, default=2e-5,   help="Learning rate")
    parser.add_argument("--max_pairs",  type=int,   default=50_000,
                        help="Max (claim, passage) pairs to construct (default: 50000)")
    args = parser.parse_args()
    train(args)
