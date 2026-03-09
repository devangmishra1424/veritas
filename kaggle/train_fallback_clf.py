"""
VERITAS Fallback Classifier Training Script
============================================
Trains a DistilBERT 3-class claim classifier on FEVER.

Input : claim text only (no evidence).
Labels: SUPPORTS → 0 | REFUTES → 1 | NOT ENOUGH INFO → 2

Used as a fallback in verifier.py when Ollama confidence < 0.65.
Saves best checkpoint (by val accuracy) to output_dir/.
Download from Kaggle and place in models/fallback_clf/ for local use.

Install deps (Kaggle cell):
    !pip install transformers==4.36.0 torch pandas scikit-learn tqdm

Usage:
    python kaggle/train_fallback_clf.py \
        --fever_train_path ./data/fever/train.csv \
        --output_dir /kaggle/working/fallback_clf \
        --epochs 3 --batch_size 32 --n_samples 50000
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

MODEL_NAME   = "distilbert-base-uncased"
FEVER_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
LABEL2ID     = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
ID2LABEL     = {v: k for k, v in LABEL2ID.items()}


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

def load_claims(fever_train_path: str, n_samples: int = 50_000, seed: int = 42) -> pd.DataFrame:
    """
    Load a balanced sample of claims from FEVER train.csv.
    n_samples // 3 per class (or as many as available).
    Returns df with columns: claim, label, label_id.
    """
    print(f"[Data] Loading {fever_train_path}...")
    df = pd.read_csv(fever_train_path)
    df = df.drop_duplicates(subset=["claim"]).copy()
    df = df[df["label"].isin(FEVER_LABELS)]
    print(f"[Data] {len(df):,} unique claims.  Dist: {df['label'].value_counts().to_dict()}")

    n_per_class = n_samples // 3
    sampled = (
        pd.concat([
            df[df["label"] == label].sample(
                min(n_per_class, len(df[df["label"] == label])),
                random_state=seed
            )
            for label in FEVER_LABELS
        ])
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    sampled["label_id"] = sampled["label"].map(LABEL2ID)
    print(f"[Data] Sampled {len(sampled):,} claims: {sampled['label'].value_counts().to_dict()}")
    return sampled[["claim", "label", "label_id"]]


class ClaimDataset(Dataset):
    def __init__(self, claims: list, label_ids: list, tokenizer, max_length: int = 128):
        self.claims    = claims
        self.label_ids = label_ids
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.label_ids)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.claims[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.label_ids[idx], dtype=torch.long),
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
    from sklearn.metrics import accuracy_score, classification_report

    set_seed(42)
    device = _device()
    print(f"[Train] Device: {device}  Model: {MODEL_NAME}")

    # Build dataset
    df       = load_claims(args.fever_train_path, n_samples=args.n_samples)
    val_size = max(300, len(df) // 10)
    df_val   = df.iloc[:val_size].reset_index(drop=True)
    df_train = df.iloc[val_size:].reset_index(drop=True)
    print(f"[Train] Train: {len(df_train):,}  Val: {len(df_val):,}")

    # Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    # DataLoaders
    train_ds = ClaimDataset(df_train["claim"].tolist(), df_train["label_id"].tolist(), tokenizer)
    val_ds   = ClaimDataset(df_val["claim"].tolist(),   df_val["label_id"].tolist(),   tokenizer)
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
    best_val_acc = 0.0
    history      = []

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
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]  "):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds   = outputs.logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        val_acc  = accuracy_score(all_labels, all_preds)
        report   = classification_report(
            all_labels, all_preds,
            target_names=FEVER_LABELS,
            output_dict=True,
            zero_division=0,
        )
        macro_f1 = report["macro avg"]["f1-score"]

        print(
            f"\n[Epoch {epoch}] train_loss={avg_train_loss:.4f} | "
            f"val_acc={val_acc:.4f} | macro_f1={macro_f1:.4f}"
        )
        for label in FEVER_LABELS:
            s = report[label]
            print(f"  {label:20s}  P={s['precision']:.3f}  R={s['recall']:.3f}  F1={s['f1-score']:.3f}")

        history.append({
            "epoch":      epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_acc":    round(val_acc, 4),
            "macro_f1":   round(macro_f1, 4),
        })

        # Save best checkpoint by validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"  ✓ Best checkpoint saved (val_acc={best_val_acc:.4f})")

    # Save training summary
    summary = {
        "model":        MODEL_NAME,
        "task":         "claim_classification_3class",
        "label2id":     LABEL2ID,
        "epochs":       args.epochs,
        "batch_size":   args.batch_size,
        "lr":           args.lr,
        "n_samples":    args.n_samples,
        "best_val_acc": round(best_val_acc, 4),
        "history":      history,
        "output_dir":   args.output_dir,
        "usage":        "Place output_dir/ contents in models/fallback_clf/ for local use",
    }
    summary_path = os.path.join(args.output_dir, "train_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Done] Best val_acc={best_val_acc:.4f}")
    print(f"       Model + tokenizer → {args.output_dir}")
    print(f"       Summary           → {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train VERITAS fallback claim classifier (DistilBERT 3-class)"
    )
    parser.add_argument("--fever_train_path", default="./data/fever/train.csv",
                        help="Path to FEVER train.csv")
    parser.add_argument("--output_dir", default="/kaggle/working/fallback_clf",
                        help="Directory to save best model checkpoint and tokenizer")
    parser.add_argument("--epochs",     type=int,   default=3,      help="Training epochs")
    parser.add_argument("--batch_size", type=int,   default=32,     help="Training batch size")
    parser.add_argument("--lr",         type=float, default=2e-5,   help="Learning rate")
    parser.add_argument("--n_samples",  type=int,   default=50_000,
                        help="Total claims to sample from train.csv (balanced, default: 50000)")
    args = parser.parse_args()
    train(args)
