"""
VERITAS QLoRA Fine-tuning Script
==================================
Fine-tunes Qwen2.5-1.5B-Instruct with QLoRA on FEVER + LIAR for verdict generation.

Input format:
    "Claim: {claim}\nEvidence: {evidence_text}\nVerdict: {SUPPORTS|REFUTES|NOT ENOUGH INFO}"

Only the verdict tokens contribute to the loss (prompt tokens masked with -100).

LIAR label mapping (int 0-5 → verdict string):
    0, 1 → REFUTES          (pants_on_fire, false)
    2, 3 → NOT ENOUGH INFO  (barely_true, half_true)
    4, 5 → SUPPORTS         (mostly_true, true)

FEVER: label column used as-is (SUPPORTS / REFUTES / NOT ENOUGH INFO).
       evidence_wiki_url used as evidence when available, else "No evidence available."

LoRA adapter (not full weights) is saved to --output_dir.
Download from Kaggle and place in models/qlora_adapter/ for local loading.

Install deps (Kaggle cell):
    !pip install -q transformers peft bitsandbytes accelerate datasets tqdm

Usage:
    python kaggle/train_verifier_qlora.py \
        --fever_train_path ./data/fever/train.csv \
        --liar_train_path  ./data/liar/train.csv \
        --output_dir       /kaggle/working/qlora_adapter \
        --epochs 3 --max_samples 50000
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
GRAD_ACCUM = 8
MAX_LENGTH = 512

LIAR_LABEL_MAP = {
    0: "REFUTES",         # pants_on_fire
    1: "REFUTES",         # false
    2: "NOT ENOUGH INFO", # barely_true
    3: "NOT ENOUGH INFO", # half_true
    4: "SUPPORTS",        # mostly_true
    5: "SUPPORTS",        # true
}

VERDICT_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


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
# Data loading
# ---------------------------------------------------------------------------

def load_fever(fever_train_path: str, max_samples: int = 50_000, seed: int = 42) -> list[dict]:
    """
    Load FEVER train.csv → list of {claim, evidence, verdict} dicts.
    Deduplicates on claim. evidence_wiki_url used as evidence when present.
    """
    print(f"[Data] Loading FEVER from {fever_train_path}...")
    df = pd.read_csv(fever_train_path)
    df = df[df["label"].isin(VERDICT_LABELS)].copy()

    # Aggregate evidence per claim: collect unique wiki_urls into one string
    def _agg_evidence(group):
        urls = group["evidence_wiki_url"].dropna().unique().tolist()
        return ", ".join(urls) if urls else ""

    agg = (
        df.groupby("claim", sort=False)
        .apply(lambda g: pd.Series({
            "verdict":  g["label"].iloc[0],
            "evidence": _agg_evidence(g),
        }))
        .reset_index()
    )

    agg["evidence"] = agg["evidence"].apply(
        lambda e: f"Source: {e}" if e else "No evidence available."
    )

    if len(agg) > max_samples:
        agg = agg.sample(max_samples, random_state=seed).reset_index(drop=True)

    records = agg[["claim", "evidence", "verdict"]].to_dict("records")
    print(f"[Data] FEVER: {len(records):,} records  "
          f"({pd.Series([r['verdict'] for r in records]).value_counts().to_dict()})")
    return records


def load_liar(liar_train_path: str, seed: int = 42) -> list[dict]:
    """
    Load LIAR train.csv → list of {claim, evidence, verdict} dicts.
    Uses 'statement' as the claim. Maps int labels 0-5 to verdict strings.
    No evidence available for LIAR — uses "No evidence available."
    """
    print(f"[Data] Loading LIAR from {liar_train_path}...")
    df = pd.read_csv(liar_train_path)
    df = df.dropna(subset=["statement", "label"]).copy()
    df = df[df["label"].isin(LIAR_LABEL_MAP.keys())].copy()

    records = [
        {
            "claim":    str(row.statement).strip(),
            "evidence": "No evidence available.",
            "verdict":  LIAR_LABEL_MAP[int(row.label)],
        }
        for row in df.itertuples(index=False)
    ]
    print(f"[Data] LIAR: {len(records):,} records  "
          f"({pd.Series([r['verdict'] for r in records]).value_counts().to_dict()})")
    return records


def build_records(
    fever_train_path: str,
    liar_train_path:  str,
    max_samples:      int = 50_000,
    seed:             int = 42,
) -> list[dict]:
    """Combine FEVER (up to max_samples) + all LIAR, shuffle."""
    fever   = load_fever(fever_train_path, max_samples=max_samples, seed=seed)
    liar    = load_liar(liar_train_path, seed=seed)
    combined = fever + liar
    random.Random(seed).shuffle(combined)
    print(f"[Data] Combined: {len(combined):,} records total.")
    return combined


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VerifierDataset(Dataset):
    """
    Returns tokenized (prompt + verdict) pairs with labels masked on the prompt.
    Only verdict tokens (and EOS) contribute to the loss.
    """

    def __init__(self, records: list[dict], tokenizer, max_length: int = MAX_LENGTH):
        self.records    = records
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.bos_ids    = (
            [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        prompt  = (
            f"Claim: {rec['claim']}\n"
            f"Evidence: {rec['evidence']}\n"
            f"Verdict:"
        )
        verdict_str = f" {rec['verdict']}"

        # Tokenize separately so prompt boundary is exact
        prompt_ids  = self.tokenizer.encode(prompt,      add_special_tokens=False)
        verdict_ids = self.tokenizer.encode(verdict_str, add_special_tokens=False)
        eos_ids     = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else []

        input_ids = self.bos_ids + prompt_ids + verdict_ids + eos_ids
        labels    = ([-100] * (len(self.bos_ids) + len(prompt_ids))) + verdict_ids + eos_ids

        # Truncate to max_length (from the left on the prompt if needed)
        if len(input_ids) > self.max_length:
            overflow   = len(input_ids) - self.max_length
            input_ids  = input_ids[overflow:]
            labels     = labels[overflow:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
        }


class PaddingCollator:
    """Right-pads input_ids and labels to the longest sequence in the batch."""

    def __init__(self, pad_token_id: int):
        self.pad_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(b["input_ids"].size(0) for b in batch)

        input_ids_list, attn_mask_list, labels_list = [], [], []
        for b in batch:
            seq_len  = b["input_ids"].size(0)
            pad_len  = max_len - seq_len
            input_ids_list.append(
                torch.cat([b["input_ids"], torch.full((pad_len,), self.pad_id, dtype=torch.long)])
            )
            attn_mask_list.append(
                torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
            )
            labels_list.append(
                torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
            )

        return {
            "input_ids":      torch.stack(input_ids_list),
            "attention_mask": torch.stack(attn_mask_list),
            "labels":         torch.stack(labels_list),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def count_parameters(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total


def adapter_size_mb(output_dir: str) -> float:
    total = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, f))
    )
    return round(total / 1024 / 1024, 2)


def train(args):
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        get_linear_schedule_with_warmup,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    set_seed(42)
    device = _device()
    print(f"[Train] Device: {device}  Model: {MODEL_NAME}")

    # ------------------------------------------------------------------
    # 1. Build dataset
    # ------------------------------------------------------------------
    records = build_records(
        args.fever_train_path,
        args.liar_train_path,
        max_samples=args.max_samples,
    )

    # ------------------------------------------------------------------
    # 2. Tokenizer
    # ------------------------------------------------------------------
    print("[Train] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # 3. Model — 4-bit QLoRA
    # ------------------------------------------------------------------
    print("[Train] Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit               = True,
        bnb_4bit_quant_type        = "nf4",
        bnb_4bit_compute_dtype     = torch.float16,
        bnb_4bit_use_double_quant  = True,   # nested quantization for memory
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config = bnb_config,
        device_map          = "auto",
        trust_remote_code   = True,
    )
    model.config.use_cache    = False   # required for gradient checkpointing
    model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare for k-bit training (freezes quant weights, enables grad checkpointing)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # ------------------------------------------------------------------
    # 4. LoRA adapters
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        r              = 16,
        lora_alpha     = 32,
        lora_dropout   = 0.05,
        target_modules = ["q_proj", "v_proj"],
        bias           = "none",
        task_type      = "CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable, total = count_parameters(model)
    print(f"[Train] Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.4f}%)")

    # ------------------------------------------------------------------
    # 5. DataLoader
    # ------------------------------------------------------------------
    dataset    = VerifierDataset(records, tokenizer, max_length=MAX_LENGTH)
    collator   = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = 0,           # 0 avoids CUDA fork issues on Kaggle
        pin_memory  = (device == "cuda"),
        collate_fn  = collator,
    )
    print(f"[Train] {len(dataset):,} samples  {len(dataloader):,} batches/epoch  "
          f"grad_accum={GRAD_ACCUM}  effective_bs={args.batch_size * GRAD_ACCUM}")

    # ------------------------------------------------------------------
    # 6. Optimizer + scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    # Scheduler steps on optimizer updates (not raw batches)
    optimizer_steps_per_epoch = max(1, len(dataloader) // GRAD_ACCUM)
    total_optimizer_steps     = optimizer_steps_per_epoch * args.epochs
    warmup_steps              = int(0.10 * total_optimizer_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_optimizer_steps,
    )
    print(f"[Train] Optimizer steps: {total_optimizer_steps}  warmup: {warmup_steps}")

    os.makedirs(args.output_dir, exist_ok=True)
    history = []

    # ------------------------------------------------------------------
    # 7. Training loop
    # ------------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss  = 0.0
        optim_step    = 0
        optimizer.zero_grad()

        for step, batch in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        ):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                labels         = labels,
            )

            # Scale loss for gradient accumulation
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()
            running_loss += outputs.loss.item()   # track unscaled loss

            is_last_step    = (step + 1) == len(dataloader)
            is_accum_step   = (step + 1) % GRAD_ACCUM == 0

            if is_accum_step or is_last_step:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optim_step += 1

        avg_loss = running_loss / len(dataloader)
        print(f"\n[Epoch {epoch}] avg_loss={avg_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")
        history.append({"epoch": epoch, "avg_loss": round(avg_loss, 4)})

        # Save adapter after every epoch (overwrites → last epoch = final adapter)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"  ✓ Adapter saved to {args.output_dir}")

    # ------------------------------------------------------------------
    # 8. Final summary
    # ------------------------------------------------------------------
    trainable_final, total_final = count_parameters(model)
    adapter_mb = adapter_size_mb(args.output_dir)

    summary = {
        "base_model":       MODEL_NAME,
        "lora_r":           16,
        "lora_alpha":       32,
        "lora_dropout":     0.05,
        "target_modules":   ["q_proj", "v_proj"],
        "quant":            "nf4_4bit",
        "epochs":           args.epochs,
        "batch_size":       args.batch_size,
        "grad_accum":       GRAD_ACCUM,
        "effective_bs":     args.batch_size * GRAD_ACCUM,
        "lr":               args.lr,
        "max_samples":      args.max_samples,
        "total_records":    len(records),
        "trainable_params": trainable_final,
        "total_params":     total_final,
        "trainable_pct":    round(100 * trainable_final / total_final, 4),
        "adapter_size_mb":  adapter_mb,
        "history":          history,
        "output_dir":       args.output_dir,
        "usage":            "Place output_dir/ contents in models/qlora_adapter/ for local use",
    }
    summary_path = os.path.join(args.output_dir, "qlora_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("QLORA TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Trainable params : {trainable_final:,} / {total_final:,} "
          f"({summary['trainable_pct']}%)")
    print(f"  Adapter size     : {adapter_mb} MB")
    print(f"  Adapter saved to : {args.output_dir}")
    print(f"  Summary          : {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning of Qwen2.5-1.5B-Instruct on FEVER + LIAR"
    )
    parser.add_argument("--fever_train_path", default="./data/fever/train.csv",
                        help="Path to FEVER train.csv")
    parser.add_argument("--liar_train_path",  default="./data/liar/train.csv",
                        help="Path to LIAR train.csv")
    parser.add_argument("--output_dir",       default="/kaggle/working/qlora_adapter",
                        help="Directory to save LoRA adapter and tokenizer")
    parser.add_argument("--epochs",     type=int,   default=3,      help="Training epochs")
    parser.add_argument("--batch_size", type=int,   default=4,      help="Per-device batch size")
    parser.add_argument("--lr",         type=float, default=2e-4,   help="Learning rate")
    parser.add_argument("--max_samples", type=int,  default=50_000,
                        help="Max samples drawn from FEVER (all LIAR always included, default: 50000)")
    args = parser.parse_args()
    train(args)
