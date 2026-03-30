"""
finetuner.py
------------
Fine-tunes FinBERT for tone classification on labeled earnings call chunks.

Architecture (Session 3):
  FinBERT backbone (ProsusAI/finbert)
      -> [CLS] token embedding (768-dim)
      -> Linear(768, 3)
      -> Softmax
  Loss: CrossEntropy  (exactly one label per chunk)
  Labels: 0=optimistic, 1=cautious, 2=negative

Input : data/labeled_chunks.parquet
Output: models/finbert_tone/   (saved model + tokenizer)
        data/eval_report.txt   (classification report on test set)

Usage:
    python finetuner.py
    python finetuner.py --epochs 5 --batch_size 32
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

LABELED_PATH = Path("data/labeled_chunks.parquet")
MODEL_DIR    = Path("models/finbert_tone")
EVAL_PATH    = Path("data/eval_report.txt")
BASE_MODEL   = "ProsusAI/finbert"

LABEL2ID = {"optimistic": 0, "cautious": 1, "negative": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = 3
MAX_TOKENS = 512


# ── 1. Dataset ────────────────────────────────────────────────────────────────

class ToneDataset(Dataset):
    """Tokenizes chunks and returns tensors ready for FinBERT."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ── 2. Load and split data ────────────────────────────────────────────────────

def load_data(path: Path):
    """
    Load labeled chunks and split 80/10/10 train/val/test.
    Stratified on labels to keep class balance across splits.
    """
    df = pd.read_parquet(path)
    log.info(f"Loaded {len(df)} labeled chunks")
    log.info(f"Label distribution:\n{df['label'].value_counts()}")

    # Filter low-confidence labels
    if "confidence" in df.columns:
        before = len(df)
        df = df[df["confidence"] >= 0.6]
        log.info(f"Dropped {before - len(df)} low-confidence labels (<0.6)")

    df["label_id"] = df["label"].map(LABEL2ID)
    df = df.dropna(subset=["label_id"])
    df["label_id"] = df["label_id"].astype(int)

    texts  = df["content"].tolist()
    labels = df["label_id"].tolist()

    # 80/10/10 stratified split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
    )

    log.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── 3. Training loop ──────────────────────────────────────────────────────────

def train(model, loader, optimizer, scheduler, device) -> float:
    model.train()
    total_loss = 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device) -> tuple[float, list, list]:
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            preds = outputs.logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    return total_loss / len(loader), all_preds, all_labels


# ── 4. Main ───────────────────────────────────────────────────────────────────

def run(epochs: int = 3, batch_size: int = 16, lr: float = 2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(LABELED_PATH)

    # Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,  # replaces FinBERT's original 3-class head
    )
    model.to(device)

    # Datasets + loaders
    train_ds = ToneDataset(X_train, y_train, tokenizer)
    val_ds   = ToneDataset(X_val,   y_val,   tokenizer)
    test_ds  = ToneDataset(X_test,  y_test,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    # Optimizer + scheduler with linear warmup
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps   = len(train_loader) * epochs
    warmup_steps  = total_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training
    log.info(f"\nStarting fine-tuning — {epochs} epochs, batch_size={batch_size}, lr={lr}")
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, scheduler, device)
        val_loss, val_preds, val_labels = evaluate(model, val_loader, device)

        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))
        log.info(f"Epoch {epoch}/{epochs} — train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            log.info(f"  -> Best model saved to {MODEL_DIR}")

    # Final evaluation on test set
    log.info("\nEvaluating on test set...")
    _, test_preds, test_labels = evaluate(model, test_loader, device)

    report = classification_report(
        test_labels, test_preds,
        target_names=["optimistic", "cautious", "negative"]
    )
    log.info(f"\n{report}")

    EVAL_PATH.write_text(report)
    log.info(f"Classification report saved to {EVAL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=2e-5)
    args = parser.parse_args()
    run(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
