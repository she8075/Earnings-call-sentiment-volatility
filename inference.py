"""
inference.py
------------
Applies the fine-tuned FinBERT tone classifier on all Prepared Remarks chunks
and aggregates the results into one tone score per (ticker, date).

Output: data/tone_scores.parquet

Usage:
    python inference.py
    python inference.py --batch_size 16
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

CHUNKS_PATH = Path("data/chunks.parquet")
MODEL_DIR   = Path("models/finbert_tone")
OUTPUT_PATH = Path("data/tone_scores.parquet")
MAX_TOKENS  = 512


def run_inference(df: pd.DataFrame, model, tokenizer, device, batch_size: int) -> pd.DataFrame:
    """
    Tokenize and score chunks batch by batch — no full-dataset tokenization in memory.
    """
    texts     = df["content"].tolist()
    all_probs = []
    model.eval()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        )
        input_ids      = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()

        all_probs.append(probs)

        if (i // batch_size + 1) % 100 == 0:
            log.info(f"  {i + batch_size}/{len(texts)} chunks scored")

    probs_arr = np.vstack(all_probs)

    df = df.copy()
    df["p_optimistic"] = probs_arr[:, 0]
    df["p_cautious"]   = probs_arr[:, 1]
    df["p_negative"]   = probs_arr[:, 2]
    return df


def aggregate_tone_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate chunk-level probabilities to one row per call."""
    agg = df.groupby(["ticker", "sector", "company", "date", "quarter"]).agg(
        optimistic_share=("p_optimistic", "mean"),
        cautious_share  =("p_cautious",   "mean"),
        negative_share  =("p_negative",   "mean"),
        n_chunks        =("chunk_id",      "count"),
    ).reset_index()

    share_sum = agg[["optimistic_share", "cautious_share", "negative_share"]].sum(axis=1)
    assert (share_sum.between(0.98, 1.02)).all(), "Tone shares do not sum to 1"
    return agg


def run(batch_size: int = 16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    df = pd.read_parquet(CHUNKS_PATH)
    df = df[df["section"] == "Prepared"].copy()
    log.info(f"Scoring {len(df)} Prepared Remarks chunks across {df['ticker'].nunique()} tickers")

    log.info(f"Loading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)

    df_scored   = run_inference(df, model, tokenizer, device, batch_size)
    tone_scores = aggregate_tone_scores(df_scored)
    tone_scores.to_parquet(OUTPUT_PATH, index=False)

    log.info(f"\nDone — {len(tone_scores)} calls scored -> {OUTPUT_PATH}")
    log.info(f"\nTone score summary:\n{tone_scores[['cautious_share','optimistic_share','negative_share']].describe().round(4)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    run(batch_size=args.batch_size)