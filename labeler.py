"""
labeler.py — Knowledge distillation labeling for FinBERT fine-tuning.

Generates tone labels for a stratified sample of Prepared Remarks chunks.
Labels: optimistic | cautious | negative

Training data combines:
  - financial_phrasebank (Malo et al., 2014): 3,876 expert-annotated sentences
  - LLM-generated labels via Groq API (Llama-3.1-8B): ~939 earnings call chunks

This script handles the LLM labeling portion. See assemble_finbert_dataset.py
to merge both sources into the final training set.

Setup:
    export ANTHROPIC_API_KEY=sk-ant-...
    python labeler.py
"""

import os
import json
import time
import logging
import pandas as pd
import anthropic
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

CHUNKS_PATH      = Path("data/chunks.parquet")
OUTPUT_PATH      = Path("data/labeled_chunks.parquet")
CHECKPOINT_PATH  = Path("data/labels_checkpoint.parquet")
TARGET_SAMPLES   = 5000
CHECKPOINT_EVERY = 100
LABEL2ID         = {"optimistic": 0, "cautious": 1, "negative": 2}

SYSTEM_PROMPT = """You are a financial analyst classifying earnings call tone.
Return ONLY valid JSON: {"label": "<tone>", "confidence": <0.0-1.0>}
Labels: optimistic | cautious | negative

- optimistic: confidence, growth, strong guidance, positive outlook
- cautious: hedged language, uncertainty, monitoring risks, mixed signals
- negative: warnings, deterioration, downward revisions, distress

No other text. No markdown."""


def label_chunk(client: anthropic.Anthropic, text: str, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=64,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": text[:2000]}],
            )
            result = json.loads(response.content[0].text.strip())
            if result.get("label") in LABEL2ID:
                return result
        except (json.JSONDecodeError, KeyError):
            pass
        except anthropic.RateLimitError:
            time.sleep(2 ** attempt)
        except Exception as e:
            log.warning(f"Attempt {attempt+1} failed: {e}")
            time.sleep(1)
    return None


def sample_chunks(df: pd.DataFrame, n: int) -> pd.DataFrame:
    df   = df[df["section"] == "Prepared"].copy()
    df["year"] = df["date"].str[:4]
    strata    = df.groupby(["sector", "year"])
    sampled   = strata.apply(
        lambda g: g.sample(min(len(g), max(1, n // len(strata))), random_state=42),
        include_groups=False
    ).reset_index(level=[0, 1])
    if len(sampled) < n:
        remaining = df[~df["chunk_id"].isin(sampled["chunk_id"])]
        extra     = remaining.sample(min(n - len(sampled), len(remaining)), random_state=42)
        sampled   = pd.concat([sampled, extra], ignore_index=True)
    return sampled.head(n)


def run():
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    df     = pd.read_parquet(CHUNKS_PATH)
    sample = sample_chunks(df, TARGET_SAMPLES)
    log.info(f"Sampled {len(sample)} Prepared Remarks chunks")

    results, done_ids = [], set()
    if CHECKPOINT_PATH.exists():
        done_df  = pd.read_parquet(CHECKPOINT_PATH)
        done_df  = done_df[done_df["confidence"] >= 0.6]
        results  = done_df.to_dict("records")
        done_ids = set(done_df["chunk_id"])
        log.info(f"Resuming — {len(done_ids)} already labeled")

    sample = sample[~sample["chunk_id"].isin(done_ids)]
    log.info(f"{len(sample)} chunks remaining")

    failed = 0
    for i, row in enumerate(sample.itertuples(), 1):
        result = label_chunk(client, row.content)
        if result is None:
            failed += 1
            continue
        results.append({
            "chunk_id":   row.chunk_id,
            "ticker":     row.ticker,
            "sector":     row.sector,
            "company":    row.company,
            "date":       row.date,
            "quarter":    row.quarter,
            "content":    row.content,
            "label":      result["label"],
            "label_id":   LABEL2ID[result["label"]],
            "confidence": result["confidence"],
        })
        if i % CHECKPOINT_EVERY == 0:
            pd.DataFrame(results).to_parquet(CHECKPOINT_PATH, index=False)
            log.info(f"  [{i}/{len(sample)}] labeled | failed: {failed}")
        time.sleep(0.3)

    df_out = pd.DataFrame(results)
    df_out.to_parquet(OUTPUT_PATH, index=False)
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    log.info(f"\nDone — {len(df_out)} chunks labeled → {OUTPUT_PATH}")
    log.info(f"Failed: {failed}")
    log.info(f"\nLabel distribution:\n{df_out['label'].value_counts()}")


if __name__ == "__main__":
    run()
