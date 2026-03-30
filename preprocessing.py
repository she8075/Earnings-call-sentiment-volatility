"""
preprocessing.py
----------------
Complete pipeline: JSON/JSONL => chunks.parquet ready for FinBERT.

Steps:
1. Upload JSON/JSONL
2. Resolve ticker via sp500_aliases.csv file
3. Parse date + quarter
4. Strip header Event participants
5. Segmentation by speaker
6. Section labeling: using cumsum trick (when analyst starts talking, whats next is QA)
7. text cleaning + boilerplate operator filtering
8. Chunking semantic <510 tokens for FinBERT (semchunk + FinBERT tokenizer)
9. Validation + Quality report

Usage:
    python preprocessing.py --input data/raw --ouput data/chunks.parquet
"""

import re
import csv
import json
import uuid
import logging
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
from transformers import AutoTokenizer
from semchunk import chunkerify

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log=logging.getLogger(__name__)

ALIASES_CSV= Path("data/sp500_aliases.csv")
OUTPUT_PATH= Path("data/chunks.parquet")
CHECKPOINT_PATH= Path("data/chunks_checkpoint.parquet")
MAX_TOKENS=510 #for FinBERT max is 512 tokens, we will leave 2 for [CLS] and [SEP]
CHECKPOINT_EVERY = 500
MIN_CHUNK_CHARS = 50 #ignore chunks that are too short

tokenizer= AutoTokenizer.from_pretrained("ProsusAI/finbert")
chunker=chunkerify('ProsusAI/finbert', chunk_size=MAX_TOKENS)

# ── 1. Ticker resolution from sp500_aliases.csv ───────────────────────────────

def build_alias_lookup(csv_path: Path) -> dict:
    """
    Build {alias_lowercase: {ticker, sector, company}} from the aliases CSV.
    The aliases column is pipe-separated: "Apple|Apple Inc."
    Sorted by descending length so longer (more specific) aliases match first.
    Aliases shorter than 4 chars are excluded to avoid false positives.
    """
    lookup = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta = {
                "ticker":  row["canonical_ticker"],
                "sector":  row["sector"],
                "company": row["official_name"],
            }
            for alias in row["aliases"].split("|"):
                alias_clean = alias.strip().lower()
                if len(alias_clean) >= 4:
                    lookup[alias_clean] = meta
    return dict(sorted(lookup.items(), key=lambda x: -len(x[0])))


def resolve_ticker(company: str, lookup: dict) -> dict | None:
    """Match a company name against the alias lookup. Returns meta or None."""
    company_lower = company.lower().strip()
    if company_lower in lookup:
        return lookup[company_lower]
    for alias, meta in lookup.items():
        if alias in company_lower:
            return meta
    return None


# ── 2. Date and quarter parsing ───────────────────────────────────────────────

def parse_date(event_datetime_text: str) -> str:
    """
    Normalize to YYYY-MM-DD.
    Koyfin format: "Friday, December 30, 2016 10:00 AM"
    """
    if not isinstance(event_datetime_text, str):
        return ""
    try:
        dt = datetime.strptime(event_datetime_text.strip(), "%A, %B %d, %Y %I:%M %p")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        m = re.search(r"\d{4}", event_datetime_text)
        return m.group() if m else ""


def parse_quarter(title: str, date: str) -> str:
    """Extract quarter from title or infer from date. Output format: '2016Q4'"""
    if isinstance(title, str):
        m = re.search(r"Q([1-4])\s+(\d{4})", title)
        if m:
            return f"{m.group(2)}Q{m.group(1)}"
    if date and len(date) >= 7:
        month = int(date[5:7])
        year  = date[:4]
        q     = (month - 1) // 3 + 1
        return f"{year}Q{q}"
    return "unknown"


# ── 3. Strip Event Participants header ────────────────────────────────────────

def strip_header(text: str) -> str:
    """Remove the participant listing block before the first speaker turn."""
    marker = re.search(r"OperatorOperator|\w+Executive\n|\w+Analyst\n", text)
    if marker:
        return text[marker.start():]
    return text


# ── 4. Speaker segmentation (Session 1) ──────────────────────────────────────

SPEAKER_PATTERN = re.compile(
    r"(\w.{0,30}\wExecutive)"
    r"|(\w.{0,30}\wAnalyst)"
    r"|(OperatorOperator)"
    r"|(\w.{0,30}\wShareholder)"
    r"|(\w.{0,30}\wAttendee)"
)

def split_by_speaker(text: str) -> list[tuple[str, str]]:
    matches = list(SPEAKER_PATTERN.finditer(text))
    if not matches:
        return []
    segments = []
    for i, m in enumerate(matches):
        start   = m.end()
        end     = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            segments.append((m.group(), content))
    return segments


# ── 5. Section labeling — cumsum trick (Session 1) ───────────────────────────

def label_sections(segments: list[tuple[str, str]]) -> pd.DataFrame:
    """
    Prepared = management speaking before the first analyst turn
    A        = management speaking after the first analyst turn (answers)
    Q        = analyst questions
    O        = operator / shareholder / other
    """
    df = pd.DataFrame(segments, columns=["speaker", "content"])
    df["qa_started"] = (df["speaker"].str.endswith("Analyst").cumsum() > 1)
    df["section"] = df.apply(
        lambda x:
            "Prepared" if x["speaker"].endswith("Executive") and not x["qa_started"] else
            "A"        if x["speaker"].endswith("Executive") and x["qa_started"] else
            "Q"        if x["speaker"].endswith("Analyst") else "O",
        axis=1
    )
    return df


# ── 6. Text cleaning + boilerplate filter ────────────────────────────────────

OPERATOR_BOILERPLATE = re.compile(
    r"this concludes (?:today'?s|our) conference"
    r"|thank you for (?:participating|joining)"
    r"|you may (?:now )?disconnect"
    r"|there are no further questions"
    r"|next question (?:is|comes) from"
    r"|our (?:next|first) question comes from",
    re.IGNORECASE
)

def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)   # residual HTML tags
    text = re.sub(r"\[.*?\]", "", text)     # [Operator Instructions] etc.
    text = re.sub(r"\s+", " ", text)        # normalize whitespace
    return text.strip()

def is_boilerplate(text: str, section: str) -> bool:
    if len(text) < MIN_CHUNK_CHARS:
        return True
    if section == "O" and OPERATOR_BOILERPLATE.search(text):
        return True
    return False


# ── 7. Semantic chunking <=510 tokens ─────────────────────────────────────────

def chunk_segments(df: pd.DataFrame, meta: dict) -> list[dict]:
    rows = []
    for _, row in df.iterrows():
        content_clean = clean_text(row["content"])
        if is_boilerplate(content_clean, row["section"]):
            continue
        for chunk in chunker(content_clean):
            n_tokens = len(tokenizer.encode(chunk, add_special_tokens=False))
            rows.append({
                "chunk_id": str(uuid.uuid4()),
                "ticker":   meta["ticker"],
                "sector":   meta["sector"],
                "company":  meta["company"],
                "date":     meta["date"],
                "quarter":  meta["quarter"],
                "speaker":  row["speaker"],
                "section":  row["section"],
                "content":  chunk,
                "n_tokens": n_tokens,
            })
    return rows


# ── 8. Validation (Session 1) ─────────────────────────────────────────────────

def validate(raw_text: str, segments: list, df_chunks: pd.DataFrame):
    if not segments or df_chunks.empty:
        return
    segmented_length = sum(len(s) + len(c) for s, c in segments)
    drop_rate = 1 - segmented_length / max(len(raw_text), 1)
    assert drop_rate < 0.05, f"Lost {drop_rate:.0%} of text during segmentation!"
    assert df_chunks["n_tokens"].max() <= MAX_TOKENS, "Chunks exceed token limit!"


# ── 9. Process one JSON record ────────────────────────────────────────────────

def process_record(record: dict, alias_lookup: dict) -> list[dict]:
    company  = record.get("company", "")
    resolved = resolve_ticker(company, alias_lookup)
    if not resolved:
        return []   # not S&P 500 -> skip

    raw_text = record.get("transcript_text", "")
    if not raw_text or len(raw_text) < 100:
        return []

    text     = strip_header(raw_text)
    segments = split_by_speaker(text)
    if not segments:
        return []

    date  = parse_date(record.get("event_datetime_text", ""))
    title = record.get("title", "")
    meta  = {
        "ticker":  resolved["ticker"],
        "sector":  resolved["sector"],
        "company": resolved["company"],
        "date":    date,
        "quarter": parse_quarter(title, date),
    }

    df        = label_sections(segments)
    chunks    = chunk_segments(df, meta)
    df_chunks = pd.DataFrame(chunks)
    if not df_chunks.empty:
        validate(text, segments, df_chunks)
    return chunks


# ── 10. Quality report ────────────────────────────────────────────────────────

def quality_report(df: pd.DataFrame):
    log.info("\n=== Quality Report ===")
    log.info(f"Total chunks        : {len(df)}")
    log.info(f"Unique tickers      : {df['ticker'].nunique()}")
    log.info(f"Unique transcripts  : {df[['ticker','date']].drop_duplicates().shape[0]}")
    log.info(f"\nSection distribution:\n{df['section'].value_counts()}")
    log.info(f"\nToken distribution  :\n{df['n_tokens'].describe().round(1)}")
    log.info(f"\nSectors covered     :\n{df.groupby('sector')['ticker'].nunique().sort_values(ascending=False)}")
    log.info(f"\nTop 10 tickers      :\n{df.groupby('ticker').size().sort_values(ascending=False).head(10)}")
    log.info("======================")


# ── 11. Main ──────────────────────────────────────────────────────────────────

def run(input_path: Path, output_path: Path):
    alias_lookup = build_alias_lookup(ALIASES_CSV)
    log.info(f"Loaded {len(alias_lookup)} aliases for {len(set(v['ticker'] for v in alias_lookup.values()))} tickers")

    files = sorted(input_path.glob("*.jsonl")) + sorted(input_path.glob("*.json")) \
            if input_path.is_dir() else [input_path]
    log.info(f"Found {len(files)} input file(s)")

    results = []
    if CHECKPOINT_PATH.exists():
        done_df = pd.read_parquet(CHECKPOINT_PATH)
        results = done_df.to_dict("records")
        log.info(f"Resuming from checkpoint — {len(results)} chunks already processed")

    processed = failed = skipped = 0

    for filepath in files:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    failed += 1
                    continue

                try:
                    chunks = process_record(record, alias_lookup)
                    if not chunks:
                        skipped += 1
                        continue
                    results.extend(chunks)
                    processed += 1
                except Exception as e:
                    log.warning(f"[SKIP] {record.get('company', '?')} — {e}")
                    failed += 1

                if processed % CHECKPOINT_EVERY == 0 and processed > 0:
                    pd.DataFrame(results).to_parquet(CHECKPOINT_PATH, index=False)
                    log.info(f"Checkpoint — {processed} transcripts, {len(results)} chunks")

    df_out = pd.DataFrame(results)
    df_out.to_parquet(output_path, index=False)
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    log.info(f"\nDone — {len(df_out)} chunks | {df_out['ticker'].nunique()} tickers")
    log.info(f"Processed: {processed} | Skipped (not S&P 500): {skipped} | Failed: {failed}")
    quality_report(df_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/raw",            help="Input folder or .jsonl file")
    parser.add_argument("--output", default="data/chunks.parquet", help="Output .parquet file")
    args = parser.parse_args()
    run(Path(args.input), Path(args.output))