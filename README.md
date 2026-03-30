# Earnings Call Sentiment & Realized Volatility

**Research question:** Does the cautious tone of management's prepared remarks in S&P 500 earnings calls predict realized stock volatility in the 20 trading days following the call, beyond historical volatility and the VIX?

**Result:** Null result — cautious tone does not predict post-call realized volatility (β = 0.0007, p = 0.945, ΔR² ≈ 0). Consistent with semi-strong market efficiency: tone in scripted prepared remarks is efficiently priced by market participants.

---

*Master 2 Finance Technology Data — Paris 1 Panthéon-Sorbonne*  
*Applied Big Data Analytics in Finance, Thomas Rigou — March 2026*  
*Rodrigue Mieuzet · Shéhérazade Bouziani · Mourad Sylla · Nam Khanh Nguyen*

---

## Pipeline

```
data/raw/*.jsonl             (113,990 Koyfin transcripts)
        ↓  preprocessing.py
data/chunks.parquet          (952,520 chunks, 478 tickers)
        ↓  labeler.py  +  financial_phrasebank (HuggingFace)
data/labeled_chunks.parquet  (4,120 labeled examples)
        ↓  finetune_colab.ipynb  (Google Colab T4 GPU, ~20 min)
models/finbert_tone/         (fine-tuned FinBERT, 86% accuracy, F1-macro 0.86)
        ↓  inference.py
data/tone_scores.parquet     (15,039 call-level tone scores)
        ↓  build_features.py
data/features.parquet        (10,226 observations, 457 tickers)
        ↓  analysis.py
data/regression_results.txt
```

---

## Repository Structure

```
├── scraper.py                 Selenium scraper for Koyfin transcripts
├── driver.py                  Chrome WebDriver factory
├── config.py                  Scraper settings (credentials via env vars)
├── build_sp500_aliases.py     Build canonical S&P 500 alias table
├── preprocessing.py           JSONL → chunks.parquet
├── labeler.py                 LLM labeling via Groq API (Llama-3.1-8B)
├── finetuner.py               FinBERT fine-tuning (local CPU/GPU)
├── finetune_colab.ipynb       FinBERT fine-tuning (Google Colab T4 — recommended)
├── inference.py               Batch inference → tone_scores.parquet
├── inference.py               Batch inference → tone_scores.parquet (Google Colab T4 - recommended)
├── build_features.py          Financial variables via yfinance → features.parquet
├── analysis.py                OLS regression with fixed effects
├── requirements.txt
├── .gitignore
└── data/
    ├── sp500_aliases.csv      Canonical ticker → alias mapping (500 firms, 11 sectors)
    └── ...                    Intermediate parquet files (see Google Drive for raw)
```

---

## Setup

```bash
pip install -r requirements.txt
```

Set environment variables (never hardcode credentials):
```bash
export KOYFIN_EMAIL=your_email@example.com
export KOYFIN_PASSWORD=your_password
export GROQ_API_KEY=gsk_...            # only needed for labeler.py
```

---

## Reproducing the Pipeline

### 1. Build S&P 500 alias table
```bash
python build_sp500_aliases.py
```
Fetches current S&P 500 constituents from Wikipedia and writes `data/sp500_aliases.csv`.

### 2. Scrape transcripts
```bash
python scraper.py --start 01/01/2021 --end 12/31/2021 --by-month
```
Outputs `data/raw/koyfin_transcripts_2021.jsonl`. Repeat for each year 2016–2024.  
Raw transcripts (~5GB) were sourced via [Koyfin](https://www.koyfin.com) and cannot be redistributed. Please refer to Koyfin directly to access earnings call transcripts.

### 3. Preprocess
```bash
python preprocessing.py --input data/raw --output data/chunks.parquet
```
Produces 952,520 chunks across 478 tickers. Runtime ~2h on CPU.

### 4. Build training dataset

Combine [financial_phrasebank](https://huggingface.co/datasets/atrost/financial_phrasebank) with LLM-generated labels:
```bash
# Generate LLM labels via Groq
export GROQ_API_KEY=gsk_...
python labeler.py

# Merge both sources
python assemble_finbert_dataset.py --manifest data/manifest.json --output data/labeled_chunks.parquet
```
Or download pre-built `labeled_chunks.parquet` from Google Drive.

### 5. Fine-tune FinBERT
**Recommended:** open `finetune_colab.ipynb` on [Google Colab](https://colab.research.google.com) with T4 GPU (~20 min).

Local alternative (slow on CPU):
```bash
python finetuner.py --epochs 3 --batch_size 16
```
Saves model to `models/finbert_tone/`.

### 6. Run inference
**Recommended:** open `inference_colab.ipynb` on [Google Colab](https://colab.research.google.com) with T4 GPU (~1h).

Local alternative (slow on CPU):
```bash
python inference.py --batch_size 32
```
Outputs `data/tone_scores.parquet` (15,039 call-level scores).

### 7. Build financial features
```bash
python build_features.py
```
Downloads stock prices via yfinance (cached after first run in `data/prices_cache.parquet`).

### 8. Run regression
```bash
python analysis.py
```
Outputs `data/regression_results.txt` and `data/regression_results.csv`.

---

## Key Results

| Metric | Value |
|---|---|
| Transcripts scraped | 113,990 |
| S&P 500 transcripts | 14,963 |
| Chunks (total) | 952,520 |
| Chunks (Prepared Remarks) | 183,491 |
| Training examples | 4,120 |
| FinBERT accuracy | 86% |
| FinBERT F1-macro | 0.86 |
| Regression observations | 10,226 |
| Baseline R² | 0.445 |
| β (cautious_share) | 0.0007 |
| p-value | 0.945 |
| COVID dummy | +8.2 pp (p < 0.001) |
| Rate hike dummy | +2.6 pp (p < 0.001) |

---

## Model

```
ProsusAI/finbert backbone (BERT, 12 layers, 768-dim)
    → Linear(768 → 3)
    → Softmax
    → CrossEntropy loss

Labels : optimistic (0) | cautious (1) | negative (2)
Training: AdamW lr=2e-5, 3 epochs, batch_size=16
Split   : 80/10/10 stratified by sector × year
Hardware: Google Colab T4 GPU
```

**cautious_share** per call = mean P(cautious) across all Prepared Remarks chunks.

**OLS specification:**
```
rv_post = α + β1·rv_hist + β2·vix + β3·covid + β4·rate_hike
            + [β5·cautious_share]   ← full model only
            + FE_sector + FE_quarter + ε
```
HC3-robust standard errors. Winsorization at 1st/99th percentile.

---

## External Data

Raw transcripts (~5GB) were sourced via [Koyfin](https://www.koyfin.com) and cannot be redistributed. If you wish to reproduce this study, please refer to Koyfin directly to obtain earnings call transcripts.

Files tracked in this repository:
- `data/sp500_aliases.csv` — S&P 500 canonical alias table
- `models/finbert_tone/` — fine-tuned model weights (438 MB, hosted on HuggingFace: [She8075/Model.safetensors](https://huggingface.co/She8075/Model.safetensors))
- All intermediate `.parquet` files except raw transcripts and price cache

---

## References

- Price, Doran, Peterson & Bliss (2012). *Earnings conference calls and stock returns.* Journal of Banking & Finance.
- Hassan, Hollander, van Lent & Tahoun (2019). *Firm-level political risk.* Quarterly Journal of Economics.
- Malo et al. (2014). *Good debt or bad debt: Detecting semantic orientations in economic texts.* JASIST.
- Araci (2019). *FinBERT: Financial sentiment analysis with pre-trained language models.* arXiv:1908.10063.
