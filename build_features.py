"""
build_features.py — Financial variables for OLS regression.

For each (ticker, call_date) in tone_scores.parquet:
  rv_post : annualized realized vol, t+1 to t+21 after call (no look-ahead bias)
  rv_hist : annualized realized vol, 60 trading days before call
  vix     : VIX closing level on call date

Output: data/features.parquet
Usage:  python build_features.py
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

TONE_SCORES_PATH = Path("data/tone_scores.parquet")
OUTPUT_PATH      = Path("data/features.parquet")
PRICE_CACHE_PATH = Path("data/prices_cache.parquet")

RV_POST_DAYS = 20
RV_HIST_DAYS = 60
ANNUALIZE    = np.sqrt(252)


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    if PRICE_CACHE_PATH.exists():
        log.info("Loading prices from cache...")
        return pd.read_parquet(PRICE_CACHE_PATH)
    log.info(f"Downloading prices for {len(tickers)} tickers ({start} to {end})...")
    raw    = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=True)
    prices = raw["Close"] if "Close" in raw.columns else raw.xs("Close", axis=1, level=0)
    prices = prices.dropna(axis=1, thresh=int(0.7 * len(prices)))
    prices.to_parquet(PRICE_CACHE_PATH)
    log.info(f"Prices cached — {prices.shape[1]} tickers, {len(prices)} days")
    return prices


def fetch_vix(start: str, end: str) -> pd.Series:
    log.info("Downloading VIX...")
    vix   = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
    close = vix["Close"].squeeze()
    return close.rename("vix")


def realized_vol(series: pd.Series, date: pd.Timestamp, offset: int, window: int) -> float:
    prices = series.dropna()
    try:
        idx = prices.index.get_indexer([date], method="ffill")[0]
        s, e = idx + offset, idx + offset + window + 1
        if s < 0 or e > len(prices):
            return np.nan
        log_ret = np.log(prices.iloc[s:e] / prices.iloc[s:e].shift(1)).dropna()
        return float(log_ret.std() * ANNUALIZE) if len(log_ret) >= window // 2 else np.nan
    except Exception:
        return np.nan


def build_features(tone_df: pd.DataFrame, prices: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    records = []
    for i, row in enumerate(tone_df.itertuples(), 1):
        if row.ticker not in prices.columns:
            continue
        try:
            date = pd.Timestamp(row.date)
        except Exception:
            continue
        vix_idx = vix.index.get_indexer([date], method="ffill")[0]
        records.append({
            "ticker":           row.ticker,
            "sector":           row.sector,
            "company":          row.company,
            "date":             row.date,
            "quarter":          row.quarter,
            "cautious_share":   row.cautious_share,
            "optimistic_share": row.optimistic_share,
            "negative_share":   row.negative_share,
            "rv_post": realized_vol(prices[row.ticker], date, offset=1,             window=RV_POST_DAYS),
            "rv_hist": realized_vol(prices[row.ticker], date, offset=-RV_HIST_DAYS, window=RV_HIST_DAYS),
            "vix":     float(vix.iloc[vix_idx]) if vix_idx >= 0 else np.nan,
        })
        if i % 500 == 0:
            log.info(f"  {i}/{len(tone_df)} calls processed")

    df = pd.DataFrame(records).dropna(subset=["rv_post", "rv_hist"])
    assert (df["rv_post"] > 0).all() and (df["rv_post"] < 5).all(), "RV_post out of range"
    log.info("Validation passed.")
    return df


def run():
    tone_df = pd.read_parquet(TONE_SCORES_PATH)
    log.info(f"Loaded {len(tone_df)} call-level tone scores")
    prices  = fetch_prices(tone_df["ticker"].unique().tolist(), "2018-01-01", "2025-01-01")
    vix     = fetch_vix("2018-01-01", "2025-01-01")
    df      = build_features(tone_df, prices, vix)
    df.to_parquet(OUTPUT_PATH, index=False)
    log.info(f"\nDone — {len(df)} observations → {OUTPUT_PATH}")
    log.info(f"\nFeature summary:\n{df[['rv_post','rv_hist','vix','cautious_share']].describe().round(4)}")
    log.info(f"\nMissing values:\n{df.isnull().sum()}")


if __name__ == "__main__":
    run()
