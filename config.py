import os
from pathlib import Path
from calendar import monthrange

TICKER_UNIVERSE = {
    "Information Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "CRM"],
    "Health Care":            ["LLY",  "UNH",  "JNJ",  "ABBV", "MRK"],
    "Financials":             ["JPM",  "V",    "MA",   "BAC",  "WFC"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD",   "MCD",  "NKE"],
    "Communication Services": ["META", "GOOGL","NFLX", "DIS",  "CMCSA"],
    "Industrials":            ["GE",   "CAT",  "RTX",  "HON",  "UPS"],
    "Consumer Staples":       ["WMT",  "PG",   "KO",   "PEP",  "COST"],
    "Energy":                 ["XOM",  "CVX",  "COP",  "EOG",  "SLB"],
    "Materials":              ["LIN",  "APD",  "SHW",  "ECL",  "NEM"],
    "Real Estate":            ["PLD",  "AMT",  "EQIX", "SPG",  "CCI"],
    "Utilities":              ["NEE",  "DUK",  "SO",   "AEP",  "EXC"],
}

ALL_TICKERS = [t for tickers in TICKER_UNIVERSE.values() for t in tickers]

SCRAPE_YEAR           = 2021
START_DATE            = "01/01/2021"
END_DATE              = "12/31/2021"
AUTO_SPLIT_BY_MONTH   = True
DEFAULT_SECURITY_LIST = "S&P 500 (US Core)"
OUTPUT_FILENAME       = f"koyfin_transcripts_{SCRAPE_YEAR}.jsonl"

KOYFIN_EMAIL    = os.environ.get("KOYFIN_EMAIL", "")
KOYFIN_PASSWORD = os.environ.get("KOYFIN_PASSWORD", "")

SCROLL_PAUSE    = 1.5
REQUEST_PAUSE   = 3.0
SESSION_REFRESH = 50

RAW_DIR    = Path("data/raw")
LOG_FILE   = "logs/scraper.log"
CHECKPOINT = "logs/scraped_log.csv"


def month_windows_for_year(year: int) -> list[tuple[str, str]]:
    return [
        (f"{m:02d}/01/{year}", f"{m:02d}/{monthrange(year, m)[1]:02d}/{year}")
        for m in range(1, 13)
    ]
