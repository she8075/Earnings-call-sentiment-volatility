import re
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


RAW_INPUT = Path("data/sp500_universe_raw.csv")
OUTPUT = Path("data/reference/sp500_aliases.csv")
WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
RAW_REQUIRED_COLUMNS = ["Symbol", "Security", "GICS Sector"]


CANONICAL_GROUPS = {
    "GOOGL": ["GOOGL", "GOOG"],
    "FOXA": ["FOXA", "FOX"],
    "NWSA": ["NWSA", "NWS"],
}


EXTRA_ALIASES = {
    "GOOGL": {"Alphabet", "Alphabet Inc", "Google", "Google Inc"},
    "META": {"Meta", "Meta Platforms", "Meta Platforms Inc", "Facebook", "Facebook Inc"},
    "XYZ": {"Block", "Block Inc", "Square", "Square Inc"},
    "GE": {"GE", "General Electric", "General Electric Company"},
    "LLY": {"Eli Lilly", "Eli Lilly and Company"},
    "APD": {"Air Products", "Air Products and Chemicals", "Air Products and Chemicals Inc"},
    "DUK": {"Duke Energy", "Duke Energy Corporation"},
    "PEP": {"PepsiCo", "PepsiCo Inc"},
    "MRK": {"Merck", "Merck & Co", "Merck and Co", "Merck & Co Inc"},
    "BF.B": {"Brown-Forman", "Brown Forman", "Brown-Forman Corporation"},
    "BRK.B": {"Berkshire Hathaway", "Berkshire Hathaway Inc"},
    "BK": {"Bank of New York Mellon", "BNY Mellon", "The Bank of New York Mellon Corporation"},
    "ELV": {"Elevance Health", "Elevance Health Inc", "Anthem", "Anthem Inc"},
    "SLB": {"Schlumberger", "SLB", "Schlumberger NV"},
}


def strip_parenthetical(text: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", text).strip()


def derive_aliases(name: str) -> set[str]:
    if not isinstance(name, str) or not name.strip():
        return set()

    variants = set()
    cleaned = name.strip()
    variants.add(cleaned)
    variants.add(strip_parenthetical(cleaned))
    variants.add(re.sub(r",?\s+Inc\.?$", "", cleaned, flags=re.IGNORECASE).strip())
    variants.add(re.sub(r",?\s+Corporation$", "", cleaned, flags=re.IGNORECASE).strip())
    variants.add(re.sub(r"\s+\(The\)$", "", cleaned, flags=re.IGNORECASE).strip())
    variants.add(re.sub(r"^The\s+", "", cleaned, flags=re.IGNORECASE).strip())
    variants.add(cleaned.replace(",", ""))
    return {v for v in variants if v}


def fetch_sp500_universe() -> pd.DataFrame:
    request = Request(
        WIKIPEDIA_SP500_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0 Safari/537.36"
            )
        },
    )
    with urlopen(request) as response:
        html = response.read().decode("utf-8")

    tables = pd.read_html(StringIO(html))
    for table in tables:
        columns = [str(col) for col in table.columns]
        if all(column in columns for column in RAW_REQUIRED_COLUMNS):
            universe = table[RAW_REQUIRED_COLUMNS].copy()
            universe = universe.dropna(subset=["Symbol", "Security", "GICS Sector"])
            universe["Symbol"] = universe["Symbol"].astype(str).str.strip()
            universe["Security"] = universe["Security"].astype(str).str.strip()
            universe["GICS Sector"] = universe["GICS Sector"].astype(str).str.strip()
            universe = universe[universe["Symbol"] != ""]
            return universe.reset_index(drop=True)
    raise ValueError("Impossible de trouver une table S&P 500 valide depuis Wikipedia.")


def load_or_fetch_raw_input() -> pd.DataFrame:
    if RAW_INPUT.exists():
        raw = pd.read_csv(RAW_INPUT)
        if not raw.empty:
            missing_columns = [col for col in RAW_REQUIRED_COLUMNS if col not in raw.columns]
            if missing_columns:
                raise ValueError(
                    f"{RAW_INPUT} existe mais il manque les colonnes {missing_columns}."
                )
            return raw
        print(f"[warn] {RAW_INPUT} est vide, recuperation automatique de la liste S&P 500.")
    else:
        print(f"[warn] {RAW_INPUT} introuvable, recuperation automatique de la liste S&P 500.")

    try:
        raw = fetch_sp500_universe()
    except Exception as exc:
        raise RuntimeError(
            "Impossible de recuperer automatiquement la liste S&P 500. "
            "Wikipedia a probablement bloque la requete. "
            "Remplis data/sp500_universe_raw.csv manuellement avec les colonnes "
            "Symbol, Security, GICS Sector."
        ) from exc
    RAW_INPUT.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_INPUT, index=False)
    print(f"[save] {RAW_INPUT}")
    print(f"[rows] {len(raw)} constituents recuperes")
    return raw


def build_canonical_rows(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["Symbol"] = raw["Symbol"].astype(str)

    grouped_symbols = {ticker for group in CANONICAL_GROUPS.values() for ticker in group}
    rows = []
    used = set()

    for canonical_ticker, members in CANONICAL_GROUPS.items():
        subset = raw[raw["Symbol"].isin(members)].copy()
        if subset.empty:
            continue
        used.update(members)
        canonical_row = subset[subset["Symbol"] == canonical_ticker].iloc[0]
        official_name = strip_parenthetical(str(canonical_row["Security"]))
        aliases = set()
        for _, r in subset.iterrows():
            aliases.update(derive_aliases(str(r["Security"])))
        aliases.update(EXTRA_ALIASES.get(canonical_ticker, set()))
        rows.append(
            {
                "canonical_ticker": canonical_ticker,
                "sector": canonical_row["GICS Sector"],
                "official_name": official_name,
                "member_tickers": "|".join(members),
                "aliases": "|".join(sorted(aliases)),
            }
        )

    for _, r in raw.iterrows():
        symbol = r["Symbol"]
        if symbol in used:
            continue
        official_name = strip_parenthetical(str(r["Security"]))
        aliases = set()
        aliases.update(derive_aliases(str(r["Security"])))
        aliases.update(EXTRA_ALIASES.get(symbol, set()))
        rows.append(
            {
                "canonical_ticker": symbol,
                "sector": r["GICS Sector"],
                "official_name": official_name,
                "member_tickers": symbol,
                "aliases": "|".join(sorted(aliases)),
            }
        )

    if not rows:
        raise ValueError(
            "Aucune ligne exploitable dans la liste S&P 500. "
            "Verifie le CSV brut ou la recuperation automatique."
        )

    df = pd.DataFrame(rows).sort_values("canonical_ticker").reset_index(drop=True)
    return df


def main():
    raw = load_or_fetch_raw_input()
    aliases = build_canonical_rows(raw)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    aliases.to_csv(OUTPUT, index=False)
    print(f"[save] {OUTPUT}")
    print(f"[rows] {len(aliases)} canonical firms")


if __name__ == "__main__":
    main()
