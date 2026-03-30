"""
analysis.py — OLS regression: cautious tone vs. post-call realized volatility.

Model:
  rv_post = α + β1·rv_hist + β2·vix + β3·covid + β4·rate_hike
            + [β5·cautious_share]   ← full model only
            + FE_sector + FE_quarter + ε

Econometric safeguards:
  - No look-ahead bias: rv_post computed strictly after call_date (t+1 to t+21)
  - Crisis dummies: covid (2020Q1-Q3), rate_hike (2022Q1-2023Q2)
  - HC3 heteroskedasticity-robust standard errors
  - VIF check for multicollinearity
  - Winsorization of RV at 1st/99th percentile

Output:
  data/regression_results.txt
  data/regression_results.csv
Usage: python analysis.py
"""

import logging, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

FEATURES_PATH = Path("data/features.parquet")
RESULTS_TXT   = Path("data/regression_results.txt")
RESULTS_CSV   = Path("data/regression_results.csv")

BASE_CONTROLS = "rv_hist + vix + covid + rate_hike + C(sector) + C(quarter)"


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH)
    log.info(f"Loaded {len(df)} observations across {df['ticker'].nunique()} tickers")
    df["sector"]    = df["sector"].astype("category")
    df["quarter"]   = df["quarter"].astype("category")
    df["date_dt"]   = pd.to_datetime(df["date"])
    df["covid"]     = ((df["date_dt"] >= "2020-01-01") & (df["date_dt"] <= "2020-09-30")).astype(int)
    df["rate_hike"] = ((df["date_dt"] >= "2022-01-01") & (df["date_dt"] <= "2023-06-30")).astype(int)
    for col in ["rv_post", "rv_hist"]:
        lo, hi = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lo, hi)
    return df


def check_vif(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["cautious_share", "rv_hist", "vix"]
    X    = df[cols].dropna()
    X    = (X - X.mean()) / X.std()
    vif  = pd.DataFrame({
        "variable": cols,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(len(cols))]
    })
    log.info(f"\nVIF check:\n{vif.to_string(index=False)}")
    return vif


def run_regression(df: pd.DataFrame):
    m1 = smf.ols(f"rv_post ~ {BASE_CONTROLS}",                  data=df).fit(cov_type="HC3")
    m2 = smf.ols(f"rv_post ~ cautious_share + {BASE_CONTROLS}", data=df).fit(cov_type="HC3")
    return m1, m2


def run():
    df      = load_data()
    vif     = check_vif(df)
    m1, m2  = run_regression(df)

    log.info(f"\nBaseline R²: {m1.rsquared:.4f} | Full R²: {m2.rsquared:.4f} | ΔR²: {m2.rsquared - m1.rsquared:.4f}")

    coef = m2.params.get("cautious_share", np.nan)
    ci   = m2.conf_int().loc["cautious_share"]
    pval = m2.pvalues.get("cautious_share", np.nan)
    log.info(
        f"\n=== KEY RESULT ===\n"
        f"A 10 pp increase in cautious_share is associated with a {coef*0.10*100:.3f}% "
        f"change in rv_post over the 20 days following the call.\n"
        f"β = {coef:.4f} | 95% CI [{ci[0]:.4f}, {ci[1]:.4f}] | p = {pval:.4f} | "
        f"Significant at 5%: {'Yes' if pval < 0.05 else 'No'}"
    )

    RESULTS_TXT.write_text("\n".join([
        "=" * 60, "BASELINE MODEL", "=" * 60,
        m1.summary().as_text(), "",
        "=" * 60, "FULL MODEL (+ cautious_share)", "=" * 60,
        m2.summary().as_text(), "",
        "=" * 60, "VIF", "=" * 60,
        vif.to_string(index=False),
    ]))

    pd.DataFrame({
        "variable":    m2.params.index,
        "coefficient": m2.params.values,
        "std_err":     m2.bse.values,
        "pvalue":      m2.pvalues.values,
        "ci_low":      m2.conf_int().iloc[:, 0].values,
        "ci_high":     m2.conf_int().iloc[:, 1].values,
    }).to_csv(RESULTS_CSV, index=False)

    log.info(f"Results → {RESULTS_TXT} | {RESULTS_CSV}")


if __name__ == "__main__":
    run()
