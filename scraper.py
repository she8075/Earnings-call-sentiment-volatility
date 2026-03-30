"""
scraper.py — Selenium scraper for Koyfin earnings call transcripts.

Logs into Koyfin, applies Earnings Calls + date range + S&P 500 filters,
scrolls through results, and saves each transcript as a JSON line.

Supports:
  - Monthly window splitting (--by-month) to avoid scroll caps
  - Resume from existing JSONL output (deduplication by title)
  - Universe filtering (--universe) to restrict to S&P 500 tickers
  - Missing-only mode (--missing-only) to fill gaps in an existing corpus

Setup:
    export KOYFIN_EMAIL=...
    export KOYFIN_PASSWORD=...
    python scraper.py --start 01/01/2021 --end 12/31/2021 --by-month

Output: data/raw/koyfin_transcripts_YYYY.jsonl
"""

import argparse
import calendar
import csv
import json
import time
from datetime import datetime

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException

from driver import make_driver
from config import (
    AUTO_SPLIT_BY_MONTH, DEFAULT_SECURITY_LIST,
    END_DATE, KOYFIN_EMAIL, KOYFIN_PASSWORD,
    OUTPUT_FILENAME, RAW_DIR, START_DATE,
)


def safe_click(driver, element):
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", element)
    time.sleep(0.05)
    try:
        element.click()
    except Exception:
        driver.execute_script("arguments[0].click();", element)


def parse_lines(text):
    return [x.strip() for x in text.splitlines() if x.strip()]


def extract_company_from_title(title):
    parts = title.split(",")
    return parts[0].strip() if len(parts) >= 2 else title.strip()


def is_earnings_call_title(title):
    return "earnings call" in " ".join((title or "").lower().split())


def normalize_name(value):
    if not isinstance(value, str):
        return ""
    value = value.lower().replace("&", " and ")
    for token in ["incorporated", "corporation", "company", "companies", "inc", "corp",
                  "co", "plc", "ltd", "limited", "group", "holdings", "nv", "sa", "se"]:
        value = value.replace(token, " ")
    value = "".join(ch if ch.isalnum() else " " for ch in value)
    return " ".join(value.split())


def load_universe_alias_map(path):
    if not path:
        return {}
    alias_map = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ticker = (row.get("ticker") or "").strip()
            if not ticker:
                continue
            normalized_aliases = set()
            for alias_group in [row.get("company", ""), row.get("aliases", "")]:
                for alias in str(alias_group).split("|"):
                    n = normalize_name(alias)
                    if n:
                        normalized_aliases.add(n)
            alias_map[ticker] = {
                "ticker":  ticker,
                "sector":  (row.get("sector") or "").strip(),
                "company": (row.get("company") or "").strip(),
                "aliases": normalized_aliases,
            }
    return alias_map


def match_company_to_universe(company, alias_map):
    n = normalize_name(company)
    if not n:
        return None
    for item in alias_map.values():
        for alias in item["aliases"]:
            if n == alias or n.startswith(alias) or alias.startswith(n):
                return item
    return None


def load_found_tickers_from_jsonl(path, alias_map):
    found = set()
    if not path or not alias_map:
        return found
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            company = (obj.get("company") or "").strip() or extract_company_from_title(obj.get("title") or "")
            match = match_company_to_universe(company, alias_map)
            if match:
                found.add(match["ticker"])
    return found


def extract_metadata_from_transcript(title, transcript_text):
    lines = parse_lines(transcript_text)
    event_datetime_text = ""
    for line in lines[:10]:
        if any(d in line.lower() for d in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]):
            event_datetime_text = line.strip()
            break
    return {
        "title":                title,
        "company":              extract_company_from_title(title),
        "event_type":           "Earnings Calls",
        "event_datetime_text":  event_datetime_text,
        "scraped_at":           datetime.now().isoformat(timespec="seconds"),
        "transcript_text":      transcript_text,
    }


def set_input_value_native(driver, element, value):
    driver.execute_script("""
        const input = arguments[0];
        const value = arguments[1];
        const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
        nativeSetter.call(input, value);
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
        input.dispatchEvent(new Event('blur', { bubbles: true }));
    """, element, value)


def get_visible_date_inputs(driver):
    inputs = driver.find_elements(By.XPATH, "//input[@placeholder='MM/DD/YYYY' and @type='text']")
    return [i for i in inputs if i.is_displayed() and i.is_enabled()]


def get_input_value(element):
    return (element.get_attribute("value") or "").strip()


def parse_us_date(date_str):
    return datetime.strptime(date_str, "%m/%d/%Y").date()


def format_us_date(value):
    return value.strftime("%m/%d/%Y")


def month_ranges_between(start_date_str, end_date_str):
    start_date = parse_us_date(start_date_str)
    end_date   = parse_us_date(end_date_str)
    if start_date > end_date:
        raise ValueError(f"Invalid date range: {start_date_str} > {end_date_str}")
    windows, year, month = [], start_date.year, start_date.month
    while (year, month) <= (end_date.year, end_date.month):
        month_start = start_date if (year, month) == (start_date.year, start_date.month) \
                      else start_date.replace(year=year, month=month, day=1)
        month_last  = calendar.monthrange(year, month)[1]
        month_end   = end_date if (year, month) == (end_date.year, end_date.month) \
                      else end_date.replace(year=year, month=month, day=month_last)
        windows.append((format_us_date(month_start), format_us_date(month_end)))
        month = month % 12 + 1
        if month == 1:
            year += 1
    return windows


def login_koyfin(driver):
    wait = WebDriverWait(driver, 20)
    driver.get("https://app.koyfin.com/search/transcripts")
    try:
        cookie = WebDriverWait(driver, 8).until(EC.element_to_be_clickable((By.XPATH, "//button[contains(.,'Accept')]")))
        cookie.click()
    except TimeoutException:
        pass
    safe_click(driver, wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(.,'Log In')]"))))
    email = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='email']")))
    email.clear(); email.send_keys(KOYFIN_EMAIL)
    password = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='password']")))
    password.clear(); password.send_keys(KOYFIN_PASSWORD)
    safe_click(driver, wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(.,'Sign in')]"))))
    print("Logged in")
    time.sleep(2)


def go_to_transcripts(driver):
    wait = WebDriverWait(driver, 20)
    safe_click(driver, wait.until(EC.element_to_be_clickable((By.XPATH, "//span[contains(.,'Advanced Search')]"))))
    time.sleep(0.3)
    safe_click(driver, wait.until(EC.element_to_be_clickable((By.XPATH, "//span[contains(.,'Transcripts Search')]"))))
    wait.until(EC.presence_of_element_located((By.XPATH, "//*[contains(.,'Search Transcripts')]")))
    time.sleep(0.3)
    print("Transcript Search page loaded")


def type_date_into_input(driver, element, date_str):
    safe_click(driver, element)
    time.sleep(0.2)
    element.send_keys(Keys.COMMAND + "a")
    time.sleep(0.05)
    element.send_keys(Keys.BACKSPACE)
    time.sleep(0.1)
    for ch in date_str.replace("/", ""):
        element.send_keys(ch)
        time.sleep(0.05)
    time.sleep(0.2)
    element.send_keys(Keys.TAB)
    time.sleep(0.2)


def set_date_range(driver, start_date, end_date):
    wait = WebDriverWait(driver, 20)
    date_box = wait.until(EC.element_to_be_clickable(
        (By.XPATH, "//div[contains(@class,'time-range__root') and contains(., '-')]")
    ))
    safe_click(driver, date_box)
    time.sleep(1.0)

    visible = get_visible_date_inputs(driver)
    if len(visible) < 2:
        raise Exception(f"Expected 2 date inputs, found {len(visible)}")

    type_date_into_input(driver, visible[0], start_date)
    time.sleep(0.3)
    visible = get_visible_date_inputs(driver)
    type_date_into_input(driver, visible[1], end_date)
    time.sleep(0.3)

    actual_start = get_input_value(visible[0])
    actual_end   = get_input_value(visible[1])
    if actual_start != start_date or actual_end != end_date:
        print("Values mismatch — trying native setter fallback...")
        set_input_value_native(driver, visible[0], start_date)
        time.sleep(0.2)
        visible = get_visible_date_inputs(driver)
        set_input_value_native(driver, visible[1], end_date)
        time.sleep(0.2)
        actual_start = get_input_value(visible[0])
        actual_end   = get_input_value(visible[1])

    if actual_start != start_date or actual_end != end_date:
        raise Exception(f"Date range not applied: expected {start_date} → {end_date}, got {actual_start} → {actual_end}")

    try:
        apply_btn = WebDriverWait(driver, 5).until(EC.element_to_be_clickable(
            (By.XPATH, "//button[.//label[contains(text(),'Apply Dates')]]")
        ))
        safe_click(driver, apply_btn)
    except Exception:
        try:
            apply_btn = WebDriverWait(driver, 3).until(EC.element_to_be_clickable(
                (By.XPATH, "//*[contains(text(),'Apply Dates')]")
            ))
            safe_click(driver, apply_btn)
        except Exception as e:
            print(f"WARNING: Could not click Apply Dates: {e}")

    time.sleep(0.8)
    print(f"Set date range: {actual_start} → {actual_end}")


def xpath_literal(s):
    if "'" not in s:
        return f"'{s}'"
    if '"' not in s:
        return f'"{s}"'
    parts = s.split("'")
    return "concat(" + ", \"'\", ".join([f"'{p}'" for p in parts]) + ")"


def set_security_list_filter(driver, security_list_name):
    if not security_list_name:
        return
    wait = WebDriverWait(driver, 20)
    try:
        for el in driver.find_elements(By.XPATH, f"//*[contains(normalize-space(.), {xpath_literal(security_list_name)})]"):
            if el.is_displayed():
                print(f"Security list already visible: {security_list_name}")
                return
    except Exception:
        pass

    selector = None
    for el in driver.find_elements(By.XPATH,
        "//div[contains(.,'Security Lists')]//button | "
        "//div[contains(.,'Security Lists')]//*[@role='button'] | "
        "//div[contains(.,'Security Lists')]//input"
    ):
        try:
            if el.is_displayed() and el.is_enabled():
                selector = el
                break
        except Exception:
            continue
    if selector is None:
        raise Exception("Could not find Security Lists selector")

    safe_click(driver, selector)
    time.sleep(0.5)
    option = wait.until(EC.element_to_be_clickable((
        By.XPATH,
        f"//*[self::div or self::span or self::button][contains(normalize-space(.), {xpath_literal(security_list_name)})]"
    )))
    safe_click(driver, option)
    time.sleep(0.5)
    print(f"Applied security list: {security_list_name}")


def apply_filters_and_search(driver, start_date=None, end_date=None, security_list_name=None):
    wait = WebDriverWait(driver, 20)
    for box in driver.find_elements(By.XPATH, "//input[@type='checkbox' and @checked]"):
        try:
            if box.is_displayed():
                safe_click(driver, box)
                time.sleep(0.1)
        except Exception:
            continue
    time.sleep(0.2)

    earnings_checkbox = None
    for lab in driver.find_elements(By.XPATH, "//label[normalize-space()='Earnings Calls']"):
        try:
            if lab.is_displayed():
                try:
                    earnings_checkbox = lab.find_element(By.XPATH, ".//input[@type='checkbox']")
                except Exception:
                    earnings_checkbox = lab
                break
        except Exception:
            continue
    if earnings_checkbox is None:
        raise Exception("Could not find Earnings Calls checkbox")
    safe_click(driver, earnings_checkbox)
    print("Clicked Earnings Calls filter")
    time.sleep(0.3)

    if security_list_name:
        set_security_list_filter(driver, security_list_name)
    if start_date and end_date:
        set_date_range(driver, start_date, end_date)

    visible_search = [el for el in driver.find_elements(By.XPATH, "//*[contains(.,'Search Transcripts')]") if el.is_displayed()]
    if not visible_search:
        raise Exception("Could not find Search Transcripts button")
    safe_click(driver, visible_search[-1])
    print("Clicked Search Transcripts")

    wait.until(EC.presence_of_element_located((By.XPATH, "//*[contains(.,'Showing most recent results')]")))
    time.sleep(0.5)
    print("Results loaded")


def extract_visible_earnings_titles(driver):
    items = driver.find_elements(By.CSS_SELECTOR, "div.koy-news-item__koyNewsItem___StpWe")
    visible_rows, seen = [], set()
    for item in items:
        try:
            if not item.is_displayed():
                continue
            labels = item.find_elements(By.CSS_SELECTOR, "label.text-label__textLabelContainer___kNkG9")
            if not labels:
                continue
            title = labels[0].text.strip()
            if not title or title in seen or not is_earnings_call_title(title):
                continue
            seen.add(title)
            visible_rows.append((item.rect.get("y", 0), title))
        except (StaleElementReferenceException, Exception):
            continue
    visible_rows.sort(key=lambda r: r[0])
    return [title for _, title in visible_rows]


def click_earnings_row_by_title(driver, title, timeout=15):
    wait  = WebDriverWait(driver, timeout)
    xpath = (
        f"//div[contains(@class,'koy-news-item__koyNewsItem')]"
        f"[.//label[normalize-space()={xpath_literal(title)}]]"
    )
    safe_click(driver, wait.until(EC.element_to_be_clickable((By.XPATH, xpath))))


def scroll_results_panel(driver, pause=0.3):
    scrolled = driver.execute_script("""
        const divs = document.querySelectorAll('div');
        let best = null, bestScore = -1;
        for (const div of divs) {
            const sh = div.scrollHeight, ch = div.clientHeight;
            if (sh > ch && ch > 200) {
                const score = sh * ch;
                if (score > bestScore) { bestScore = score; best = div; }
            }
        }
        if (best) { best.scrollTop += 600; return true; }
        return false;
    """)
    time.sleep(pause)
    return bool(scrolled)


def wait_for_transcript_ready(driver, timeout=20):
    end_time = time.time() + timeout
    while time.time() < end_time:
        try:
            body = driver.find_element(By.TAG_NAME, "body").text
            if len(body) > 500 and "Loading" not in body[:100]:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutException("Transcript did not load in time")


def extract_transcript_text(driver, title, timeout=15):
    wait = WebDriverWait(driver, timeout)
    try:
        article = wait.until(EC.presence_of_element_located(
            (By.XPATH, "//article | //div[contains(@class,'transcript')]")
        ))
        return article.text.strip()
    except TimeoutException:
        time.sleep(0.8)
        return driver.find_element(By.TAG_NAME, "body").text.strip()


def dismiss_results_setup_overlay(driver):
    try:
        btn = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, "//button[contains(.,'Get Started')]")))
        safe_click(driver, btn)
        time.sleep(0.8)
    except Exception:
        pass


def wait_for_results_list_ready(driver, timeout=20):
    dismiss_results_setup_overlay(driver)
    end_time = time.time() + timeout
    while time.time() < end_time:
        try:
            body_text = driver.find_element(By.TAG_NAME, "body").text
        except Exception:
            body_text = ""
        if "Please wait while we set up this page for you..." in body_text:
            dismiss_results_setup_overlay(driver)
            time.sleep(1.0)
            continue
        if extract_visible_earnings_titles(driver):
            return
        time.sleep(1.0)
    raise TimeoutException("Results list did not become ready")


def close_article_panel(driver):
    try:
        btn = WebDriverWait(driver, 5).until(EC.element_to_be_clickable(
            (By.XPATH, "//button[contains(@class,'base-button') and .//i[contains(@class,'fa-times')]]")
        ))
        safe_click(driver, btn)
        time.sleep(0.2)
    except Exception:
        pass


def get_output_path(filename="koyfin_transcripts.jsonl"):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR / filename


def load_existing_titles_jsonl(path):
    existing = set()
    if not path.exists():
        return existing
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                title = json.loads(line).get("title", "").strip()
                if title:
                    existing.add(title)
            except Exception:
                continue
    print(f"Loaded {len(existing)} already-saved titles from {path}")
    return existing


def append_transcript_jsonl(record, filename="koyfin_transcripts.jsonl"):
    path = get_output_path(filename)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved: {record['title']}")


def scrape_transcripts_in_results(driver, output_filename="koyfin_transcripts.jsonl",
                                   max_new_transcripts=None, max_empty_scrolls=8,
                                   alias_map=None, allowed_tickers=None):
    output_path  = get_output_path(output_filename)
    saved_titles = load_existing_titles_jsonl(output_path)
    scraped, no_new_streak, pending = 0, 0, []

    while True:
        if not pending:
            titles  = extract_visible_earnings_titles(driver)
            if not titles:
                print("No visible titles")
                break
            pending = [t for t in titles if t not in saved_titles]

        if not pending:
            found_new = False
            for _ in range(3):
                scroll_results_panel(driver, pause=0.3)
                fresh = [t for t in extract_visible_earnings_titles(driver) if t not in saved_titles]
                if fresh:
                    pending        = fresh
                    no_new_streak  = 0
                    found_new      = True
                    break
            if not found_new:
                no_new_streak += 1
                print(f"No new titles after scroll burst: {no_new_streak}/{max_empty_scrolls}")
                if no_new_streak >= max_empty_scrolls:
                    print("Scroll blocked — closing article panel to unlock...")
                    close_article_panel(driver)
                    time.sleep(1.0)
                    no_new_streak = 0
                    if not scroll_results_panel(driver, pause=0.5):
                        print("Still blocked — stopping.")
                        break
            continue

        no_new_streak = 0
        title = pending.pop(0)
        try:
            if not is_earnings_call_title(title):
                saved_titles.add(title)
                continue
            company = extract_company_from_title(title)
            match   = match_company_to_universe(company, alias_map or {})
            if alias_map and not match:
                saved_titles.add(title)
                continue
            if allowed_tickers is not None and match and match["ticker"] not in allowed_tickers:
                saved_titles.add(title)
                continue
            click_earnings_row_by_title(driver, title)
            wait_for_transcript_ready(driver)
            transcript = extract_transcript_text(driver, title)
            record     = extract_metadata_from_transcript(title, transcript)
            if match:
                record["ticker"] = match["ticker"]
                record["sector"] = match["sector"]
            append_transcript_jsonl(record, filename=output_filename)
            saved_titles.add(title)
            scraped += 1
            print(f"[{scraped}] Scraped: {title}")
            if max_new_transcripts and scraped >= max_new_transcripts:
                break
        except Exception as e:
            print(f"FAILED: {title} — {e}")
            saved_titles.add(title)

    print(f"Done. Scraped {scraped} new transcripts.")


def run_single_search_window(driver, start_date, end_date, output_filename,
                              alias_map=None, allowed_tickers=None, security_list_name=None):
    print(f"Running search window: {start_date} → {end_date}")
    apply_filters_and_search(driver, start_date=start_date, end_date=end_date, security_list_name=security_list_name)
    wait_for_results_list_ready(driver)
    scrape_transcripts_in_results(driver, output_filename=output_filename,
                                   alias_map=alias_map, allowed_tickers=allowed_tickers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",          default=START_DATE)
    parser.add_argument("--end",            default=END_DATE)
    parser.add_argument("--output",         default=OUTPUT_FILENAME)
    parser.add_argument("--headless",       action="store_true")
    parser.add_argument("--security-list",  default=DEFAULT_SECURITY_LIST)
    parser.add_argument("--universe",       help="CSV file with ticker, sector, company, aliases")
    parser.add_argument("--existing-corpus",help="Existing JSONL corpus for coverage check")
    parser.add_argument("--missing-only",   action="store_true",
                        help="Only scrape tickers not yet in --existing-corpus")
    parser.add_argument("--by-month",       action="store_true", default=AUTO_SPLIT_BY_MONTH)
    args = parser.parse_args()

    alias_map      = load_universe_alias_map(args.universe) if args.universe else {}
    allowed_tickers = None
    if args.missing_only:
        if not args.universe or not args.existing_corpus:
            raise ValueError("--missing-only requires both --universe and --existing-corpus")
        found         = load_found_tickers_from_jsonl(args.existing_corpus, alias_map)
        allowed_tickers = set(alias_map.keys()) - found
        print(f"Universe: {len(alias_map)} | Covered: {len(found)} | Missing: {len(allowed_tickers)}")

    driver = make_driver(headless=args.headless)
    try:
        login_koyfin(driver)
        go_to_transcripts(driver)
        if args.by_month:
            windows = month_ranges_between(args.start, args.end)
            print(f"Monthly mode: {len(windows)} windows")
            for idx, (s, e) in enumerate(windows, 1):
                print(f"[{idx}/{len(windows)}] {s} → {e}")
                run_single_search_window(driver, s, e, args.output,
                                         alias_map=alias_map, allowed_tickers=allowed_tickers,
                                         security_list_name=args.security_list)
        else:
            run_single_search_window(driver, args.start, args.end, args.output,
                                     alias_map=alias_map, allowed_tickers=allowed_tickers,
                                     security_list_name=args.security_list)
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
