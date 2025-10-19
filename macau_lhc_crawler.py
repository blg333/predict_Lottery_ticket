# -*- coding:utf-8 -*-
"""
Crawler for 新澳门六合彩 (Macau Mark Six) draws from https://kj.123720c.com/kj/

- Target: 2025年001期至291期（含）
- Saves CSV to data/lhc_macau/2025.csv with columns:
  期数, 开奖日期, 红球_1..红球_6, 蓝球
- Robust parsing using BeautifulSoup + regex; supports monthly pages and list pages.
- Idempotent: re-runs will update/merge without duplication.

Note: This script accesses the network. The user has authorized access.
"""
from __future__ import annotations

import argparse
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

try:
    import requests  # type: ignore
    from bs4 import BeautifulSoup  # type: ignore
except Exception as e:  # pragma: no cover
    requests = None  # type: ignore
    BeautifulSoup = None  # type: ignore

BASE_URL = "https://kj.123720c.com/kj/"

ISSUE_RE = re.compile(r"(?:(?:第)?\s*)(\d{3})(?:\s*期)?")
DATE_RE = re.compile(r"(20\d{2}[-/年]\s*\d{1,2}[-/月]\s*\d{1,2}日?)")
NUMBER_RE = re.compile(r"\b(\d{1,2})\b")


def fetch(url: str, retries: int = 3, sleep_s: float = 1.0) -> Optional[str]:
    if requests is None:
        raise RuntimeError("requests not installed. Please install requirements.")
    for i in range(retries):
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            r.encoding = r.apparent_encoding or "utf-8"
            return r.text
        except Exception as e:
            logger.warning(f"Fetch failed {url}: {e}")
            time.sleep(sleep_s * (i + 1))
    return None


def parse_draws_from_html(html: str) -> List[Dict[str, str]]:
    if BeautifulSoup is None:
        raise RuntimeError("bs4 not installed. Please install requirements.")
    soup = BeautifulSoup(html, "lxml")
    draws: List[Dict[str, str]] = []

    # Heuristic: find rows containing 7 numbers (1..49) and an issue number
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        for tr in rows:
            text = tr.get_text(" ", strip=True)
            # Find issue
            m_issue = ISSUE_RE.search(text)
            if not m_issue:
                continue
            issue = m_issue.group(1)
            # Extract numbers in this row
            nums = [int(x) for x in NUMBER_RE.findall(text) if 1 <= int(x) <= 49]
            # Try to constrain: keep last 7 numbers as draw
            if len(nums) < 7:
                continue
            seven = nums[-7:]
            # Extract date if present
            m_date = DATE_RE.search(text)
            date_str = m_date.group(1) if m_date else ""
            item = {
                "期数": f"{issue}",
                "开奖日期": date_str,
                **{f"红球_{i+1}": f"{seven[i]:02d}" for i in range(6)},
                "蓝球": f"{seven[6]:02d}",
            }
            draws.append(item)
    # Deduplicate by issue
    uniq: Dict[str, Dict[str, str]] = {d["期数"]: d for d in draws}
    return [uniq[k] for k in sorted(uniq.keys())]


def discover_year_pages(year: int) -> List[str]:
    # Try common patterns for monthly/yearly listing
    candidates = [
        f"{BASE_URL}",
        f"{BASE_URL}{year}/",
        f"{BASE_URL}{year}.html",
    ]
    # Additionally, try monthly pages like kj/2025/01.html ... 12.html
    for m in range(1, 13):
        candidates.append(f"{BASE_URL}{year}/{m:02d}.html")
    return candidates


def crawl_year(year: int) -> List[Dict[str, str]]:
    all_draws: Dict[str, Dict[str, str]] = {}
    for url in discover_year_pages(year):
        html = fetch(url)
        if not html:
            continue
        rows = parse_draws_from_html(html)
        for r in rows:
            all_draws[r["期数"]] = r
    result = [all_draws[k] for k in sorted(all_draws.keys())]
    logger.info(f"Discovered {len(result)} issues for {year}")
    return result


def filter_range(draws: List[Dict[str, str]], start_issue: int, end_issue: int) -> List[Dict[str, str]]:
    flt = []
    for d in draws:
        try:
            iss = int(d["期数"])
        except Exception:
            continue
        if start_issue <= iss <= end_issue:
            flt.append(d)
    return sorted(flt, key=lambda x: int(x["期数"]))


def load_existing(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def merge_save(path: str, new_rows: List[Dict[str, str]]) -> pd.DataFrame:
    df_new = pd.DataFrame(new_rows)
    df_old = load_existing(path)
    if not df_old.empty:
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["期数"], keep="last")
        df = df.sort_values(by=["期数"]).reset_index(drop=True)
    else:
        df = df_new.sort_values(by=["期数"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=291)
    parser.add_argument("--out", type=str, default="data/lhc_macau/2025.csv")
    args = parser.parse_args()

    draws = crawl_year(args.year)
    draws = filter_range(draws, args.start, args.end)
    df = merge_save(args.out, draws)
    logger.info(f"Saved {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
