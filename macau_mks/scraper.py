# -*- coding: utf-8 -*-
"""
Scraper for 新澳门六合彩 2025年开奖记录 from kj.123720c.com/kj/.

- Iterates months/years or uses the year parameter (?year=2025)
- Parses each <div class="kj-tit"> and its following <div class="kj-box">
- Extracts six normal balls and one special (特码) from the list
- Captures date and period index

Output helpers save to CSV/JSON for reproducibility.
"""
import re
import time
import json
import random
from typing import Iterable, List, Tuple
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore
try:
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover
    from .log import logger

from .types import DrawRecord

BASE_URL = "https://kj.123720c.com"
INDEX_PATH = "/kj/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9"
}


def _fetch_with_requests(url: str) -> str:
    s = requests.Session()
    r = s.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text


def _fetch_with_urllib(url: str) -> str:  # fallback when requests missing
    import urllib.request
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    try:
        return raw.decode("utf-8")
    except Exception:
        return raw.decode("gb18030", errors="ignore")


def fetch_year_html(year: int, session=None) -> str:
    url = f"{BASE_URL}{INDEX_PATH}?year={year}&sort=l"
    if requests is None:
        return _fetch_with_urllib(url)
    else:
        return _fetch_with_requests(url)


def parse_draws_from_html_bs4(html: str) -> List[DrawRecord]:
    assert BeautifulSoup is not None
    soup = BeautifulSoup(html, "html.parser")
    container = soup
    records: List[DrawRecord] = []
    for tit in container.find_all("div", class_="kj-tit"):
        title_text = tit.get_text(" ", strip=True)
        m_date = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", title_text)
        m_period = re.search(r"第\s*(\d{1,3})\s*期", title_text)
        if not m_date or not m_period:
            # attempt to strip tags in between
            m_period = re.search(r"第(?:<[^>]+>)*?(\d{1,3})(?:<[^>]+>)*?期", str(tit))
            if not m_date or not m_period:
                logger.warning(f"Skip unmatched title: {title_text}")
                continue
        year, month, day = map(int, m_date.groups())
        period = int(m_period.group(1))
        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        box = tit.find_next_sibling("div", class_="kj-box")
        if not box:
            logger.warning(f"No result box for period {period}")
            continue
        nums = []
        for dt in box.select("dt[class^='ball-']"):
            t = dt.get_text(strip=True)
            if t.isdigit():
                nums.append(int(t))
        if len(nums) < 7:
            logger.warning(f"Period {period} has <7 numbers parsed: {nums}")
            continue
        records.append(DrawRecord(period=period, date=date_str, normals=nums[:6], special=nums[6]))
    return records


def parse_draws_from_html_regex(html: str) -> List[DrawRecord]:
    # Split on kj-tit blocks
    parts = re.split(r'<div\s+class="kj-tit">', html)
    records: List[DrawRecord] = []
    for block in parts[1:]:
        # Extract date
        md = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', block)
        # Extract period allowing tags between
        mp = re.search(r'第(?:<[^>]*>)*?([0-9]{1,3})(?:<[^>]*>)*?期', block)
        if not md or not mp:
            continue
        year, month, day = map(int, md.groups())
        period = int(mp.group(1))
        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        # Extract numbers inside dt class="ball-...">NN</dt>
        nums = [int(x) for x in re.findall(r'<dt\s+class="ball-[^"]+">\s*([0-9]{2})\s*</dt>', block)]
        if len(nums) < 7:
            # try single-digit too
            nums = [int(x) for x in re.findall(r'<dt\s+class="ball-[^"]+">\s*([0-9]{1,2})\s*</dt>', block)]
        if len(nums) < 7:
            continue
        records.append(DrawRecord(period=period, date=date_str, normals=nums[:6], special=nums[6]))
    return records


def parse_draws_from_html(html: str) -> List[DrawRecord]:
    if BeautifulSoup is not None:
        try:
            return parse_draws_from_html_bs4(html)
        except Exception as e:
            logger.warning(f"bs4 parse failed, fallback to regex: {e}")
            return parse_draws_from_html_regex(html)
    return parse_draws_from_html_regex(html)


def crawl_year(year: int) -> List[DrawRecord]:
    html = fetch_year_html(year, None)
    recs = parse_draws_from_html(html)
    # Sort by period ascending (001..291)
    recs.sort(key=lambda r: r.period)
    return recs


def save_records_csv_json(records: List[DrawRecord], csv_path: str, json_path: str) -> None:
    import csv
    import os
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    # CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["period", "date", "n1", "n2", "n3", "n4", "n5", "n6", "special"])
        for r in records:
            row = [r.period, r.date] + r.normals + [r.special]
            writer.writerow(row)
    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in records], f, ensure_ascii=False, indent=2)


def crawl_2025_and_save(data_dir: str) -> List[DrawRecord]:
    records = crawl_year(2025)
    save_records_csv_json(
        records,
        csv_path=f"{data_dir}/macau_2025.csv",
        json_path=f"{data_dir}/macau_2025.json",
    )
    logger.info(f"Crawled {len(records)} records for 2025 and saved to {data_dir}")
    return records
