# -*- coding: utf-8 -*-
from __future__ import annotations

import random
import time
from typing import Dict, List, Tuple

import requests
from bs4 import BeautifulSoup
from loguru import logger

from .config import (
    ZODIAC_SOURCE_URL,
    RESULTS_BASE_URL,
    REQUEST_TIMEOUT,
    RETRY_TIMES,
    RETRY_BACKOFF,
    USER_AGENT,
    CURRENT_YEAR,
)

HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9"}


def _get(url: str) -> requests.Response:
    last_exc = None
    for i in range(RETRY_TIMES):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp
            last_exc = RuntimeError(f"HTTP {resp.status_code}")
        except Exception as e:
            last_exc = e
        sleep_s = RETRY_BACKOFF ** i + random.random()
        logger.warning(f"GET {url} failed, retry in {sleep_s:.2f}s: {last_exc}")
        time.sleep(sleep_s)
    raise last_exc  # type: ignore[misc]


def parse_zodiac_2025(html: str) -> Tuple[Dict[str, List[int]], Dict[int, Dict[str, str]]]:
    """Parse zodiac mapping page for 2025.

    Returns:
      - zodiac_to_numbers: {生肖: [numbers]}
      - per_number_attrs: {num: {zodiac, color, odd_even, home_wild}}
    """
    soup = BeautifulSoup(html, "lxml")
    zodiac_to_numbers: Dict[str, List[int]] = {}
    per_number_attrs: Dict[int, Dict[str, str]] = {}

    # The actual DOM needs live inspection; try several common patterns defensively.
    tables = soup.find_all("table")
    for table in tables:
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if not headers:
            continue
        # Heuristics: table with 生肖/号码 or 波色/单双/家野
        if any("生肖" in h for h in headers) and any("号码" in h for h in headers):
            for tr in table.find_all("tr"):
                tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
                if len(tds) < 2:
                    continue
                zodiac = tds[0]
                nums = []
                for token in tds[1].replace("，", ",").replace(" ", ",").split(","):
                    token = token.strip()
                    if token.isdigit():
                        nums.append(int(token))
                if zodiac and nums:
                    zodiac_to_numbers[zodiac] = sorted(set(nums))
        # Attributes table
        if {"号码", "波色", "单双", "家野"}.issubset(set(headers)):
            col_idx = {h: i for i, h in enumerate(headers)}
            for tr in table.find_all("tr"):
                tds = [td.get_text(strip=True) for td in tr.find_all("td")]
                if len(tds) < len(headers):
                    continue
                try:
                    n = int(tds[col_idx["号码"]])
                except Exception:
                    continue
                per_number_attrs[n] = {
                    "zodiac": "",  # fill later from zodiac_to_numbers
                    "color": tds[col_idx["波色"]],
                    "odd_even": tds[col_idx["单双"]],
                    "home_wild": tds[col_idx["家野"]],
                }

    # backfill zodiac per number from zodiac_to_numbers
    for z, nums in zodiac_to_numbers.items():
        for n in nums:
            per_number_attrs.setdefault(n, {"zodiac": z, "color": "", "odd_even": "", "home_wild": ""})
            per_number_attrs[n]["zodiac"] = z

    if not zodiac_to_numbers:
        logger.warning("Failed to parse zodiac_to_numbers; the page DOM may differ")
    return zodiac_to_numbers, per_number_attrs


def fetch_zodiac_and_attrs(year: int = CURRENT_YEAR) -> Tuple[Dict[str, List[int]], Dict[int, Dict[str, str]]]:
    url = ZODIAC_SOURCE_URL
    if str(year) not in url:
        # Same URL contains year tabs; we keep the canonical URL
        pass
    resp = _get(url)
    zodiac_to_numbers, per_number_attrs = parse_zodiac_2025(resp.text)
    return zodiac_to_numbers, per_number_attrs


def parse_kj_year_page(html: str, fallback_year: int = CURRENT_YEAR) -> List[Tuple[int, int, List[int]]]:
    """Parse 2025 year page '/kj/?year=2025' into (year, issue, [7 numbers]).

    The DOM groups each issue as:
      <div class="kj-tit">... 第<span>289</span>期</div>
      <div class="kj-box"><ul class="clearfix"> <li> <dt class="ball-red">40</dt> ...
    We collect 7 numbers under the subsequent "kj-box".
    """
    soup = BeautifulSoup(html, "lxml")
    rows: List[Tuple[int, int, List[int]]] = []
    # iterate titles and next sibling kj-box
    for tit in soup.select('div.kj-tit'):
        text = tit.get_text(" ", strip=True)
        # find issue number between '第' and '期'
        issue_int = None
        try:
            if "第" in text and "期" in text:
                seg = text.split("第", 1)[1]
                seg = seg.split("期", 1)[0]
                digits = "".join(ch for ch in seg if ch.isdigit())
                if digits:
                    issue_int = int(digits)
        except Exception:
            issue_int = None
        if not issue_int:
            continue
        # find the next .kj-box sibling
        box = tit.find_next_sibling('div', class_='kj-box')
        if not box:
            continue
        # numbers appear as <dt class="ball-red|ball-green|ball-blue">NN</dt>
        nums: List[int] = []
        for dt in box.select('dt[class^="ball-"]'):
            val = dt.get_text(strip=True)
            if not val:
                continue
            # Some numbers might be like '05'
            if val.isdigit():
                nums.append(int(val))
            if len(nums) >= 7:
                break
        if len(nums) >= 7:
            rows.append((fallback_year, issue_int, nums[:7]))
    return rows


def fetch_issue_page(url: str) -> List[Tuple[int, int, List[int]]]:
    resp = _get(url)
    # try parse as kj-year page first; fallback to table parser
    rows = parse_kj_year_page(resp.text)
    if rows:
        return rows
    return []


def build_issue_list_urls(year: int, start_issue: int, end_issue: int) -> List[str]:
    # Year page includes all issues for the year
    base = RESULTS_BASE_URL.rstrip("/")
    return [f"{base}/?year={year}"]


def fetch_draws_for_year(year: int, start_issue: int, end_issue: int) -> List[Tuple[int, int, List[int]]]:
    urls = build_issue_list_urls(year, start_issue, end_issue)
    all_rows: List[Tuple[int, int, List[int]]] = []
    for url in urls:
        try:
            rows = fetch_issue_page(url)
            all_rows.extend(rows)
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
    # filter range and dedup by (year, issue)
    dedup = {}
    for y, iss, nums in all_rows:
        if y != year:
            continue
        if not (start_issue <= iss <= end_issue):
            continue
        dedup[(y, iss)] = (y, iss, nums)
    return sorted(dedup.values(), key=lambda x: x[1])
