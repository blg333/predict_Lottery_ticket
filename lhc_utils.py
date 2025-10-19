# -*- coding:utf-8 -*-
"""
Utilities for Macau LHC scraping helpers and attribute parsing.
"""
from __future__ import annotations

import re
import time
from typing import Dict, List

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


def fetch_html(url: str, params: dict | None = None, retries: int = 2) -> str:
    last_err = None
    for i in range(retries + 1):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=20)
            resp.encoding = resp.apparent_encoding or "utf-8"
            return resp.text
        except Exception as e:
            last_err = e
            time.sleep(0.5 * (i + 1))
    raise last_err


def get_bose_map(year: int = 2025) -> Dict[str, List[int]]:
    """Parse 波色 lists (红/蓝/绿) from statistics page."""
    html = fetch_html("https://kj.123720c.com/kj/zl.html", params={"year": year})
    soup = BeautifulSoup(html, "lxml")
    bose = {"红": [], "蓝": [], "绿": []}
    # Classes: hongbospan, lanbospan, lvbospan
    mapping = {
        "红": soup.select_one("td.hongbospan"),
        "蓝": soup.select_one("td.lanbospan"),
        "绿": soup.select_one("td.lvbospan"),
    }
    for k, td in mapping.items():
        if not td:
            continue
        nums = [int(span.get_text(strip=True)) for span in td.select("span") if span.get_text(strip=True).isdigit()]
        bose[k] = nums
    return bose


def get_zodiac_map(year: int = 2025) -> Dict[str, List[int]]:
    """Parse zodiac mapping. Prefer zl.html (tab 2), fallback to sx.html."""
    html = fetch_html("https://kj.123720c.com/kj/zl.html", params={"year": year})
    soup = BeautifulSoup(html, "lxml")
    zodiac_map: Dict[str, List[int]] = {}
    # zl.html structure
    block = soup.select_one(".con.sxsuxing2")
    if block:
        tds = block.select("td")
        for td in tds:
            i = td.select_one("i")
            if not i:
                continue
            name = i.get_text(strip=True)
            nums = [int(span.get_text(strip=True)) for span in td.select("span") if span.get_text(strip=True).isdigit()]
            if nums:
                zodiac_map[name] = nums
    if zodiac_map:
        return zodiac_map
    # fallback: sx.html
    html = fetch_html("https://kj.123720c.com/kj/sx.html", params={"year": year})
    soup = BeautifulSoup(html, "lxml")
    lis = soup.select("ul.sxdz li")
    for li in lis:
        sx = li.select_one(".sx")
        if not sx:
            continue
        name = re.sub(r"\s*\[.*?\]", "", sx.get_text(strip=True))
        nums = [int(s) for s in re.findall(r"\d+", li.get_text())]
        if nums:
            zodiac_map[name] = nums
    return zodiac_map


def number_to_bose(bose_map: Dict[str, List[int]]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for color, nums in bose_map.items():
        for n in nums:
            mapping[n] = color
    return mapping


def number_to_zodiac(zodiac_map: Dict[str, List[int]]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for name, nums in zodiac_map.items():
        for n in nums:
            mapping[n] = name
    return mapping


def compute_basic_attributes(numbers: List[int]) -> Dict[int, dict]:
    """Compute parity, size (大/小), head/tail for each number."""
    attrs: Dict[int, dict] = {}
    for n in numbers:
        attrs[n] = {
            "单双": "单" if n % 2 == 1 else "双",
            "大小": "大" if n >= 25 else "小",
            "头数": n // 10,
            "尾数": n % 10,
        }
    return attrs
