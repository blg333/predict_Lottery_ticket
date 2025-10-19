# -*- coding:utf-8 -*-
"""
Utilities for Mark Six (六合彩) meta information:
- Wave color (波色) mapping for numbers 1..49
- Zodiac (生肖) mapping for numbers, fetched from the provided reference page
  `https://kj.123720c.com/kj/zl.html` with a cached fallback.

If fetching zodiac mapping fails, functions gracefully return None or "未知".
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

from loguru import logger

# Lazy imports for optional dependencies
try:
    import requests  # type: ignore
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore
    BeautifulSoup = None  # type: ignore


# Static wave color mapping, widely used in Mark Six
# 红波: 1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,45,46
# 蓝波: 3,4,9,10,14,15,20,25,26,31,36,37,41,42,47,48
# 绿波: 5,6,11,16,17,21,22,27,28,32,33,38,39,43,44,49
RED_WAVE = {1, 2, 7, 8, 12, 13, 18, 19, 23, 24, 29, 30, 34, 35, 40, 45, 46}
BLUE_WAVE = {3, 4, 9, 10, 14, 15, 20, 25, 26, 31, 36, 37, 41, 42, 47, 48}
GREEN_WAVE = {5, 6, 11, 16, 17, 21, 22, 27, 28, 32, 33, 38, 39, 43, 44, 49}


def get_wave_color_map() -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for n in range(1, 50):
        if n in RED_WAVE:
            mapping[n] = "红波"
        elif n in BLUE_WAVE:
            mapping[n] = "蓝波"
        elif n in GREEN_WAVE:
            mapping[n] = "绿波"
        else:
            mapping[n] = "未知"
    return mapping


def get_wave_color(number: int) -> str:
    return get_wave_color_map().get(int(number), "未知")


ZODIACS = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]


def _parse_numbers_from_text(text: str) -> Dict[str, set]:
    """Parse zodiac lines like '鼠: 01 13 25 37 49' into dict.
    Robust to different separators.
    """
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    result: Dict[str, set] = {}
    for name in ZODIACS:
        if name in text:
            # try split by the zodiac name
            parts = text.split(name)
            # join with marker and split again to capture after name occurrences
            text = (" "+name+" ").join(parts)
    # After normalization, extract per zodiac by scanning
    for name in ZODIACS:
        idx = text.find(name)
        if idx == -1:
            continue
        tail = text[idx + len(name):]
        # cut at next zodiac name occurrence
        next_indices = [tail.find(z) for z in ZODIACS if tail.find(z) != -1]
        cut = min(next_indices) if next_indices else len(tail)
        seg = tail[:cut]
        # keep only digits and spaces/commas
        buf = []
        for ch in seg:
            if ch.isdigit() or ch in {" ", ",", ":", "-", "—", "|", "/"}:
                buf.append(ch)
            else:
                buf.append(" ")
        nums = []
        for token in "".join(buf).replace(",", " ").split():
            try:
                n = int(token)
                if 1 <= n <= 49:
                    nums.append(n)
            except Exception:
                continue
        if nums:
            result[name] = set(nums)
    return result


def fetch_zodiac_mapping(url: str = "https://kj.123720c.com/kj/zl.html") -> Optional[Dict[str, set]]:
    if requests is None or BeautifulSoup is None:
        logger.warning("requests/bs4 not installed; skip zodiac fetch")
        return None
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        soup = BeautifulSoup(r.text, "lxml")
        text_chunks = []
        # Gather visible text within relevant sections
        for tag in soup.find_all(text=True):  # type: ignore
            t = str(tag).strip()
            if any(name in t for name in ZODIACS):
                text_chunks.append(t)
        merged = "\n".join(text_chunks)
        mapping = _parse_numbers_from_text(merged)
        if not mapping:
            logger.warning("Failed to parse zodiac mapping from page text")
            return None
        # ensure coverage for all zodiacs by filling empty sets
        for name in ZODIACS:
            mapping.setdefault(name, set())
        return mapping
    except Exception as e:  # pragma: no cover
        logger.warning(f"Fetch zodiac mapping failed: {e}")
        return None


def cache_path() -> str:
    return os.path.join("data", "lhc", "zodiac_mapping.json")


def load_cached_zodiac_mapping() -> Optional[Dict[str, set]]:
    path = cache_path()
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return {k: set(v) for k, v in raw.items()}
    except Exception as e:  # pragma: no cover
        logger.warning(f"Load cached zodiac mapping failed: {e}")
    return None


def save_cached_zodiac_mapping(mapping: Dict[str, set]) -> None:
    path = cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    safe = {k: sorted(list(v)) for k, v in mapping.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe, f, ensure_ascii=False)


def get_zodiac_map(use_cache: bool = True) -> Optional[Dict[int, str]]:
    """Return number->zodiac map if available, else None.
    Tries cache first, then network fetch.
    """
    mapping: Optional[Dict[str, set]] = None
    if use_cache:
        cached = load_cached_zodiac_mapping()
        if cached:
            mapping = cached
    if mapping is None:
        fetched = fetch_zodiac_mapping()
        if fetched:
            mapping = fetched
            save_cached_zodiac_mapping(fetched)
    if mapping is None:
        return None
    # invert mapping
    inv: Dict[int, str] = {}
    for name, nums in mapping.items():
        for n in nums:
            inv[int(n)] = name
    return inv


def get_number_zodiac(number: int, number_to_zodiac: Optional[Dict[int, str]] = None) -> str:
    if number_to_zodiac is None:
        number_to_zodiac = get_zodiac_map(use_cache=True)
    if number_to_zodiac is None:
        return "未知"
    return number_to_zodiac.get(int(number), "未知")
