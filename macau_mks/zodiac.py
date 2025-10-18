# -*- coding: utf-8 -*-
"""
Zodiac mapping for 新澳门六合彩 numbers for a given year.
Fetches /kj/sx.html?year=YYYY and parses number lists per zodiac.
Provides models based on zodiac transitions for 特码.
"""
import re
from typing import Dict, List

try:
    import requests  # type: ignore
except Exception:
    requests = None

HEADERS = {
    "User-Agent": "Mozilla/5.0",
}
BASE_URL = "https://kj.123720c.com"


def _fetch(url: str) -> str:
    if requests is not None:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        return r.text
    else:
        import urllib.request
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
        try:
            return raw.decode("utf-8")
        except Exception:
            return raw.decode("gb18030", errors="ignore")


def fetch_number_to_zodiac(year: int) -> Dict[int, str]:
    html = _fetch(f"{BASE_URL}/kj/sx.html?year={year}")
    # Page likely contains blocks like: <span class="sx sx-shu">鼠</span> ... numbers ...
    # We'll find zodiac sections and capture subsequent numbers in <a> tags or plain text
    mapping: Dict[int, str] = {}
    # Find sections: zodiac header and following list of numbers up to next header
    blocks = re.split(r'<span[^>]*class="sx[^>]*">', html)
    for blk in blocks[1:]:
        # zodiac char before closing span
        mname = re.match(r'([^<]+)</span>', blk)
        if not mname:
            continue
        zname = mname.group(1).strip()
        # Find all numbers in this block up to the next zodiac span
        seg = blk.split('<span', 1)[0]
        nums = re.findall(r'(?:>|\s)([0-9]{1,2})(?:<|\s)', seg)
        for s in nums:
            n = int(s)
            if 1 <= n <= 49:
                mapping[n] = zname
    return mapping


def build_zodiac_transition(records: List, num_to_zodiac: Dict[int, str]):
    # Transition counts from prev special's zodiac to current special's zodiac
    zodiacs = [num_to_zodiac.get(r.special, "?") for r in records]
    zset = sorted(set([z for z in zodiacs if z != "?"]))
    idx = {z: i for i, z in enumerate(zset)}
    tm = [[0 for _ in zset] for __ in zset]
    for i in range(1, len(zodiacs)):
        a, b = zodiacs[i-1], zodiacs[i]
        if a in idx and b in idx:
            tm[idx[a]][idx[b]] += 1
    return zset, tm


def zodiac_follow_model(records: List, num_to_zodiac: Dict[int, str]):
    # Predict distribution over numbers for next draw, based on prev zodiac -> next zodiac proportions
    if not records:
        return [1/49.0] * 49
    zset, tm = build_zodiac_transition(records, num_to_zodiac)
    if not zset:
        return [1/49.0] * 49
    last_z = num_to_zodiac.get(records[-1].special)
    if last_z not in zset:
        return [1/49.0] * 49
    zi = zset.index(last_z)
    row = tm[zi]
    s = sum(row)
    if s == 0:
        # fallback uniform zodiac
        row_probs = [1/len(zset)] * len(zset)
    else:
        row_probs = [v/s for v in row]
    # Distribute zodiac probabilities across numbers in that zodiac uniformly
    probs = [0.0] * 49
    # Build per zodiac number lists
    z_to_nums: Dict[str, List[int]] = {}
    for n, z in num_to_zodiac.items():
        z_to_nums.setdefault(z, []).append(n)
    for z, pz in zip(zset, row_probs):
        nums = [n for n in z_to_nums.get(z, []) if 1 <= n <= 49]
        if not nums:
            continue
        share = pz / len(nums)
        for n in nums:
            probs[n-1] += share
    # Normalize
    s2 = sum(probs)
    if s2 == 0:
        return [1/49.0] * 49
    return [v/s2 for v in probs]
