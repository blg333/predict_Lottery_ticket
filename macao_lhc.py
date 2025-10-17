# -*- coding: utf-8 -*-
"""
Scraper + analysis for 新澳门六合彩 2025 issues. 
- Crawls issues 001-290 from kj.123720c.com (year=2025), monthly pages supported
- Computes frequency stats, hot/cold numbers, simple transition matrix, backtest,
  and recommends 290期特碼 and predicts 291期特碼候选。

Note: This script is independent from the existing ssq/dlt Tensorflow model code.

Usage:
  python macao_lhc.py fetch --start 1 --end 290 --out data/macao_2025.csv
  python macao_lhc.py analyze --data data/macao_2025.csv --show 10
  python macao_lhc.py recommend --data data/macao_2025.csv

"""
import argparse
import csv
import dataclasses
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://kj.123720c.com/kj/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

@dataclass
class IssueRecord:
    issue: int  # 1..n within year
    date: str   # YYYY-MM-DD
    numbers: List[int]  # length 7: 6平碼 + 1特碼 (assumed last is 特碼)


def fetch_year_html(year: int, month: Optional[int] = None) -> str:
    params = {"year": str(year)}
    if month is not None:
        params["month"] = str(month)
    resp = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def parse_issues_from_html(html: str) -> List[IssueRecord]:
    soup = BeautifulSoup(html, "lxml")
    results: List[IssueRecord] = []
    # Each block uses .kj-tit (contains date and issue number), followed by .kj-box with 7 numbers
    for tit in soup.select(".kj-tit"):
        text = tit.get_text(" ", strip=True)
        m_issue = re.search(r"第\s*(\d+)\s*期", text)
        m_date = re.search(r"(\d{4})年(\d{2})月(\d{2})日|((\d{4})-(\d{2})-(\d{2}))", text)
        issue = int(m_issue.group(1)) if m_issue else None
        # Date may only exist in sibling elements; fallback not critical for analysis
        date_str = ""
        if m_date:
            if m_date.group(1):
                date_str = f"{m_date.group(1)}-{m_date.group(2)}-{m_date.group(3)}"
            elif m_date.group(4):
                date_str = f"{m_date.group(5)}-{m_date.group(6)}-{m_date.group(7)}"
        box = tit.find_next("div", class_="kj-box")
        nums: List[int] = []
        if box:
            for dt in box.select("dt"):
                s = dt.get_text(strip=True)
                s = s.strip()
                if s.isdigit():
                    nums.append(int(s))
        if issue and len(nums) >= 7:
            # Normally 6平碼 + 空格 + 1特碼; some pages have extra markup; trim to first 7
            numbers = nums[:7]
            results.append(IssueRecord(issue=issue, date=date_str, numbers=numbers))
    return results


def fetch_issues_2025(start_issue: int, end_issue: int) -> List[IssueRecord]:
    all_issues: Dict[int, IssueRecord] = {}
    # Pages list latest first; also month pages present
    # Crawl full year page first
    html = fetch_year_html(2025)
    for rec in parse_issues_from_html(html):
        if start_issue <= rec.issue <= end_issue:
            all_issues[rec.issue] = rec
    # Crawl per-month to ensure coverage
    for m in range(1, 13):
        try:
            html_m = fetch_year_html(2025, month=m)
            for rec in parse_issues_from_html(html_m):
                if start_issue <= rec.issue <= end_issue:
                    all_issues[rec.issue] = rec
            time.sleep(0.3)
        except Exception:
            continue
    issues = [all_issues[i] for i in sorted(all_issues.keys()) if start_issue <= i <= end_issue and i in all_issues]
    # Ensure continuous sequence, but tolerate gaps
    return issues


def write_csv(path: str, records: List[IssueRecord]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["issue", "date", "n1", "n2", "n3", "n4", "n5", "n6", "tm"])  # tm=特碼
        for r in records:
            row = [r.issue, r.date] + r.numbers[:7]
            w.writerow(row)


def read_csv(path: str) -> List[IssueRecord]:
    out: List[IssueRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nums = [int(row[k]) for k in ["n1", "n2", "n3", "n4", "n5", "n6", "tm"]]
            out.append(IssueRecord(issue=int(row["issue"]), date=row.get("date", ""), numbers=nums))
    return out


def compute_frequency(records: List[IssueRecord]) -> Tuple[Counter, Counter]:
    freq_all = Counter()
    freq_tm = Counter()
    for r in records:
        for n in r.numbers[:6]:
            freq_all[n] += 1
        tm = r.numbers[6]
        freq_tm[tm] += 1
    return freq_all, freq_tm


def hot_cold(freq: Counter, top_k: int = 10) -> Tuple[List[Tuple[int,int]], List[Tuple[int,int]]]:
    items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    cold_items = sorted(freq.items(), key=lambda x: (x[1], x[0]))
    return items[:top_k], cold_items[:top_k]


def transition_matrix(records: List[IssueRecord]) -> Dict[int, Counter]:
    trans: Dict[int, Counter] = {i: Counter() for i in range(1, 50)}
    # Use 特碼 transitions only (tm_t -> tm_{t+1})
    prev_tm: Optional[int] = None
    for r in sorted(records, key=lambda x: x.issue):
        tm = r.numbers[6]
        if prev_tm is not None:
            trans[prev_tm][tm] += 1
        prev_tm = tm
    return trans


def recommend_tm_for_issue(records: List[IssueRecord], target_issue: int) -> Tuple[int, List[Tuple[int, float]]]:
    # Simple hybrid score: frequency_tm normalized + transition likelihood from last tm + recency decay
    if not records:
        return 0, []
    records_sorted = sorted(records, key=lambda x: x.issue)
    freq_all, freq_tm = compute_frequency(records)
    max_freq = max(freq_tm.values()) if freq_tm else 1
    last_tm = records_sorted[-1].numbers[6]
    trans = transition_matrix(records)
    cand_scores: Dict[int, float] = {}
    N = len(records_sorted)
    # Recency: boost numbers that appeared in last 30 issues as tm less often (anti-streak)
    recent_window = records_sorted[-30:] if N >= 30 else records_sorted
    recent_tm_counts = Counter([r.numbers[6] for r in recent_window])
    for n in range(1, 50):
        score = 0.0
        score += (freq_tm[n] / max_freq) * 0.5
        next_count = trans[last_tm][n]
        total_from_last = sum(trans[last_tm].values()) or 1
        score += (next_count / total_from_last) * 0.35
        # Anti-hot recent penalty (favor colder in last 30)
        recent_penalty = recent_tm_counts[n] / max(1, len(recent_window))
        score += (0.15 * (1 - recent_penalty))
        cand_scores[n] = score
    ranked = sorted(cand_scores.items(), key=lambda x: (-x[1], x[0]))
    best_tm = ranked[0][0]
    return best_tm, ranked[:10]


def backtest_tm_top1(records: List[IssueRecord]) -> float:
    # Predict next tm for each step using history up to t-1
    correct = 0
    total = 0
    for i in range(1, len(records)):
        hist = records[:i]
        pred, _ = recommend_tm_for_issue(hist, records[i].issue)
        if pred == records[i].numbers[6]:
            correct += 1
        total += 1
    return correct / total if total else 0.0


def cmd_fetch(args):
    records = fetch_issues_2025(args.start, args.end)
    if not records or records[0].issue != args.start or records[-1].issue != args.end:
        print(f"Warning: fetched {len(records)} records; expected coverage {args.start}-{args.end}")
    write_csv(args.out, records)
    print(f"Saved {len(records)} issues to {args.out}")


def cmd_analyze(args):
    records = read_csv(args.data)
    freq_all, freq_tm = compute_frequency(records)
    hot_all, cold_all = hot_cold(freq_all, top_k=args.show)
    hot_tm, cold_tm = hot_cold(freq_tm, top_k=args.show)
    acc = backtest_tm_top1(records)
    last_issue = max(r.issue for r in records)
    best_tm_291, top_tm_scores = recommend_tm_for_issue(records, last_issue + 1)
    print("分析概览")
    print(f"期数范围: {records[0].issue}-{records[-1].issue}, 总计: {len(records)}")
    print(f"平碼热门(top{args.show}): {hot_all}")
    print(f"平碼冷门(top{args.show}): {cold_all}")
    print(f"特碼热门(top{args.show}): {hot_tm}")
    print(f"特碼冷门(top{args.show}): {cold_tm}")
    print(f"特碼过往TOP1回测准确率: {acc:.3f}")
    print(f"第{last_issue}期开奖结果: {records[-1].numbers}")
    print(f"第{last_issue}期特碼号: {records[-1].numbers[6]}")
    print(f"第{last_issue+1}期(291)推荐特碼: {best_tm_291}")
    print(f"候选Top10: {top_tm_scores}")


def cmd_recommend(args):
    records = read_csv(args.data)
    last_issue = max(r.issue for r in records)
    # 推荐第290期特码（已开奖的事实值）和预测291
    rec_290 = [r for r in records if r.issue == 290]
    if rec_290:
        print(f"第290期开奖: {rec_290[0].numbers}")
        print(f"第290期特码: {rec_290[0].numbers[6]}")
    best_tm_291, top_tm_scores = recommend_tm_for_issue(records, last_issue + 1)
    print(f"第291期候选特碼Top10: {top_tm_scores}")
    print(f"第291期预测特碼: {best_tm_291}")


def build_arg_parser():
    p = argparse.ArgumentParser(description="新澳门六合彩 2025 爬虫+分析")
    sub = p.add_subparsers(dest="cmd", required=True)
    pf = sub.add_parser("fetch", help="抓取期数范围")
    pf.add_argument("--start", type=int, default=1)
    pf.add_argument("--end", type=int, default=290)
    pf.add_argument("--out", type=str, default="data/macao_2025.csv")
    pf.set_defaults(func=cmd_fetch)

    pa = sub.add_parser("analyze", help="频率统计与回测")
    pa.add_argument("--data", type=str, default="data/macao_2025.csv")
    pa.add_argument("--show", type=int, default=10)
    pa.set_defaults(func=cmd_analyze)

    pr = sub.add_parser("recommend", help="输出指定推荐")
    pr.add_argument("--data", type=str, default="data/macao_2025.csv")
    pr.set_defaults(func=cmd_recommend)
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
