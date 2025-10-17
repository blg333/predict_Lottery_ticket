# -*- coding: utf-8 -*-
from __future__ import annotations
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
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np  # type: ignore
except Exception:  # Fallback if numpy not installed yet
    class _NPUnavailable:  # type: ignore
        ndarray = None
    np = None  # type: ignore

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


def _last_occurrence_index(records: List[IssueRecord], is_tm: bool, number: int) -> Optional[int]:
    # Return distance from end (1=last issue) where number last appeared; None if never
    for idx, r in enumerate(reversed(sorted(records, key=lambda x: x.issue))):
        if is_tm:
            if r.numbers[6] == number:
                return idx + 1
        else:
            if number in r.numbers[:6]:
                return idx + 1
    return None


def frequency_balanced_tm_scores(records: List[IssueRecord], recent_boost_window: int = 20) -> List[Tuple[int, float]]:
    # Balance by deficit from uniform expectation + recency boost
    if not records:
        return []
    _, freq_tm = compute_frequency(records)
    expected = len(records) / 49.0
    # Normalize deficits and recency
    scores: Dict[int, float] = {}
    max_recency = 0
    recencies: Dict[int, float] = {}
    for n in range(1, 50):
        d = expected - freq_tm[n]
        deficit = max(0.0, d) / expected if expected > 0 else 0.0
        dist = _last_occurrence_index(records, is_tm=True, number=n)
        rec = recent_boost_window if dist is None else min(recent_boost_window, float(dist))
        recencies[n] = rec
        if rec > max_recency:
            max_recency = rec
        scores[n] = deficit  # initial
    # Add normalized recency component
    if max_recency > 0:
        for n in range(1, 50):
            scores[n] = 0.7 * scores[n] + 0.3 * (recencies[n] / max_recency)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return ranked


def frequency_balanced_all_picks(records: List[IssueRecord], k: int = 6, recent_boost_window: int = 10) -> Tuple[List[int], List[Tuple[int, float]]]:
    # Balance by deficit for 平码 slots across history; greedily pick top-k
    if not records:
        return [], []
    freq_all, _ = compute_frequency(records)
    expected = (len(records) * 6) / 49.0
    scores: Dict[int, float] = {}
    max_recency = 0
    recencies: Dict[int, float] = {}
    for n in range(1, 50):
        d = expected - freq_all[n]
        deficit = max(0.0, d) / expected if expected > 0 else 0.0
        dist = _last_occurrence_index(records, is_tm=False, number=n)
        rec = recent_boost_window if dist is None else min(recent_boost_window, float(dist))
        recencies[n] = rec
        if rec > max_recency:
            max_recency = rec
        scores[n] = deficit
    if max_recency > 0:
        for n in range(1, 50):
            scores[n] = 0.7 * scores[n] + 0.3 * (recencies[n] / max_recency)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    picks = [n for n, _ in ranked[:k]]
    return picks, ranked[:max(k, 10)]


def cmd_balance(args):
    records = read_csv(args.data)
    ranked_tm = frequency_balanced_tm_scores(records, recent_boost_window=args.recent_tm)
    tm_pred = ranked_tm[0][0] if ranked_tm else 0
    picks, ranked_all = frequency_balanced_all_picks(records, k=args.k, recent_boost_window=args.recent_all)
    last_issue = max(r.issue for r in records)
    print("频率平衡预测")
    print(f"数据期数范围: {records[0].issue}-{records[-1].issue}，总计{len(records)}期")
    print(f"第{last_issue+1}期特碼(频率平衡)预测: {tm_pred}")
    print(f"特碼候选Top10(频率平衡): {ranked_tm[:10]}")
    print(f"平衡选取6个平码: {sorted(picks)}")
    print(f"平码候选Top{max(10, args.k)}(频率平衡): {ranked_all}")


def omission_scores(records: List[IssueRecord], is_tm: bool) -> List[Tuple[int, int]]:
    # 当前遗漏值 = 距离上次出现的期数（上期出现则为0），从最新往回
    N = len(records)
    ranked: List[Tuple[int, int]] = []
    for n in range(1, 50):
        dist = _last_occurrence_index(records, is_tm=is_tm, number=n)
        miss = N if dist is None else max(0, dist - 1)
        ranked.append((n, miss))
    ranked.sort(key=lambda x: (-x[1], x[0]))
    return ranked


def cmd_omit(args):
    records = read_csv(args.data)
    last_issue = max(r.issue for r in records)
    tm_ranked = omission_scores(records, is_tm=True)
    pm_ranked = omission_scores(records, is_tm=False)
    tm_pick = tm_ranked[0][0]
    pm_picks = sorted([n for n, _ in pm_ranked[:args.k]])
    print("遗漏值预测")
    print(f"数据期数范围: {records[0].issue}-{records[-1].issue}，总计{len(records)}期")
    print(f"第{last_issue+1}期特碼(遗漏)预测: {tm_pick}")
    print(f"特碼候选Top10(遗漏): {tm_ranked[:10]}")
    print(f"遗漏选取{args.k}个平码: {pm_picks}")
    print(f"平码候选Top{max(10, args.k)}(遗漏): {pm_ranked[:max(10, args.k)]}")


def compute_trend_slopes(records: List[IssueRecord], is_tm: bool, window: Optional[int] = None) -> List[Tuple[int, float]]:
    seq = sorted(records, key=lambda x: x.issue)
    if window is not None and window > 0:
        seq = seq[-window:]
    N = len(seq)
    if N <= 1:
        return [(n, 0.0) for n in range(1, 50)]
    t_vals = list(range(1, N + 1))
    t_bar = (N + 1) / 2.0
    S_tt = sum((t - t_bar) ** 2 for t in t_vals)
    slopes: List[Tuple[int, float]] = []
    for n in range(1, 50):
        y = []
        for r in seq:
            if is_tm:
                y.append(1.0 if r.numbers[6] == n else 0.0)
            else:
                y.append(1.0 if n in r.numbers[:6] else 0.0)
        y_bar = sum(y) / N
        S_ty = sum((t - t_bar) * (y_i - y_bar) for t, y_i in zip(t_vals, y))
        slope = S_ty / S_tt if S_tt > 0 else 0.0
        slopes.append((n, slope))
    slopes.sort(key=lambda x: (-x[1], x[0]))
    return slopes


def cmd_trend(args):
    records = read_csv(args.data)
    last_issue = max(r.issue for r in records)
    tm_slopes = compute_trend_slopes(records, is_tm=True, window=args.window)
    pm_slopes = compute_trend_slopes(records, is_tm=False, window=args.window)
    tm_pick = tm_slopes[0][0]
    pm_picks = sorted([n for n, _ in pm_slopes[:args.k]])
    print("趋势预测")
    print(f"使用窗口: {args.window if args.window else '全部'}；数据期数范围: {records[0].issue}-{records[-1].issue}")
    print(f"第{last_issue+1}期特碼(趋势)预测: {tm_pick}")
    print(f"特碼候选Top10(趋势): {tm_slopes[:10]}")
    print(f"趋势选取{args.k}个平码: {pm_picks}")
    print(f"平码候选Top{max(10, args.k)}(趋势): {pm_slopes[:max(10, args.k)]}")


def pm_transition_matrix(records: List[IssueRecord]) -> Dict[int, Counter]:
    # For each a in PM_t and b in PM_{t+1}, increment a->b
    trans: Dict[int, Counter] = {i: Counter() for i in range(1, 50)}
    seq = sorted(records, key=lambda x: x.issue)
    for i in range(len(seq) - 1):
        cur = set(seq[i].numbers[:6])
        nxt = set(seq[i + 1].numbers[:6])
        for a in cur:
            for b in nxt:
                trans[a][b] += 1
    return trans


def cmd_assoc(args):
    records = read_csv(args.data)
    last_issue = max(r.issue for r in records)
    # TM via transition from last TM
    tm_trans = transition_matrix(records)
    last_tm = sorted(records, key=lambda x: x.issue)[-1].numbers[6]
    cand_tm = tm_trans[last_tm]
    total = sum(cand_tm.values())
    tm_ranked = []
    for n in range(1, 50):
        p = (cand_tm[n] + 1) / (total + 49) if total else 1.0 / 49
        tm_ranked.append((n, p))
    tm_ranked.sort(key=lambda x: (-x[1], x[0]))
    tm_pick = tm_ranked[0][0]
    # PM via PM->PM association from last PM set
    pm_trans = pm_transition_matrix(records)
    last_pm = set(sorted(records, key=lambda x: x.issue)[-1].numbers[:6])
    pm_scores: Dict[int, float] = defaultdict(float)
    for a in last_pm:
        row = pm_trans[a]
        row_total = sum(row.values())
        for b in range(1, 50):
            pm_scores[b] += ((row[b] + 1) / (row_total + 49))
    pm_ranked = sorted(pm_scores.items(), key=lambda x: (-x[1], x[0]))
    pm_picks = []
    for n, _ in pm_ranked:
        if n not in last_pm:
            pm_picks.append(n)
        if len(pm_picks) >= args.k:
            break
    pm_picks = sorted(pm_picks)
    print("关联性预测")
    print(f"第{last_issue+1}期特碼(转移)预测: {tm_pick}")
    print(f"特碼候选Top10(转移): {tm_ranked[:10]}")
    print(f"关联选取{args.k}个平码: {pm_picks}")
    print(f"平码候选Top{max(10, args.k)}(转移): {pm_ranked[:max(10, args.k)]}")


def kmeans_cluster(X: List[List[float]], k: int, iters: int = 30) -> Tuple[List[int], List[List[float]]]:
    n = len(X)
    d = len(X[0]) if n else 0
    k = max(1, min(k, n))
    # init centers by picking k evenly spaced samples
    idxs = [int(i * n / k) for i in range(k)]
    centers = [X[i][:] for i in idxs]
    assign = [0] * n
    def dist2(a: List[float], b: List[float]) -> float:
        return sum((ai - bi) * (ai - bi) for ai, bi in zip(a, b))
    for _ in range(iters):
        changed = False
        for i, x in enumerate(X):
            best_j = 0
            best_d = dist2(x, centers[0])
            for j in range(1, k):
                dj = dist2(x, centers[j])
                if dj < best_d:
                    best_d = dj
                    best_j = j
            if assign[i] != best_j:
                assign[i] = best_j
                changed = True
        # recompute centers
        sums = [[0.0] * d for _ in range(k)]
        counts = [0] * k
        for a, x in zip(assign, X):
            counts[a] += 1
            for j in range(d):
                sums[a][j] += x[j]
        for j in range(k):
            if counts[j] > 0:
                centers[j] = [s / counts[j] for s in sums[j]]
        if not changed:
            break
    return assign, centers


def cmd_cluster(args):
    records = read_csv(args.data)
    last_issue = max(r.issue for r in records)
    seq = sorted(records, key=lambda x: x.issue)
    # TM clustering
    X_tm = [[1.0 if r.numbers[6] == n else 0.0 for n in range(1, 50)] for r in seq]
    assign_tm, _ = kmeans_cluster(X_tm, k=args.kc, iters=30)
    last_cluster = assign_tm[-1]
    tm_counts = Counter([seq[i].numbers[6] for i in range(len(seq)) if assign_tm[i] == last_cluster])
    total = sum(tm_counts.values()) or 1
    tm_ranked = sorted([(n, tm_counts[n] / total) for n in range(1, 50)], key=lambda x: (-x[1], x[0]))
    tm_pick = tm_ranked[0][0]
    # PM clustering
    X_pm = [[1.0 if n in r.numbers[:6] else 0.0 for n in range(1, 50)] for r in seq]
    assign_pm, centers_pm = kmeans_cluster(X_pm, k=args.kc, iters=30)
    last_c_pm = assign_pm[-1]
    center = centers_pm[last_c_pm]
    pm_ranked = sorted([(i + 1, center[i]) for i in range(49)], key=lambda x: (-x[1], x[0]))
    pm_picks = sorted([n for n, _ in pm_ranked[:args.k]])
    print("聚类预测")
    print(f"第{last_issue+1}期特碼(聚类)预测: {tm_pick}")
    print(f"特碼候选Top10(聚类): {tm_ranked[:10]}")
    print(f"聚类选取{args.k}个平码: {pm_picks}")
    print(f"平码候选Top{max(10, args.k)}(聚类): {pm_ranked[:max(10, args.k)]}")


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    s = e / np.sum(e, axis=1, keepdims=True)
    return s


def build_ml_dataset(records: List[IssueRecord], window: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    seq = sorted(records, key=lambda x: x.issue)
    window = max(1, min(window, len(seq) - 1))
    X: List[List[float]] = []
    y: List[int] = []
    for i in range(window, len(seq)):
        hist = seq[i - window:i]
        # features: counts of TM in last W (49), one-hot last TM (49), last issue PM membership (49)
        tm_counts = [0.0] * 49
        for r in hist:
            tm_counts[r.numbers[6] - 1] += 1.0
        tm_counts = [c / window for c in tm_counts]
        last_tm = seq[i - 1].numbers[6] - 1
        last_tm_onehot = [0.0] * 49
        last_tm_onehot[last_tm] = 1.0
        last_pm = [0.0] * 49
        for n in seq[i - 1].numbers[:6]:
            last_pm[n - 1] = 1.0
        feat = tm_counts + last_tm_onehot + last_pm
        X.append(feat)
        y.append(seq[i].numbers[6] - 1)
    return np.array(X, dtype=float), np.array(y, dtype=int)


def train_softmax_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.5, iters: int = 300, l2: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    n, d = X.shape
    classes = 49
    W = np.zeros((d, classes), dtype=float)
    b = np.zeros((1, classes), dtype=float)
    y_one = np.eye(classes)[y]
    for _ in range(iters):
        z = X.dot(W) + b  # (n, C)
        p = softmax(z)
        grad_W = X.T.dot(p - y_one) / n + l2 * W
        grad_b = np.sum(p - y_one, axis=0, keepdims=True) / n
        W -= lr * grad_W
        b -= lr * grad_b
    return W, b


def cmd_ml(args):
    if np is None:
        print("需要numpy支持，请先安装: pip install numpy")
        return
    records = read_csv(args.data)
    last_issue = max(r.issue for r in records)
    X, y = build_ml_dataset(records, window=args.window)
    if len(X) < 5:
        print("数据不足以训练机器学习模型")
        return
    # Train on all but last sample; predict next from last window
    W, b = train_softmax_regression(X, y, lr=args.lr, iters=args.iters, l2=args.l2)
    # Build feature for next issue
    seq = sorted(records, key=lambda x: x.issue)
    hist = seq[-args.window:]
    tm_counts = [0.0] * 49
    for r in hist:
        tm_counts[r.numbers[6] - 1] += 1.0
    tm_counts = [c / args.window for c in tm_counts]
    last_tm = seq[-1].numbers[6] - 1
    last_tm_onehot = [0.0] * 49
    last_tm_onehot[last_tm] = 1.0
    last_pm = [0.0] * 49
    for n in seq[-1].numbers[:6]:
        last_pm[n - 1] = 1.0
    x_next = np.array([tm_counts + last_tm_onehot + last_pm], dtype=float)
    p = softmax(x_next.dot(W) + b)[0]
    ranking = sorted([(i + 1, float(p[i])) for i in range(49)], key=lambda x: (-x[1], x[0]))
    tm_pick = ranking[0][0]
    print("机器学习预测（多类逻辑回归）")
    print(f"训练样本: {len(X)}，特征维: {X.shape[1]}，窗口: {args.window}")
    print(f"第{last_issue+1}期特碼(ML)预测: {tm_pick}")
    print(f"特碼候选Top10(ML 概率): {[(n, round(s,4)) for n,s in ranking[:10]]}")

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

    pb = sub.add_parser("balance", help="频率平衡预测（特码+平码）")
    pb.add_argument("--data", type=str, default="data/macao_2025.csv")
    pb.add_argument("--k", type=int, default=6, help="输出k个平衡平码")
    pb.add_argument("--recent-tm", dest="recent_tm", type=int, default=20, help="特码近期窗口上限")
    pb.add_argument("--recent-all", dest="recent_all", type=int, default=10, help="平码近期窗口上限")
    pb.set_defaults(func=cmd_balance)

    po = sub.add_parser("omit", help="遗漏值预测（特码+平码）")
    po.add_argument("--data", type=str, default="data/macao_2025.csv")
    po.add_argument("--k", type=int, default=6, help="输出k个平码")
    po.set_defaults(func=cmd_omit)

    pt = sub.add_parser("trend", help="趋势预测（线性趋势，特码+平码）")
    pt.add_argument("--data", type=str, default="data/macao_2025.csv")
    pt.add_argument("--k", type=int, default=6, help="输出k个平码")
    pt.add_argument("--window", type=int, default=60, help="趋势窗口期数")
    pt.set_defaults(func=cmd_trend)

    pa2 = sub.add_parser("assoc", help="关联性预测（转移/共现，特码+平码）")
    pa2.add_argument("--data", type=str, default="data/macao_2025.csv")
    pa2.add_argument("--k", type=int, default=6, help="输出k个平码")
    pa2.set_defaults(func=cmd_assoc)

    pc = sub.add_parser("cluster", help="聚类预测（KMeans，特码+平码）")
    pc.add_argument("--data", type=str, default="data/macao_2025.csv")
    pc.add_argument("--k", type=int, default=6, help="输出k个平码")
    pc.add_argument("--kc", type=int, default=4, help="聚类簇数")
    pc.set_defaults(func=cmd_cluster)

    pml = sub.add_parser("ml", help="机器学习预测（逻辑回归，特码）")
    pml.add_argument("--data", type=str, default="data/macao_2025.csv")
    pml.add_argument("--window", type=int, default=50, help="特征窗口")
    pml.add_argument("--lr", type=float, default=0.5, help="学习率")
    pml.add_argument("--iters", type=int, default=300, help="迭代轮数")
    pml.add_argument("--l2", type=float, default=1e-3, help="L2正则")
    pml.set_defaults(func=cmd_ml)
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
