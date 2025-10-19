# -*- coding:utf-8 -*-
"""
Macau LHC analysis and prediction strategies with chat-style reporting.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger
try:
    from sklearn.cluster import KMeans  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    KMeans = None  # type: ignore

from config import name_path, data_file_name


def load_lhc_data() -> pd.DataFrame:
    path = f"{name_path['lhc']['path']}{data_file_name}"
    df = pd.read_csv(path)
    for c in [f"红球_{i}" for i in range(1, 7)] + ["蓝球"]:
        df[c] = df[c].astype(int)
    df = df.sort_values(by="期数")
    return df


def numbers_by_row(df: pd.DataFrame) -> List[List[int]]:
    rows: List[List[int]] = []
    for _, r in df.iterrows():
        nums = [int(r[f"红球_{i}"]) for i in range(1, 7)] + [int(r["蓝球"])]
        rows.append(nums)
    return rows


def frequency_stats(rows: List[List[int]], window: int | None = None) -> Counter:
    data = rows if window is None else rows[-window:]
    cnt: Counter = Counter()
    for arr in data:
        cnt.update(arr)
    return cnt


def omission_stats(rows: List[List[int]]) -> Dict[int, int]:
    last_seen = {n: None for n in range(1, 50)}
    for idx, arr in enumerate(rows, start=1):
        for n in arr:
            last_seen[n] = idx
    total = len(rows)
    omission = {n: (total - last_seen[n]) if last_seen[n] is not None else total for n in range(1, 50)}
    return omission


def trend_stats(rows: List[List[int]], windows: List[int]) -> Dict[int, Dict[int, int]]:
    res: Dict[int, Dict[int, int]] = {}
    for w in windows:
        res[w] = dict(frequency_stats(rows, w))
    return res


def cooccurrence_matrix(rows: List[List[int]]) -> np.ndarray:
    size = 50
    mat = np.zeros((size, size), dtype=int)
    for arr in rows:
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                a, b = arr[i], arr[j]
                mat[a, b] += 1
                mat[b, a] += 1
    return mat


def kmeans_cluster_numbers(freq: Counter, k: int = 6) -> Dict[int, int]:
    xs = np.array([[n, freq.get(n, 0)] for n in range(1, 50)], dtype=float)
    if len(np.unique(xs[:, 1])) <= 1:
        return {int(n): int((n - 1) % k) for n in range(1, 50)}
    if KMeans is None:
        # Fallback: bucket by frequency quantiles
        counts = xs[:, 1]
        quantiles = np.quantile(counts, np.linspace(0, 1, k + 1))
        labels = np.digitize(counts, quantiles[1:-1], right=True)
        return {int(n): int(lbl) for (n, _), lbl in zip(xs, labels)}
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(xs)
    return {int(n): int(lbl) for (n, _), lbl in zip(xs, labels)}


def transition_matrix(rows: List[List[int]]) -> np.ndarray:
    size = 50
    mat = np.zeros((size, size), dtype=int)
    for t in range(len(rows) - 1):
        a_set = rows[t]
        b_set = rows[t + 1]
        for a in a_set:
            for b in b_set:
                mat[a, b] += 1
    return mat


def pick_frequency_balance(freq: Counter, pick_count: int = 7) -> List[int]:
    pools = [range(1, 17), range(17, 33), range(33, 50)]
    chosen: List[int] = []
    for pool in pools:
        candidates = sorted(pool, key=lambda n: (-freq.get(n, 0), n))
        for c in candidates:
            if c not in chosen:
                chosen.append(c)
            if len(chosen) >= pick_count:
                break
        if len(chosen) >= pick_count:
            break
    rest = sorted(range(1, 50), key=lambda n: (-freq.get(n, 0), n))
    for n in rest:
        if n not in chosen:
            chosen.append(n)
        if len(chosen) >= pick_count:
            break
    return chosen[:pick_count]


def pick_omission(omission: Dict[int, int], pick_count: int = 7) -> List[int]:
    return [n for n, _ in sorted(omission.items(), key=lambda kv: (-kv[1], kv[0]))[:pick_count]]


def pick_trend(trend: Dict[int, Dict[int, int]], window: int, pick_count: int = 7) -> List[int]:
    freq = trend.get(window, {})
    return [n for n, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:pick_count]]


def pick_association(mat: np.ndarray, seed: List[int], pick_count: int = 7) -> List[int]:
    score = np.sum(mat[seed, :], axis=0)
    order = np.argsort(-score)
    res: List[int] = []
    for n in order:
        if n == 0:
            continue
        if n in seed:
            continue
        res.append(int(n))
        if len(res) >= pick_count:
            break
    return res


def pick_cluster(freq: Counter, k: int = 6, pick_count: int = 7) -> List[int]:
    labels = kmeans_cluster_numbers(freq, k)
    by_cluster: Dict[int, List[int]] = defaultdict(list)
    for n in range(1, 50):
        by_cluster[labels[n]].append(n)
    chosen: List[int] = []
    for cid, nums in sorted(by_cluster.items()):
        nums_sorted = sorted(nums, key=lambda n: (-freq.get(n, 0), n))
        if nums_sorted:
            chosen.append(nums_sorted[0])
        if len(chosen) >= pick_count:
            break
    rest = sorted(range(1, 50), key=lambda n: (-freq.get(n, 0), n))
    for n in rest:
        if n not in chosen:
            chosen.append(n)
        if len(chosen) >= pick_count:
            break
    return chosen[:pick_count]


def build_report(df: pd.DataFrame) -> str:
    rows = numbers_by_row(df)
    freq_all = frequency_stats(rows)
    omission = omission_stats(rows)
    trend = trend_stats(rows, [291, 120, 80, 50, 15])
    mat = cooccurrence_matrix(rows)
    tmat = transition_matrix(rows)

    pick_freq = pick_frequency_balance(freq_all)
    pick_omit = pick_omission(omission)
    pick_trend_291 = pick_trend(trend, 291)
    pick_trend_120 = pick_trend(trend, 120)
    pick_trend_80 = pick_trend(trend, 80)
    pick_trend_50 = pick_trend(trend, 50)
    pick_trend_15 = pick_trend(trend, 15)
    pick_assoc = pick_association(mat, seed=pick_freq[:3])
    pick_cluster_nums = pick_cluster(freq_all)

    hot = [n for n, _ in freq_all.most_common(10)]
    cold = [n for n, _ in sorted(freq_all.items(), key=lambda kv: (kv[1], kv[0]))[:10]]

    parts = []
    parts.append("【热门号码】:" + ",".join(f"{x:02d}" for x in hot))
    parts.append("【冷门号码】:" + ",".join(f"{x:02d}" for x in cold))
    parts.append("【频率平衡预测】:" + ",".join(f"{n:02d}" for n in pick_freq))
    parts.append("【遗漏值预测】:" + ",".join(f"{n:02d}" for n in pick_omit))
    parts.append("【291期趋势预测】:" + ",".join(f"{n:02d}" for n in pick_trend_291))
    parts.append("【120期趋势预测】:" + ",".join(f"{n:02d}" for n in pick_trend_120))
    parts.append("【80期趋势预测】:" + ",".join(f"{n:02d}" for n in pick_trend_80))
    parts.append("【50期趋势预测】:" + ",".join(f"{n:02d}" for n in pick_trend_50))
    parts.append("【15期趋势预测】:" + ",".join(f"{n:02d}" for n in pick_trend_15))
    # Transition-based suggestion: from last period
    last = rows[-1]
    score_next = np.sum(tmat[last, :], axis=0)
    order = np.argsort(-score_next)
    pick_trans = [int(n) for n in order if n != 0 and n not in last][:7]
    parts.append("【转移矩阵预测】:" + ",".join(f"{n:02d}" for n in pick_trans))
    parts.append("【关联性预测】:" + ",".join(f"{n:02d}" for n in pick_assoc))
    parts.append("【聚类预测】:" + ",".join(f"{n:02d}" for n in pick_cluster_nums))
    return "\n".join(parts)


def run() -> str:
    df = load_lhc_data()
    return build_report(df)


if __name__ == "__main__":
    print(run())
