# -*- coding:utf-8 -*-
"""
Multi-strategy prediction toolkit for Macau Mark Six (六合彩):
- Frequency balance prediction
- Omission (miss) prediction
- Trend prediction (momentum)
- Correlation (co-occurrence) prediction
- Clustering-based diversification
- Machine learning prediction (logistic regression per-number)

Input dataset format: columns [期数, 开奖日期, 红球_1..6, 蓝球]
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

try:
    from sklearn.cluster import KMeans  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
except Exception:
    KMeans = None  # type: ignore
    LogisticRegression = None  # type: ignore

from lhc_meta import get_wave_color_map, get_zodiac_map

RED_COLS = [f"红球_{i}" for i in range(1, 7)]
BLUE_COL = "蓝球"
ALL_COLS = RED_COLS + [BLUE_COL]


@dataclass
class StrategyResult:
    name: str
    picks: List[int]
    rationale: str


def _flatten_numbers(df: pd.DataFrame) -> pd.Series:
    vals = pd.to_numeric(df[ALL_COLS].values.reshape(-1), errors="coerce").dropna().astype(int)
    return vals


def recent_window(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if window <= 0 or window > len(df):
        window = len(df)
    return df.tail(window)


def frequency_counts(df: pd.DataFrame, window: int) -> pd.Series:
    sub = recent_window(df, window)
    vals = _flatten_numbers(sub)
    return vals.value_counts().reindex(range(1, 50), fill_value=0).sort_index()


def omission_counts(df: pd.DataFrame) -> pd.Series:
    # distance since last seen across entire dataset
    vals = _flatten_numbers(df).tolist()
    last_pos: Dict[int, int] = {n: None for n in range(1, 50)}  # type: ignore
    for idx, v in enumerate(vals):
        last_pos[v] = idx
    total = len(vals)
    omission = {n: (total - 1 - last_pos[n]) if last_pos[n] is not None else total for n in range(1, 50)}
    return pd.Series(omission).sort_index()


def frequency_balance_prediction(df: pd.DataFrame, window: int = 120, k: int = 7) -> StrategyResult:
    freq = frequency_counts(df, window)
    # Aim to pick near-median frequency numbers to balance
    median = freq.median()
    scores = (freq - median).abs()
    candidates = scores.sort_values().index.tolist()
    picks = candidates[:k]
    return StrategyResult(
        name="频率平衡预测",
        picks=picks,
        rationale=f"窗口{window}期，选择接近中位频率的号码以平衡覆盖"
    )


def omission_prediction(df: pd.DataFrame, k: int = 7) -> StrategyResult:
    om = omission_counts(df)
    picks = om.sort_values(ascending=False).head(k).index.tolist()
    return StrategyResult(
        name="遗漏值预测",
        picks=picks,
        rationale="选择遗漏值最大的号码，期望回补"
    )


def trend_prediction(df: pd.DataFrame, long_w: int = 291, short_w: int = 50, k: int = 7) -> StrategyResult:
    long_f = frequency_counts(df, long_w)
    short_f = frequency_counts(df, short_w)
    trend = short_f - long_f * (short_w / max(long_w, 1))
    picks = trend.sort_values(ascending=False).head(k).index.tolist()
    return StrategyResult(
        name="趋势预测",
        picks=picks,
        rationale=f"短期({short_w})相对长期({long_w})上升趋势较强"
    )


def cooccurrence_matrix(df: pd.DataFrame) -> np.ndarray:
    # 49x49 matrix of co-occur counts across draws
    mat = np.zeros((49, 49), dtype=np.float64)
    for _, row in df.iterrows():
        nums = [int(row[c]) for c in ALL_COLS if pd.notna(row[c])]
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                a, b = nums[i] - 1, nums[j] - 1
                mat[a, b] += 1
                mat[b, a] += 1
    # normalize per-row
    row_sums = mat.sum(axis=1, keepdims=True) + 1e-9
    mat = mat / row_sums
    return mat


def correlation_prediction(df: pd.DataFrame, ref_recent: int = 5, k: int = 7) -> StrategyResult:
    mat = cooccurrence_matrix(df)
    recent = recent_window(df, ref_recent)
    current_pool = set()
    for _, row in recent.iterrows():
        for c in ALL_COLS:
            if pd.notna(row[c]):
                current_pool.add(int(row[c]))
    score = np.zeros(49)
    for n in current_pool:
        score += mat[n - 1]
    picks = list(np.argsort(score)[::-1][:k] + 1)
    return StrategyResult(
        name="关联性预测",
        picks=picks,
        rationale=f"与最近{ref_recent}期号码共现概率较高"
    )


def clustering_prediction(df: pd.DataFrame, k_clusters: int = 7, k: int = 7) -> StrategyResult:
    freq = frequency_counts(df, window=len(df))
    om = omission_counts(df)
    wave = get_wave_color_map()
    zodiac = get_zodiac_map(use_cache=True) or {}

    # feature: [freq_norm, omit_norm] + wave one-hot(3) + tail one-hot(10) + zodiac one-hot(12)
    X = []
    idx_to_num = []
    for n in range(1, 50):
        f = freq.loc[n]
        o = om.loc[n]
        wave_onehot = [0, 0, 0]
        w = {"红波": 0, "蓝波": 1, "绿波": 2}.get(wave.get(n, "未知"), None)
        if w is not None:
            wave_onehot[w] = 1
        tail = n % 10
        tail_onehot = [1 if i == tail else 0 for i in range(10)]
        z_names = ["鼠","牛","虎","兔","龙","蛇","马","羊","猴","鸡","狗","猪"]
        z = zodiac.get(n, None)
        z_onehot = [1 if name == z else 0 for name in z_names]
        X.append([f, o] + wave_onehot + tail_onehot + z_onehot)
        idx_to_num.append(n)
    X = np.array(X, dtype=np.float64)
    # normalize freq/omit
    X[:, 0] = (X[:, 0] - X[:, 0].mean()) / (X[:, 0].std() + 1e-9)
    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / (X[:, 1].std() + 1e-9)

    if KMeans is None:
        # fallback: pick spread across quantiles of omission
        order = om.sort_values(ascending=False).index.tolist()
        step = max(1, len(order) // k)
        picks = [order[i] for i in range(0, len(order), step)][:k]
        return StrategyResult("聚类预测", picks, "降级为遗漏分层抽样（缺少sklearn）")

    km = KMeans(n_clusters=k_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    # pick closest to center for each cluster
    picks: List[int] = []
    for c in range(k_clusters):
        idxs = np.where(labels == c)[0]
        if len(idxs) == 0:
            continue
        center = km.cluster_centers_[c]
        best = min(idxs, key=lambda i: np.linalg.norm(X[i] - center))
        picks.append(idx_to_num[best])
    picks = picks[:k]
    return StrategyResult(
        name="聚类预测",
        picks=picks,
        rationale=f"基于号码多维特征聚类，选取各簇中心代表"
    )


def build_ml_training(df: pd.DataFrame, lags: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    # Build per-draw features: counts of numbers in last lags across attributes
    wave = get_wave_color_map()
    zodiac = get_zodiac_map(use_cache=True) or {}

    features = []
    labels = []
    series = df[ALL_COLS].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    for i in range(lags, len(series)):
        window = series.iloc[i - lags:i]
        nums = window.values.reshape(-1)
        # frequencies over window
        freq = np.bincount(nums, minlength=50)[1:]
        # wave counts
        wave_counts = np.zeros(3)
        for n in range(1, 50):
            cnt = freq[n - 1]
            w = {"红波": 0, "蓝波": 1, "绿波": 2}.get(wave.get(n, "未知"), None)
            if w is not None:
                wave_counts[w] += cnt
        # zodiac counts
        z_counts = np.zeros(12)
        z_names = ["鼠","牛","虎","兔","龙","蛇","马","羊","猴","鸡","狗","猪"]
        for n in range(1, 50):
            cnt = freq[n - 1]
            z = zodiac.get(n, None)
            if z is not None:
                z_counts[z_names.index(z)] += cnt
        x = np.concatenate([
            freq / (freq.sum() + 1e-9),
            wave_counts / (wave_counts.sum() + 1e-9),
            z_counts / (z_counts.sum() + 1e-9),
        ])
        features.append(x)
        # label: numbers in current draw
        y = np.zeros(49)
        curr = series.iloc[i].values
        for v in curr:
            if 1 <= v <= 49:
                y[v - 1] = 1
        labels.append(y)
    X = np.array(features, dtype=np.float64)
    Y = np.array(labels, dtype=np.float64)
    return X, Y


def machine_learning_prediction(df: pd.DataFrame, lags: int = 15, k: int = 7) -> StrategyResult:
    X, Y = build_ml_training(df, lags=lags)
    if len(X) < 5:
        return StrategyResult("机器学习预测", [], "样本不足")
    if LogisticRegression is None:
        return StrategyResult("机器学习预测", [], "缺少sklearn，无法训练")
    # fit one-vs-rest logistic regression across 49 outputs
    models = []
    for j in range(49):
        lr = LogisticRegression(max_iter=1000, solver="liblinear")
        lr.fit(X[:-1], Y[:-1, j])
        models.append(lr)
    probs = np.array([m.predict_proba(X[-1:].reshape(1, -1))[:, 1][0] for m in models])
    picks = list(np.argsort(probs)[::-1][:k] + 1)
    return StrategyResult(
        name="机器学习预测",
        picks=picks,
        rationale=f"Logistic回归基于最近{lags}期特征的概率排序"
    )


def summarize_results(results: List[StrategyResult]) -> str:
    lines = []
    for r in results:
        lines.append(f"【{r.name}】")
        lines.append(f"预测: {sorted(r.picks)}")
        lines.append(f"说明: {r.rationale}")
    return "\n".join(lines)
