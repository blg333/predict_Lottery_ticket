# -*- coding:utf-8 -*-
"""
End-to-end Macau LHC report:
- Crawl 2025 001-291 draws
- Analyze hot/cold, frequency, omissions
- Predict using multiple strategies
- Print detailed report to stdout

Usage:
  python run_lhc_report.py --year 2025 --start 1 --end 291 \
      --windows 291,120,80,50,15 --topk 7
"""
from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd
from loguru import logger

from macau_lhc_crawler import main as crawl_main
from lhc_strategies import (
    frequency_balance_prediction,
    omission_prediction,
    trend_prediction,
    correlation_prediction,
    clustering_prediction,
    machine_learning_prediction,
    summarize_results,
)
from lhc_meta import number_attributes


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    # Ensure correct ordering by issue
    df = df.sort_values(by=["期数"]).reset_index(drop=True)
    return df


essential_cols = ["期数", "开奖日期"] + [f"红球_{i}" for i in range(1, 7)] + ["蓝球"]


def window_hot_cold(df: pd.DataFrame, window: int, hot_k: int = 10, cold_k: int = 10):
    from lhc_strategies import frequency_counts
    freq = frequency_counts(df, window)
    hot = freq.sort_values(ascending=False).head(hot_k).index.tolist()
    cold = freq.sort_values(ascending=True).head(cold_k).index.tolist()
    return hot, cold, freq


def report(df: pd.DataFrame, windows: List[int], topk: int):
    print("==== 历史热门与冷门 ====")
    for w in windows:
        hot, cold, _ = window_hot_cold(df, w)
        print(f"窗口{w}期 热门: {hot[:topk]} 冷门: {cold[:topk]}")
    print()

    print("==== 多策略预测 ====")
    results = []
    results.append(frequency_balance_prediction(df, window=120, k=topk))
    results.append(omission_prediction(df, k=topk))
    results.append(trend_prediction(df, long_w=291, short_w=50, k=topk))
    results.append(correlation_prediction(df, ref_recent=5, k=topk))
    results.append(clustering_prediction(df, k_clusters=topk, k=topk))
    results.append(machine_learning_prediction(df, lags=15, k=topk))
    print(summarize_results(results))

    # Enrich attributes for ML picks if present
    for r in results:
        if r.name == "机器学习预测" and r.picks:
            print("\n机器学习预测属性(2025蛇年):")
            for n in sorted(r.picks):
                print(number_attributes(n))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=291)
    parser.add_argument("--windows", type=str, default="291,120,80,50,15")
    parser.add_argument("--dataset", type=str, default="data/lhc_macau/2025.csv")
    parser.add_argument("--topk", type=int, default=7)
    parser.add_argument("--crawl", action="store_true", help="force re-crawl before reporting")
    args = parser.parse_args()

    if args.crawl or not os.path.exists(args.dataset):
        # invoke crawler main
        crawl_main()

    df = load_dataset(args.dataset)
    # keep only essential columns if present
    present = [c for c in essential_cols if c in df.columns]
    df = df[present]

    windows = [int(x) for x in args.windows.split(",") if x.strip()]
    report(df, windows, args.topk)


if __name__ == "__main__":
    main()
