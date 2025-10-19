# -*- coding:utf-8 -*-
"""
Mark Six (六合彩) advanced statistics and exploratory analysis.

Reads `data/lhc/data.csv` (produced by get_data.py --name lhc) and computes:
- Per-number frequencies (1..49) for 正码(红球_1..6) and 特码(蓝球)
- Omission (since-last-seen) for each number
- Odd/Even, Big/Small distributions (Small: 1-24, Big: 25-49)
- Tail (last digit) distributions for 正码与特码
- Sum statistics of 正码与全号

Outputs a concise text summary and saves CSVs under `data/lhc/analysis/`.
"""

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from loguru import logger
from config import name_path, data_file_name
from lhc_meta import (
    get_wave_color_map,
    get_zodiac_map,
    get_number_zodiac,
)


def load_lhc_dataframe() -> pd.DataFrame:
    path = os.path.join(name_path["lhc"]["path"], data_file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Not found: {path}. Please run: python get_data.py --name lhc"
        )
    df = pd.read_csv(path)
    # Ensure expected columns exist
    expected_red = [f"红球_{i}" for i in range(1, 7)]
    expected_blue = ["蓝球"]
    missing = [c for c in expected_red + expected_blue if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df


def number_frequencies(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    red_cols = [f"红球_{i}" for i in range(1, 7)]
    blue_col = "蓝球"
    # Flatten and count
    red_values = df[red_cols].values.reshape(-1)
    blue_values = df[blue_col].values.reshape(-1)
    red_values = pd.to_numeric(pd.Series(red_values), errors="coerce").dropna().astype(int)
    blue_values = pd.to_numeric(pd.Series(blue_values), errors="coerce").dropna().astype(int)
    red_freq = red_values.value_counts().reindex(range(1, 50), fill_value=0).sort_index()
    blue_freq = blue_values.value_counts().reindex(range(1, 50), fill_value=0).sort_index()
    return red_freq, blue_freq


def omissions_since_last(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    red_cols = [f"红球_{i}" for i in range(1, 7)]
    blue_col = "蓝球"
    # Use DataFrame order as saved (historical order). Compute from end backwards.
    red_matrix = df[red_cols].apply(pd.to_numeric, errors="coerce").values
    blue_array = pd.to_numeric(df[blue_col], errors="coerce").values

    red_omission: Dict[int, int] = {}
    blue_omission: Dict[int, int] = {}

    # For red (any of the 6 positions counts as seen)
    for num in range(1, 50):
        # find last index from bottom where num appears in any red column
        last_idx = None
        for idx in range(len(red_matrix) - 1, -1, -1):
            if num in red_matrix[idx, :]:
                last_idx = idx
                break
        red_omission[num] = (len(red_matrix) - 1 - last_idx) if last_idx is not None else len(red_matrix)

    # For blue (single position)
    for num in range(1, 50):
        last_idx = None
        for idx in range(len(blue_array) - 1, -1, -1):
            if blue_array[idx] == num:
                last_idx = idx
                break
        blue_omission[num] = (len(blue_array) - 1 - last_idx) if last_idx is not None else len(blue_array)

    red_omit_s = pd.Series(red_omission).sort_index()
    blue_omit_s = pd.Series(blue_omission).sort_index()
    return red_omit_s, blue_omit_s


def odd_even_big_small(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    red_cols = [f"红球_{i}" for i in range(1, 7)]
    blue_col = "蓝球"
    red_vals = pd.to_numeric(df[red_cols].values.reshape(-1), errors="coerce").dropna().astype(int)
    blue_vals = pd.to_numeric(df[blue_col], errors="coerce").dropna().astype(int)

    def stats(values: pd.Series) -> Dict[str, int]:
        even = int((values % 2 == 0).sum())
        odd = int((values % 2 != 0).sum())
        small = int((values <= 24).sum())
        big = int((values >= 25).sum())
        return {"odd": odd, "even": even, "small": small, "big": big}

    return {"red": stats(red_vals), "blue": stats(blue_vals)}


def tail_distribution(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    red_cols = [f"红球_{i}" for i in range(1, 7)]
    blue_col = "蓝球"
    red_vals = pd.to_numeric(df[red_cols].values.reshape(-1), errors="coerce").dropna().astype(int)
    blue_vals = pd.to_numeric(df[blue_col], errors="coerce").dropna().astype(int)
    red_tail = (red_vals % 10).value_counts().reindex(range(0, 10), fill_value=0).sort_index()
    blue_tail = (blue_vals % 10).value_counts().reindex(range(0, 10), fill_value=0).sort_index()
    return red_tail, blue_tail


def sum_statistics(df: pd.DataFrame) -> Dict[str, float]:
    red_cols = [f"红球_{i}" for i in range(1, 7)]
    blue_col = "蓝球"
    red_sum = pd.to_numeric(df[red_cols].sum(axis=1), errors="coerce").dropna().astype(float)
    all_sum = pd.to_numeric(df[red_cols + [blue_col]].sum(axis=1), errors="coerce").dropna().astype(float)
    return {
        "red_sum_mean": float(red_sum.mean()),
        "red_sum_median": float(red_sum.median()),
        "red_sum_min": float(red_sum.min()),
        "red_sum_max": float(red_sum.max()),
        "all_sum_mean": float(all_sum.mean()),
        "all_sum_median": float(all_sum.median()),
        "all_sum_min": float(all_sum.min()),
        "all_sum_max": float(all_sum.max()),
    }


def save_outputs(output_dir: str,
                 red_freq: pd.Series, blue_freq: pd.Series,
                 red_omit: pd.Series, blue_omit: pd.Series,
                 odd_even_stats: Dict[str, Dict[str, int]],
                 red_tail: pd.Series, blue_tail: pd.Series,
                 sum_stats: Dict[str, float],
                 wave_dist_red: pd.Series, wave_dist_blue: pd.Series,
                 zodiac_dist_red: pd.Series, zodiac_dist_blue: pd.Series) -> None:
    os.makedirs(output_dir, exist_ok=True)
    red_freq.to_csv(os.path.join(output_dir, "red_frequency.csv"), header=["count"])
    blue_freq.to_csv(os.path.join(output_dir, "blue_frequency.csv"), header=["count"])
    red_omit.to_csv(os.path.join(output_dir, "red_omission.csv"), header=["omission"])
    blue_omit.to_csv(os.path.join(output_dir, "blue_omission.csv"), header=["omission"])
    red_tail.to_csv(os.path.join(output_dir, "red_tail_distribution.csv"), header=["count"])
    blue_tail.to_csv(os.path.join(output_dir, "blue_tail_distribution.csv"), header=["count"])
    wave_dist_red.to_csv(os.path.join(output_dir, "red_wave_distribution.csv"), header=["count"])
    wave_dist_blue.to_csv(os.path.join(output_dir, "blue_wave_distribution.csv"), header=["count"])
    zodiac_dist_red.to_csv(os.path.join(output_dir, "red_zodiac_distribution.csv"), header=["count"])
    zodiac_dist_blue.to_csv(os.path.join(output_dir, "blue_zodiac_distribution.csv"), header=["count"])

    # Save a compact JSON summary
    summary = {
        "odd_even": odd_even_stats,
        "sum_stats": sum_stats,
        "top10_red_most_frequent": red_freq.sort_values(ascending=False).head(10).to_dict(),
        "top10_blue_most_frequent": blue_freq.sort_values(ascending=False).head(10).to_dict(),
        "top10_red_largest_omission": red_omit.sort_values(ascending=False).head(10).to_dict(),
        "top10_blue_largest_omission": blue_omit.sort_values(ascending=False).head(10).to_dict(),
        "wave_top_red": wave_dist_red.sort_values(ascending=False).to_dict(),
        "wave_top_blue": wave_dist_blue.sort_values(ascending=False).to_dict(),
        "zodiac_top_red": zodiac_dist_red.sort_values(ascending=False).to_dict(),
        "zodiac_top_blue": zodiac_dist_blue.sort_values(ascending=False).to_dict(),
    }
    pd.Series(summary).to_json(os.path.join(output_dir, "summary.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="lhc", choices=["lhc"], help="仅支持六合彩分析")
    args = parser.parse_args()
    if args.name != "lhc":
        raise ValueError("仅支持 --name lhc")

    df = load_lhc_dataframe()
    red_freq, blue_freq = number_frequencies(df)
    red_omit, blue_omit = omissions_since_last(df)
    odd_even_stats = odd_even_big_small(df)
    red_tail, blue_tail = tail_distribution(df)
    sums = sum_statistics(df)

    # 波色分布
    wave_map = get_wave_color_map()
    red_cols = [f"红球_{i}" for i in range(1, 7)]
    blue_col = "蓝球"
    red_vals = pd.to_numeric(df[red_cols].values.reshape(-1), errors="coerce").dropna().astype(int)
    blue_vals = pd.to_numeric(df[blue_col], errors="coerce").dropna().astype(int)
    red_wave = red_vals.map(lambda n: wave_map.get(int(n), "未知"))
    blue_wave = blue_vals.map(lambda n: wave_map.get(int(n), "未知"))
    wave_dist_red = red_wave.value_counts().reindex(["红波", "蓝波", "绿波", "未知"], fill_value=0)
    wave_dist_blue = blue_wave.value_counts().reindex(["红波", "蓝波", "绿波", "未知"], fill_value=0)

    # 生肖分布（若拉取失败则填充未知）
    zodiac_map = get_zodiac_map(use_cache=True)
    if zodiac_map is None:
        red_zodiac = red_vals.map(lambda _: "未知")
        blue_zodiac = blue_vals.map(lambda _: "未知")
        zodiac_order = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪", "未知"]
    else:
        red_zodiac = red_vals.map(lambda n: zodiac_map.get(int(n), "未知"))
        blue_zodiac = blue_vals.map(lambda n: zodiac_map.get(int(n), "未知"))
        zodiac_order = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪", "未知"]
    zodiac_dist_red = red_zodiac.value_counts().reindex(zodiac_order, fill_value=0)
    zodiac_dist_blue = blue_zodiac.value_counts().reindex(zodiac_order, fill_value=0)

    output_dir = os.path.join(name_path["lhc"]["path"], "analysis")
    save_outputs(
        output_dir,
        red_freq, blue_freq,
        red_omit, blue_omit,
        odd_even_stats,
        red_tail, blue_tail,
        sums,
        wave_dist_red, wave_dist_blue,
        zodiac_dist_red, zodiac_dist_blue,
    )

    logger.info("保存统计输出到: {}".format(output_dir))
    logger.info("Top5 正码频率: {}".format(red_freq.sort_values(ascending=False).head(5).to_dict()))
    logger.info("Top5 特码频率: {}".format(blue_freq.sort_values(ascending=False).head(5).to_dict()))
    logger.info("最大遗漏(正码)Top5: {}".format(red_omit.sort_values(ascending=False).head(5).to_dict()))
    logger.info("最大遗漏(特码)Top5: {}".format(blue_omit.sort_values(ascending=False).head(5).to_dict()))
    logger.info("奇偶/大小: {}".format(odd_even_stats))
    logger.info("和值统计: {}".format(sums))
    logger.info("波色分布(红): {}".format(wave_dist_red.to_dict()))
    logger.info("波色分布(蓝): {}".format(wave_dist_blue.to_dict()))
    logger.info("生肖分布(红): {}".format(zodiac_dist_red.to_dict()))
    logger.info("生肖分布(蓝): {}".format(zodiac_dist_blue.to_dict()))


if __name__ == "__main__":
    main()
