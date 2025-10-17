# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from loguru import logger

from .config import CURRENT_YEAR, RECO_CONFIG
from .storage import load_draws, save_analysis_history


def compute_frequencies(draws: List[Tuple[int, int, List[int]]]) -> Dict[int, int]:
    counter: Counter[int] = Counter()
    for _, _, numbers in draws:
        counter.update(numbers)
    return dict(counter)


def hot_cold(frequencies: Dict[int, int]) -> Tuple[List[int], List[int]]:
    items = sorted(frequencies.items(), key=lambda kv: (-kv[1], kv[0]))
    hot = [n for n, _ in items[: RECO_CONFIG.top_hot_count]]
    cold = [n for n, _ in items[-RECO_CONFIG.bottom_cold_exclude :]]
    return hot, cold


def transition_matrix(draws: List[Tuple[int, int, List[int]]]) -> Dict[int, Dict[int, int]]:
    trans: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    # For each adjacent issue, count transitions from each number to next issue numbers
    for i in range(len(draws) - 1):
        _, _, a = draws[i]
        _, _, b = draws[i + 1]
        for x in a:
            for y in b:
                trans[x][y] += 1
    # convert nested defaultdict to normal dict
    return {x: dict(y) for x, y in trans.items()}


def backtest_recommendations(draws: List[Tuple[int, int, List[int]]], window: int) -> Dict[str, float]:
    if len(draws) < window + 1:
        return {"issues": float(len(draws)), "avg_hits": 0.0}
    hits = 0
    count = 0
    for i in range(window, len(draws)):
        recent = draws[i - window : i]
        freq = compute_frequencies(recent)
        hot, cold = hot_cold(freq)
        next_numbers = set(draws[i][2])
        reco = [n for n in hot if n not in set(cold)]
        hit = len(next_numbers.intersection(reco))
        hits += hit
        count += 1
    return {"issues": float(count), "avg_hits": hits / max(count, 1)}


def make_recommendation(year: int = CURRENT_YEAR, start_issue: int = 1, end_issue: int = 289) -> Dict[str, object]:
    draws = load_draws(year, start_issue, end_issue)
    if not draws:
        return {"error": "no draws"}
    recent = draws[max(0, len(draws) - RECO_CONFIG.recent_window) :]
    freq = compute_frequencies(recent)
    hot, cold = hot_cold(freq)
    recommended = [n for n in hot if n not in set(cold)]
    summary = {
        "year": year,
        "start_issue": start_issue,
        "end_issue": end_issue,
        "hot": hot,
        "cold": cold,
        "recommended": recommended[:7],
        "frequencies": freq,
    }
    save_analysis_history(year, start_issue, end_issue, "frequency_hot_cold", json.dumps(summary, ensure_ascii=False))
    return summary
