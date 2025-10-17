# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from collections import Counter, defaultdict
from itertools import combinations
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


def compute_decayed_frequencies(draws: List[Tuple[int, int, List[int]]], decay: float = 0.97) -> Dict[int, float]:
    """Exponential recency decay for frequencies (newer issues get higher weight)."""
    if not draws:
        return {}
    weights: Dict[int, float] = defaultdict(float)
    last_idx = len(draws) - 1
    for i, (_, _, nums) in enumerate(draws):
        w = decay ** (last_idx - i)
        for n in nums:
            weights[n] += w
    return dict(weights)


def compute_cooccurrence(draws: List[Tuple[int, int, List[int]]]) -> Dict[Tuple[int, int], int]:
    """Co-occurrence counts for number pairs within the same issue (orderless)."""
    co: Dict[Tuple[int, int], int] = defaultdict(int)
    for _, _, nums in draws:
        unique = sorted(set(nums))
        for a, b in combinations(unique, 2):
            key = (a, b) if a < b else (b, a)
            co[key] += 1
    return dict(co)


def _normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    mx = max(scores.values())
    if mx <= 0:
        return {k: 0.0 for k in scores}
    return {k: v / mx for k, v in scores.items()}


def _candidate_universe() -> List[int]:
    return list(range(1, 50))


def recommend_advanced(
    context_draws: List[Tuple[int, int, List[int]]],
    prev_issues: List[Tuple[int, int, List[int]]],
    weights: Dict[str, float],
    top_k: int = 7,
    decay: float = 0.97,
) -> List[int]:
    """Advanced scoring combining plain frequency, decayed frequency, co-occurrence and transitions."""
    freq = compute_frequencies(context_draws)
    dec = compute_decayed_frequencies(context_draws, decay=decay)
    co = compute_cooccurrence(context_draws)
    trans = transition_matrix(context_draws)

    # Anchors for co-occurrence and transition: use numbers from last 1-3 issues
    anchors: List[int] = []
    for _, _, nums in prev_issues:
        anchors.extend(nums)
    anchor_set = sorted(set(anchors))
    prev_set = set(prev_issues[-1][2]) if prev_issues else set()

    # Normalize components
    freq_n = _normalize_scores({n: freq.get(n, 0) for n in _candidate_universe()})
    dec_n = _normalize_scores({n: dec.get(n, 0.0) for n in _candidate_universe()})

    co_scores: Dict[int, float] = {}
    for n in _candidate_universe():
        s = 0.0
        for a in anchor_set:
            key = (a, n) if a < n else (n, a)
            s += co.get(key, 0)
        co_scores[n] = s
    co_n = _normalize_scores(co_scores)

    trans_scores: Dict[int, float] = {}
    for n in _candidate_universe():
        s = 0.0
        for a in prev_set:
            s += trans.get(a, {}).get(n, 0)
        trans_scores[n] = s
    trans_n = _normalize_scores(trans_scores)

    w_freq = float(weights.get("freq", 1.0))
    w_dec = float(weights.get("decay", 0.5))
    w_co = float(weights.get("cooc", 0.3))
    w_tr = float(weights.get("trans", 0.3))

    final: Dict[int, float] = {}
    for n in _candidate_universe():
        final[n] = (
            w_freq * freq_n.get(n, 0.0)
            + w_dec * dec_n.get(n, 0.0)
            + w_co * co_n.get(n, 0.0)
            + w_tr * trans_n.get(n, 0.0)
        )
    # Select top-k
    ranked = sorted(final.items(), key=lambda kv: (-kv[1], kv[0]))
    return [n for n, _ in ranked[:top_k]]


def backtest_advanced(
    draws: List[Tuple[int, int, List[int]]],
    start_issue: int,
    end_issue: int,
    window: int,
    prev_k: int,
    weights: Dict[str, float],
    decay: float = 0.97,
) -> Dict[str, object]:
    issue_to_numbers = {iss: nums for _, iss, nums in draws}
    results = []
    for iss in range(start_issue, end_issue + 1):
        prev_start = max(1, iss - window)
        context = [(y, i, n) for (y, i, n) in draws if prev_start <= i <= iss - 1]
        prev_context = [(y, i, n) for (y, i, n) in draws if iss - prev_k <= i <= iss - 1]
        if len(context) == 0 or iss not in issue_to_numbers or len(prev_context) == 0:
            continue
        reco = recommend_advanced(context, prev_context, weights=weights, top_k=7, decay=decay)
        actual = set(issue_to_numbers[iss])
        hit = len(actual.intersection(reco))
        results.append({
            "issue": iss,
            "recommended": reco,
            "actual": list(issue_to_numbers[iss]),
            "hits": hit,
        })
    summary = {
        "range": [start_issue, end_issue],
        "window": window,
        "prev_k": prev_k,
        "weights": weights,
        "decay": decay,
        "avg_hits": (sum(r["hits"] for r in results) / max(len(results), 1)) if results else 0.0,
        "count": len(results),
        "details": results,
    }
    return summary


def optimize_grid(
    year: int,
    start_issue: int,
    end_issue: int,
    windows: List[int] = None,
    prev_ks: List[int] = None,
    decay_list: List[float] = None,
    weight_grid: List[Dict[str, float]] = None,
) -> Dict[str, object]:
    from .storage import load_draws
    draws = load_draws(year, 1, end_issue)
    if windows is None:
        windows = [40, 50, 60]
    if prev_ks is None:
        prev_ks = [1, 2, 3]
    if decay_list is None:
        decay_list = [0.95, 0.97, 0.99]
    if weight_grid is None:
        weight_grid = [
            {"freq": 1.0, "decay": 0.5, "cooc": 0.3, "trans": 0.3},
            {"freq": 1.0, "decay": 1.0, "cooc": 0.5, "trans": 0.5},
            {"freq": 0.8, "decay": 1.2, "cooc": 0.6, "trans": 0.4},
            {"freq": 0.6, "decay": 1.2, "cooc": 0.8, "trans": 0.8},
        ]
    best = None
    best_score = -1.0
    best_summary = None
    for w in windows:
        for pk in prev_ks:
            for dec in decay_list:
                for wg in weight_grid:
                    s = backtest_advanced(draws, start_issue, end_issue, w, pk, wg, decay=dec)
                    if s["avg_hits"] > best_score:
                        best_score = s["avg_hits"]
                        best = {"window": w, "prev_k": pk, "decay": dec, "weights": wg}
                        best_summary = s
    payload = {
        "best": best,
        "best_avg_hits": best_score,
    }
    save_analysis_history(year, start_issue, end_issue, "optimize_grid", json.dumps(payload, ensure_ascii=False))
    return {"config": best, "summary": best_summary}


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


def accuracy_backtest_by_issue(year: int, start_issue: int, end_issue: int, window: int = RECO_CONFIG.recent_window) -> Dict[str, object]:
    """For each issue in [start_issue, end_issue], build recommendation using previous `window` issues,
    then compute hit count versus actual numbers for that issue.
    """
    all_draws = load_draws(year, 1, end_issue)
    issue_to_numbers = {iss: nums for _, iss, nums in all_draws}
    results = []
    for iss in range(start_issue, end_issue + 1):
        prev_start = max(1, iss - window)
        context = [(y, i, n) for (y, i, n) in all_draws if prev_start <= i <= iss - 1]
        if len(context) == 0 or iss not in issue_to_numbers:
            continue
        freq = compute_frequencies(context)
        hot, cold = hot_cold(freq)
        reco = [n for n in hot if n not in set(cold)][:7]
        actual = set(issue_to_numbers[iss])
        hit = len(actual.intersection(reco))
        results.append({
            "issue": iss,
            "recommended": reco,
            "actual": list(issue_to_numbers[iss]),
            "hits": hit,
        })
    summary = {
        "year": year,
        "range": [start_issue, end_issue],
        "window": window,
        "avg_hits": (sum(r["hits"] for r in results) / max(len(results), 1)) if results else 0.0,
        "count": len(results),
        "details": results,
    }
    save_analysis_history(year, start_issue, end_issue, "backtest_range", json.dumps(summary, ensure_ascii=False))
    return summary
