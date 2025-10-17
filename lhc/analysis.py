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


# ----------------------- Special-number advanced predictor -----------------------

RED_SET = {1, 2, 7, 8, 12, 13, 18, 19, 23, 24, 29, 30, 34, 35, 40, 45, 46}
BLUE_SET = {3, 4, 9, 10, 14, 15, 20, 25, 26, 31, 36, 37, 41, 42, 47, 48}
GREEN_SET = {5, 6, 11, 16, 17, 21, 22, 27, 28, 32, 33, 38, 39, 43, 44, 49}


def color_of(n: int) -> str:
    if n in RED_SET:
        return "red"
    if n in BLUE_SET:
        return "blue"
    return "green"


def parity_of(n: int) -> str:
    return "even" if (n % 2 == 0) else "odd"


def tail_of(n: int) -> int:
    return n % 10


def _norm_dict(d: Dict) -> Dict:
    if not d:
        return d
    mx = max(d.values()) if d else 1.0
    return {k: (v / mx if mx > 0 else 0.0) for k, v in d.items()}


def _build_category_priors(prev_specs: List[int], decay: float = 0.97) -> Dict[str, Dict]:
    color_cnt: Dict[str, float] = defaultdict(float)
    parity_cnt: Dict[str, float] = defaultdict(float)
    tail_cnt: Dict[int, float] = defaultdict(float)
    if not prev_specs:
        return {"color": {}, "parity": {}, "tail": {}}
    last = len(prev_specs) - 1
    for i, n in enumerate(prev_specs):
        w = decay ** (last - i)
        color_cnt[color_of(n)] += w
        parity_cnt[parity_of(n)] += w
        tail_cnt[tail_of(n)] += w
    return {
        "color": _norm_dict(color_cnt),
        "parity": _norm_dict(parity_cnt),
        "tail": _norm_dict(tail_cnt),
    }


def predict_special_advanced(
    year: int,
    start_issue: int,
    end_issue: int,
    window: int = 50,
    prev_k: int = 1,
    decay: float = 0.99,
    weights: Dict[str, float] | None = None,
    top_k: int = 1,
) -> Dict[str, object]:
    if weights is None:
        weights = {"dec": 1.0, "cooc": 0.8, "trans": 0.3, "color": 0.5, "parity": 0.3, "tail": 0.3}
    draws = load_draws(year, 1, end_issue)
    issue_to_numbers = {iss: nums for _, iss, nums in draws}
    issues = [iss for _, iss, _ in draws]
    specials_series = [nums[-1] for _, _, nums in draws]
    idx_map = {iss: i for i, iss in enumerate(issues)}

    details = []
    for iss in range(start_issue, end_issue + 1):
        if iss not in idx_map:
            continue
        idx = idx_map[iss]
        left = max(0, idx - window)
        context = draws[left:idx]
        prev_specs = specials_series[left:idx]
        prev_spec = specials_series[idx - 1] if idx - 1 >= 0 else None
        anchors = []
        for j in range(max(0, idx - prev_k), idx):
            anchors.extend(draws[j][2])

        # component scores
        dec_spec_scores = defaultdict(float)
        if prev_specs:
            last = len(prev_specs) - 1
            tmp = defaultdict(float)
            for i, n in enumerate(prev_specs):
                w = decay ** (last - i)
                tmp[n] += w
            mx = max(tmp.values()) if tmp else 1.0
            for n in range(1, 50):
                dec_spec_scores[n] = tmp.get(n, 0.0) / mx

        # transitions from prev_spec within context
        trans_scores = defaultdict(float)
        if prev_spec is not None and len(prev_specs) >= 1:
            cnt = defaultdict(int)
            for j in range(0, len(prev_specs) - 1):
                a = prev_specs[j]
                b = prev_specs[j + 1]
                if a == prev_spec:
                    cnt[b] += 1
            maxt = max(cnt.values()) if cnt else 1
            for n in range(1, 50):
                trans_scores[n] = cnt.get(n, 0) / maxt

        # co-occurrence with anchors using context full numbers
        co_cnt = defaultdict(int)
        for _, _, nums in context:
            unique = sorted(set(nums))
            for a in unique:
                for b in unique:
                    if a == b:
                        continue
                    co_cnt[(a, b)] += 1
        co_scores = defaultdict(float)
        if anchors:
            sums = defaultdict(float)
            for n in range(1, 50):
                s = 0.0
                for a in set(anchors):
                    s += co_cnt.get((a, n), 0)
                    s += co_cnt.get((n, a), 0)
                sums[n] = s
            mx = max(sums.values()) if sums else 1.0
            for n in range(1, 50):
                co_scores[n] = sums.get(n, 0.0) / mx

        # category priors
        cat_priors = _build_category_priors(prev_specs, decay=decay)
        color_scores = {n: cat_priors["color"].get(color_of(n), 0.0) for n in range(1, 50)}
        parity_scores = {n: cat_priors["parity"].get(parity_of(n), 0.0) for n in range(1, 50)}
        tail_scores = {n: cat_priors["tail"].get(tail_of(n), 0.0) for n in range(1, 50)}

        final = {}
        for n in range(1, 50):
            final[n] = (
                weights.get("dec", 1.0) * dec_spec_scores.get(n, 0.0)
                + weights.get("trans", 0.3) * trans_scores.get(n, 0.0)
                + weights.get("cooc", 0.8) * co_scores.get(n, 0.0)
                + weights.get("color", 0.5) * color_scores.get(n, 0.0)
                + weights.get("parity", 0.3) * parity_scores.get(n, 0.0)
                + weights.get("tail", 0.3) * tail_scores.get(n, 0.0)
            )
        ranked = sorted(final.items(), key=lambda kv: (-kv[1], kv[0]))
        recos = [n for n, _ in ranked[:top_k]]
        actual = specials_series[idx]
        details.append({
            "issue": iss,
            "pred": recos[0] if recos else None,
            "actual": actual,
            "topk": recos,
            "hit1": int((recos and recos[0] == actual) or False),
            "hit3": int(actual in recos[:3]),
            "hit5": int(actual in recos[:5]),
        })

    avg_top1 = sum(d["hit1"] for d in details) / max(len(details), 1)
    avg_top3 = sum(d["hit3"] for d in details) / max(len(details), 1)
    avg_top5 = sum(d["hit5"] for d in details) / max(len(details), 1)
    summary = {
        "range": [start_issue, end_issue],
        "window": window,
        "prev_k": prev_k,
        "decay": decay,
        "weights": weights,
        "top1": avg_top1,
        "top3": avg_top3,
        "top5": avg_top5,
        "count": len(details),
        "details": details,
    }
    save_analysis_history(year, start_issue, end_issue, "special_predict_advanced", json.dumps(summary, ensure_ascii=False))
    return summary


def optimize_special_grid(
    year: int,
    start_issue: int,
    end_issue: int,
    windows: List[int] | None = None,
    prev_ks: List[int] | None = None,
    decays: List[float] | None = None,
    weight_grid: List[Dict[str, float]] | None = None,
) -> Dict[str, object]:
    if windows is None:
        windows = [40, 50, 60]
    if prev_ks is None:
        prev_ks = [1, 2, 3]
    if decays is None:
        decays = [0.95, 0.97, 0.99]
    if weight_grid is None:
        weight_grid = [
            {"dec": 1.0, "cooc": 0.8, "trans": 0.3, "color": 0.5, "parity": 0.3, "tail": 0.3},
            {"dec": 1.2, "cooc": 0.6, "trans": 0.6, "color": 0.8, "parity": 0.4, "tail": 0.4},
            {"dec": 0.8, "cooc": 1.0, "trans": 0.8, "color": 0.8, "parity": 0.6, "tail": 0.6},
            {"dec": 1.0, "cooc": 1.0, "trans": 1.0, "color": 0.5, "parity": 0.5, "tail": 0.5},
        ]
    best = None
    best_score = -1.0
    best_summary = None
    for w in windows:
        for pk in prev_ks:
            for d in decays:
                for wg in weight_grid:
                    s = predict_special_advanced(year, start_issue, end_issue, window=w, prev_k=pk, decay=d, weights=wg, top_k=5)
                    # prioritize top1, break ties with top3/top5
                    score = s["top1"] + 0.1 * s["top3"] + 0.05 * s["top5"]
                    if score > best_score:
                        best_score = score
                        best = {"window": w, "prev_k": pk, "decay": d, "weights": wg}
                        best_summary = s
    payload = {"best": best, "score": best_score}
    save_analysis_history(year, start_issue, end_issue, "special_optimize_grid", json.dumps(payload, ensure_ascii=False))
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
