# -*- coding: utf-8 -*-
"""
Analysis utilities for 新澳门六合彩 draws.
- Frequency stats (overall and 特码 only)
- Hot/cold identification
- Omission (missing streak) metrics
- Co-occurrence matrix among normals and with 特码
- Transition matrix for 特码 (first-order Markov)
- Backtesting harness helper
"""
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    class _NP:
        def __init__(self):
            pass
        def array(self, x, dtype=float):
            return x
        def zeros(self, shape, dtype=int):
            if isinstance(shape, tuple):
                r, c = shape
                return [[0 for _ in range(c)] for __ in range(r)]
            return [0 for _ in range(shape)]
        def sum(self, a, axis=None, keepdims=False):
            import math
            if axis is None:
                if isinstance(a, list) and a and isinstance(a[0], list):
                    return sum(sum(row) for row in a)
                return sum(a)
            # simple case: axis=1 for 2D
            if axis == 1:
                return [sum(row) for row in a]
            if axis == 0:
                return [sum(col) for col in zip(*a)]
            return sum(a)
        def mean(self, x):
            return (sum(x) / len(x)) if x else 0.0
        def astype(self, a, t):
            return a
    np = _NP()
try:
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover
    from .log import logger

from .types import DrawRecord

NUM_SPACE = list(range(1, 50))  # 1..49


def compute_frequencies(records: List[DrawRecord]) -> Dict[str, Counter]:
    overall = Counter()
    special = Counter()
    for r in records:
        overall.update(r.normals)
        overall.update([r.special])
        special.update([r.special])
    return {"overall": overall, "special": special}


def hot_cold(f: Counter, top_k: int = 10) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    items = sorted(f.items(), key=lambda kv: (-kv[1], kv[0]))
    hots = items[:top_k]
    colds = sorted(items, key=lambda kv: (kv[1], kv[0]))[:top_k]
    return hots, colds


def omission_streaks(records: List[DrawRecord]) -> Dict[int, int]:
    last_seen = {n: -1 for n in NUM_SPACE}
    omission = {n: 0 for n in NUM_SPACE}
    for idx, r in enumerate(records):
        present = set(r.normals + [r.special])
        for n in NUM_SPACE:
            if n in present:
                last_seen[n] = idx
            else:
                omission[n] = idx - last_seen[n] if last_seen[n] >= 0 else idx + 1
    return omission


def omission_streaks_special(records: List[DrawRecord]) -> Dict[int, int]:
    """Compute omission streaks considering 特码 sequence only."""
    last_seen = {n: -1 for n in NUM_SPACE}
    omission = {n: 0 for n in NUM_SPACE}
    for idx, r in enumerate(records):
        s = r.special
        for n in NUM_SPACE:
            if n == s:
                last_seen[n] = idx
                omission[n] = 0
            else:
                omission[n] = idx - last_seen[n] if last_seen[n] >= 0 else idx + 1
    return omission


def cooccurrence_matrix(records: List[DrawRecord]):
    mat = np.zeros((50, 50), dtype=int)
    for r in records:
        nums = r.normals + [r.special]
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                a, b = nums[i], nums[j]
                # Support both numpy arrays and list-of-lists
                try:
                    mat[a, b] += 1
                    mat[b, a] += 1
                except TypeError:
                    mat[a][b] += 1
                    mat[b][a] += 1
    return mat


def transition_matrix_special(records: List[DrawRecord]):
    tm = np.zeros((50, 50), dtype=int)
    for i in range(1, len(records)):
        prev_s = records[i - 1].special
        curr_s = records[i].special
        try:
            tm[prev_s, curr_s] += 1
        except TypeError:
            tm[prev_s][curr_s] += 1
    return tm


def normalize_rows(mat):
    # simple Python implementation compatible with list-of-lists
    m = [[float(v) for v in row] for row in mat]
    normed = []
    for row in m:
        s = sum(row)
        if s == 0:
            s = 1.0
        normed.append([v / s for v in row])
    return normed


def markov_next_probs(tm, last_special: int):
    probs = normalize_rows(tm)[last_special]
    return probs
