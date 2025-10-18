# -*- coding: utf-8 -*-
"""
Lightweight implementations for core models to avoid heavy deps at first run.
"""
from collections import Counter
from typing import List, Dict, Tuple
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    class _NP:
        def __init__(self):
            pass
        def asarray(self, x, dtype=float):
            return x
        def array(self, x, dtype=float):
            return x
        def exp(self, x):
            import math
            return [math.exp(v) for v in x]
        def max(self, x):
            return max(x)
        def zeros(self, shape, dtype=float):
            if isinstance(shape, tuple):
                r, c = shape
                return [[0 for _ in range(c)] for __ in range(r)]
            return [0 for _ in range(shape)]
        def ones(self, n):
            return [1.0] * n
        def argsort(self, x):
            return sorted(range(len(x)), key=lambda i: x[i])
    np = _NP()

from .types import DrawRecord
from .analysis import compute_frequencies, omission_streaks, cooccurrence_matrix, transition_matrix_special, normalize_rows

NUM_SPACE = list(range(1, 50))


def softmax(x):
    try:
        import numpy as _n  # type: ignore
        x = _n.asarray(x, dtype=float)
        x = x - _n.max(x)
        e = _n.exp(x)
        return e / e.sum()
    except Exception:
        m = max(x)
        exps = [pow(2.718281828, v - m) for v in x]
        s = sum(exps)
        return [v / s for v in exps]


def proba_from_counter(cnt: Counter):
    arr = [0.0] * 50
    for n, c in cnt.items():
        arr[n] = c
    if sum(arr) == 0:
        arr[1:] = [1.0] * 49
    arr = arr[1:]
    return softmax(arr)


def frequency_balance(records: List[DrawRecord]):
    f = compute_frequencies(records)["special"]
    vals = list(f.values()) or [0]
    mean_f = sum(vals) / len(vals)
    # Penalize over-frequent numbers, boost under-frequent
    scores = [0.0] * 50
    for n in NUM_SPACE:
        diff = mean_f - f.get(n, 0)
        scores[n] = 1.0 + diff
    return softmax(scores[1:])


def omission_based(records: List[DrawRecord]):
    om = omission_streaks(records)
    scores = [0.0] * 50
    for n in NUM_SPACE:
        scores[n] = om[n]
    return softmax(scores[1:])


def trend_window_frequency(records: List[DrawRecord], window: int = 50):
    tail = records[-window:] if len(records) >= window else records
    cnt = Counter([r.special for r in tail])
    return proba_from_counter(cnt)


def association_influence(records: List[DrawRecord]):
    # Influence of co-occurrence with normals on special
    mat = cooccurrence_matrix(records)
    last = records[-1]
    infl = [0.0] * 50
    for n in last.normals:
        row = mat[n]
        infl = [a + b for a, b in zip(infl, row)]
    infl = infl[1:]
    s = sum(infl)
    if s == 0:
        return [1/49.0] * 49
    return [v / s for v in infl]


def simple_markov(records: List[DrawRecord], decay: float = 1.0):
    tm = transition_matrix_special(records)
    last_sp = records[-1].special
    row = normalize_rows(tm)[last_sp][1:]
    s = sum(row)
    if s == 0:
        return [1/49.0] * 49
    return [v / s for v in row]


def dynamic_markov(records: List[DrawRecord], lambda_decay: float = 0.98):
    # Exponentially-weighted transition counts favoring recent transitions
    if len(records) < 2:
        return [1/49.0] * 49
    size = 50
    tm = [[0.0 for _ in range(size)] for __ in range(size)]
    n = len(records)
    for i in range(1, n):
        prev_s = records[i-1].special
        curr_s = records[i].special
        # weight increases as i approaches n-1
        age = (n - 1) - i
        w = (lambda_decay ** age)
        tm[prev_s][curr_s] += w
    last_sp = records[-1].special
    row = tm[last_sp][1:]
    s = sum(row)
    if s == 0:
        return [1/49.0] * 49
    return [v / s for v in row]


def bayesian_dirichlet(records: List[DrawRecord], alpha: float = 1.0):
    cnt = Counter([r.special for r in records])
    post = [cnt.get(n, 0) + alpha for n in NUM_SPACE]
    post = post[1:]
    s = sum(post)
    return [v / s for v in post]


def ensemble_average(probas: List, weights: List[float] = None):
    if not probas:
        return np.ones(49) / 49.0
    if weights is None:
        weights = [1.0 / len(probas)] * len(probas)
    probs = [0.0] * 49
    for p, w in zip(probas, weights):
        probs = [a + w*b for a, b in zip(probs, p)]
    s = sum(probs)
    return [v / s for v in probs]
