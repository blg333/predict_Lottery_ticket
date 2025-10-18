# -*- coding: utf-8 -*-
"""
Backtesting utilities for 特码 models.
- Evaluate hit rate when picking top-K predictions per draw
"""
from typing import Callable, List, Dict
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    class _NP:
        def argsort(self, arr):
            return sorted(range(len(arr)), key=lambda i: arr[i])
    np = _NP()

from .types import DrawRecord


def topk_accuracy(records: List[DrawRecord], proba_fn: Callable[[List[DrawRecord]], list], k: int = 6) -> float:
    hits = 0
    total = 0
    for i in range(1, len(records)):
        hist = records[:i]
        probs = proba_fn(hist)
        idx_sorted = list(reversed(np.argsort(probs)))
        topk = [i + 1 for i in idx_sorted[:k]]
        # topk is list now
        if records[i].special in topk:
            hits += 1
        total += 1
    return hits / total if total > 0 else 0.0
