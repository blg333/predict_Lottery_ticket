# -*- coding: utf-8 -*-
"""
Advanced models (lightweight, optional) for 特码预测：
- kmeans_clustering: 基于共现向量对1..49聚类，并用簇级别历史权重分配概率
- ml_context_nb: 朴素贝叶斯/高阶马尔可夫（基于上1-2期上下文）
- arima_ar1: 简化AR(1) 拟合 + 高斯离散化概率
- lstm_seq: 若安装tensorflow则训练极简LSTM，否则回退到动态马尔科夫
"""
from typing import Dict, List, Tuple
import math

from .types import DrawRecord
from .analysis import cooccurrence_matrix
from .models_simple import dynamic_markov


def _number_embeddings_from_cooccurrence(records: List[DrawRecord]) -> Dict[int, List[float]]:
    mat = cooccurrence_matrix(records)  # 50x50
    emb = {}
    for n in range(1, 50):
        # 使用与其它号码的共现计数作为向量
        row = mat[n]
        # 置零自身并标准化
        vec = [float(v) for v in row]
        vec[n] = 0.0
        s = sum(vec)
        if s == 0:
            vec = [0.0 for _ in vec]
        else:
            vec = [v / s for v in vec]
        emb[n] = vec
    return emb


def _euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _kmeans(points: Dict[int, List[float]], k: int = 6, iters: int = 20) -> Dict[int, int]:
    # 初始化：选择向量范数最大的k个点作为中心（稳定且可重复）
    ordered = sorted(points.keys(), key=lambda n: sum(v * v for v in points[n]), reverse=True)[:k]
    centers = [points[n][:] for n in ordered]
    assign: Dict[int, int] = {n: 0 for n in points}
    for _ in range(iters):
        # 分配
        changed = False
        for n, v in points.items():
            dists = [ _euclidean(v, c) for c in centers ]
            cid = min(range(k), key=lambda i: dists[i])
            if assign.get(n) != cid:
                assign[n] = cid
                changed = True
        # 更新中心
        sums = [ [0.0]*len(next(iter(points.values()))) for _ in range(k) ]
        counts = [0]*k
        for n, cid in assign.items():
            counts[cid] += 1
            sums[cid] = [a + b for a, b in zip(sums[cid], points[n])]
        for i in range(k):
            if counts[i] == 0:
                continue
            centers[i] = [v / counts[i] for v in sums[i]]
        if not changed:
            break
    return assign


def kmeans_clustering(records: List[DrawRecord], k: int = 6) -> List[float]:
    if not records:
        return [1/49.0]*49
    emb = _number_embeddings_from_cooccurrence(records)
    assign = _kmeans(emb, k=k)
    # 统计簇级别的历史特号频率
    cluster_counts = [0]*k
    for r in records:
        cid = assign.get(r.special)
        if cid is not None:
            cluster_counts[cid] += 1
    total = sum(cluster_counts)
    if total == 0:
        cluster_probs = [1.0/k]*k
    else:
        cluster_probs = [c/total for c in cluster_counts]
    # 将簇概率均分至簇内号码，可加权以号码在簇内的共现强度
    probs = [0.0]*49
    # 计算每簇大小
    cluster_sizes = [0]*k
    for n in range(1,50):
        cluster_sizes[assign[n]] += 1
    for n in range(1,50):
        cid = assign[n]
        sz = max(1, cluster_sizes[cid])
        probs[n-1] = cluster_probs[cid] / sz
    s = sum(probs)
    return [v/s for v in probs] if s>0 else [1/49.0]*49


def ml_context_nb(records: List[DrawRecord], order: int = 2, alpha: float = 1.0) -> List[float]:
    # 多项式朴素贝叶斯（高阶马尔科夫退火）：P(y|x_{-1},x_{-2})用相应频数+加性平滑
    if len(records) < 3:
        return [1/49.0]*49
    specials = [r.special for r in records]
    # 三元/二元/一元计数
    tri = {}
    bi = {}
    uni = {}
    for i in range(2, len(specials)):
        x2, x1, y = specials[i-2], specials[i-1], specials[i]
        tri[(x2,x1,y)] = tri.get((x2,x1,y), 0) + 1
        bi[(x1,y)] = bi.get((x1,y), 0) + 1
        uni[y] = uni.get(y, 0) + 1
    last1 = specials[-1]
    last2 = specials[-2]
    # 计算条件分布
    probs = [0.0]*49
    V = 49
    if order >= 2:
        total = sum(tri.get((last2,last1,y), 0) + alpha for y in range(1,50))
        if total > 0:
            for y in range(1,50):
                probs[y-1] = (tri.get((last2,last1,y), 0) + alpha) / total
    # Backoff到二元
    if sum(probs) == 0:
        total = sum(bi.get((last1,y), 0) + alpha for y in range(1,50))
        for y in range(1,50):
            probs[y-1] = (bi.get((last1,y), 0) + alpha) / (total if total>0 else V*alpha)
    # Backoff到一元
    if sum(probs) == 0:
        total = sum(uni.get(y,0) + alpha for y in range(1,50))
        for y in range(1,50):
            probs[y-1] = (uni.get(y,0) + alpha) / (total if total>0 else V*alpha)
    # 归一化
    s = sum(probs)
    return [v/s for v in probs] if s>0 else [1/49.0]*49


def arima_ar1(records: List[DrawRecord]) -> List[float]:
    # 简化AR(1) y_t = phi*y_{t-1} + e_t
    specials = [r.special for r in records]
    if len(specials) < 3:
        return [1/49.0]*49
    num = 0.0
    den = 0.0
    for i in range(1, len(specials)):
        num += specials[i] * specials[i-1]
        den += specials[i-1] * specials[i-1]
    phi = num / den if den != 0 else 0.0
    yhat = phi * specials[-1]
    # 残差方差估计
    res = []
    for i in range(1, len(specials)):
        res.append(specials[i] - phi*specials[i-1])
    var = sum(r*r for r in res) / max(1, (len(res)-1))
    sd = math.sqrt(var) if var>1e-8 else 7.0
    # 高斯离散化
    probs = []
    for n in range(1,50):
        z = (n - yhat)/sd
        probs.append(math.exp(-0.5*z*z))
    s = sum(probs)
    return [v/s for v in probs] if s>0 else [1/49.0]*49


def lstm_seq(records: List[DrawRecord], window: int = 8, epochs: int = 1) -> List[float]:
    # 若安装tensorflow则训练一个极简序列分类模型，否则回退
    try:
        import numpy as np  # type: ignore
        import tensorflow as tf  # type: ignore
    except Exception:
        return dynamic_markov(records)
    specials = [r.special for r in records]
    if len(specials) <= window+1:
        return dynamic_markov(records)
    # 构造样本
    X, y = [], []
    for i in range(window, len(specials)):
        X.append([s-1 for s in specials[i-window:i]])
        y.append(specials[i]-1)
    X = np.array(X, dtype=np.int32)
    y = np.array(y, dtype=np.int32)
    # 模型
    inputs = tf.keras.layers.Input(shape=(window,))
    x = tf.keras.layers.Embedding(input_dim=49, output_dim=16)(inputs)
    x = tf.keras.layers.LSTM(32)(x)
    outputs = tf.keras.layers.Dense(49, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    try:
        model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
    except Exception:
        return dynamic_markov(records)
    last = np.array([[s-1 for s in specials[-window:]]], dtype=np.int32)
    p = model.predict(last, verbose=0)[0]
    return (p / p.sum()).tolist()
