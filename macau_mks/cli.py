# -*- coding: utf-8 -*-
import argparse
import os
try:
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover
    from .log import logger

from .scraper import crawl_2025_and_save
from .analysis import compute_frequencies, hot_cold, omission_streaks, cooccurrence_matrix, transition_matrix_special
from .models_simple import (
    frequency_balance,
    omission_based,
    trend_window_frequency,
    association_influence,
    simple_markov,
    bayesian_dirichlet,
    ensemble_average,
    dynamic_markov,
)
from .backtest import topk_accuracy
from .zodiac import fetch_number_to_zodiac, zodiac_follow_model
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Macau Mark Six 2025 analysis and prediction")
    parser.add_argument("--data_dir", default="data/macau_mks", type=str)
    parser.add_argument("--topk", default=6, type=int)
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    records = crawl_2025_and_save(args.data_dir)

    freq = compute_frequencies(records)["special"]
    hots, colds = hot_cold(freq, top_k=10)
    logger.info(f"热门特码: {hots}")
    logger.info(f"冷门特码: {colds}")

    # Build model probabilities
    p_fb = frequency_balance(records)
    p_om = omission_based(records)
    p_tr = trend_window_frequency(records)
    p_ai = association_influence(records)
    p_mk = simple_markov(records)
    p_by = bayesian_dirichlet(records)
    p_dm = dynamic_markov(records)
    # Zodiac-based model
    try:
        num2z = fetch_number_to_zodiac(2025)
        p_zd = zodiac_follow_model(records, num2z)
    except Exception:
        p_zd = [1/49.0] * 49

    ens = ensemble_average([p_fb, p_om, p_tr, p_ai, p_mk, p_by, p_dm, p_zd])

    # Recommend numbers per model
    def topn(p, n=10):
        return [i+1 for i,_ in sorted(list(enumerate(p)), key=lambda kv: -kv[1])[:n]]

    logger.info(f"频率平衡预测(特码): {topn(p_fb)}")
    logger.info(f"遗漏值预测(特码): {topn(p_om)}")
    logger.info(f"趋势预测(特码): {topn(p_tr)}")
    logger.info(f"关联性预测(特码): {topn(p_ai)}")
    logger.info(f"马尔科夫链预测(特码): {topn(p_mk)}")
    logger.info(f"动态概率矩阵预测(特码): {topn(p_dm)}")
    logger.info(f"生肖转移预测(特码): {topn(p_zd)}")
    logger.info(f"贝叶斯后验预测(特码): {topn(p_by)}")
    logger.info(f"综合推荐(特码): {topn(ens)}")

    # Backtest
    acc_fb = topk_accuracy(records, frequency_balance, k=args.topk)
    acc_om = topk_accuracy(records, omission_based, k=args.topk)
    acc_tr = topk_accuracy(records, lambda hist: trend_window_frequency(hist, 50), k=args.topk)
    acc_ai = topk_accuracy(records, association_influence, k=args.topk)
    acc_mk = topk_accuracy(records, simple_markov, k=args.topk)
    acc_by = topk_accuracy(records, bayesian_dirichlet, k=args.topk)
    acc_dm = topk_accuracy(records, dynamic_markov, k=args.topk)
    try:
        acc_zd = topk_accuracy(records, lambda hist: zodiac_follow_model(hist, fetch_number_to_zodiac(2025)), k=args.topk)
    except Exception:
        acc_zd = 0.0

    logger.info(
        {
            "acc_frequency": acc_fb,
            "acc_omission": acc_om,
            "acc_trend": acc_tr,
            "acc_association": acc_ai,
            "acc_markov": acc_mk,
            "acc_dynamic": acc_dm,
            "acc_zodiac": acc_zd,
            "acc_bayes": acc_by,
        }
    )


if __name__ == "__main__":
    main()
