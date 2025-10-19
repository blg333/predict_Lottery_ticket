# -*- coding:utf-8 -*-
"""
End-to-end pipeline: crawl LHC (2025 001-291), analyze, and print chat report.
"""
from __future__ import annotations

import argparse
from loguru import logger

from get_data import run as crawl_run
from analysis_lhc import run as analysis_run


parser = argparse.ArgumentParser()
parser.add_argument("--crawl", action="store_true", help="Run crawler for LHC 2025 001-291")
args = parser.parse_args()


def main():
    if args.crawl:
        logger.info("开始抓取新澳门六合彩 2025 年 001-291 期...")
        crawl_run("lhc")
    logger.info("开始分析并输出策略预测报告...")
    report = analysis_run()
    print(report)


if __name__ == "__main__":
    main()
