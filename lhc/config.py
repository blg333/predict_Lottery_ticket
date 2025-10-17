# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass

# Project paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "lhc.sqlite3")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
REPORT_DIR = os.path.join(DATA_DIR, "reports")
TESTDATA_DIR = os.path.join(DATA_DIR, "testdata")

# Sources
ZODIAC_SOURCE_URL = "https://kj.123720c.com/kj/zl.html"
RESULTS_BASE_URL = "https://kj.123720c.com/kj/"

CURRENT_YEAR = 2025
START_ISSUE = 1
END_ISSUE = 290  # crawl up to 290 if available

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

REQUEST_TIMEOUT = 20
RETRY_TIMES = 3
RETRY_BACKOFF = 1.7

@dataclass
class RecommendationConfig:
    top_hot_count: int = 12        # pick top-N hot numbers
    bottom_cold_exclude: int = 6   # exclude bottom-N cold numbers
    recent_window: int = 50        # frequency window

RECO_CONFIG = RecommendationConfig()
