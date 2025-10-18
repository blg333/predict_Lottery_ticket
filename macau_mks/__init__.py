"""
Macau Mark Six (新澳门六合彩) analysis toolkit.

Modules:
- scraper: Fetches 2025年001–291期 draws from kj.123720c.com
- analysis: Frequency, hot/cold, omission, co-occurrence, transition matrices
- models: Multiple predictive models for 特码 (multi-class 49)
- backtest: Strategy backtesting utilities
"""

from .types import DrawRecord

__all__ = ["DrawRecord"]
