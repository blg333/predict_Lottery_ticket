# -*- coding: utf-8 -*-
"""
Data types for Macau Mark Six (新澳门六合彩).
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DrawRecord:
    period: int  # e.g., 1..291 for 2025
    date: str  # e.g., 2025-10-18
    normals: List[int]  # six numbers
    special: int  # 特号（特码）
    normals_zodiac: Optional[List[str]] = None  # length 6 if available
    special_zodiac: Optional[str] = None
    normals_wuxing: Optional[List[str]] = None
    special_wuxing: Optional[str] = None
