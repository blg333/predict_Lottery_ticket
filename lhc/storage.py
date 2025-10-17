# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Iterable, List, Dict, Any, Tuple

from .config import DB_PATH, DATA_DIR


def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def init_db() -> None:
    ensure_dirs()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        # Draw records: one row per issue
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS draws (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              year INTEGER NOT NULL,
              issue INTEGER NOT NULL,
              code TEXT NOT NULL,           -- e.g. '01,12,23,34,35,46,07'
              numbers TEXT NOT NULL,        -- normalized JSON array string of ints length 7
              special INTEGER,              -- last one if applicable
              created_at DATETIME DEFAULT (datetime('now')),
              UNIQUE(year, issue)
            );
            """
        )
        # Number attributes (zodiac, color, odd/even, home/wild etc.) per year
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS number_attributes (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              year INTEGER NOT NULL,
              number INTEGER NOT NULL,
              zodiac TEXT,
              color TEXT,           -- 波色
              odd_even TEXT,        -- 单/双
              home_wild TEXT,       -- 家/野
              UNIQUE(year, number)
            );
            """
        )
        # Mapping from number to zodiac for given year (redundant but fast lookups)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS zodiac_map (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              year INTEGER NOT NULL,
              zodiac TEXT NOT NULL,
              numbers TEXT NOT NULL,
              UNIQUE(year, zodiac)
            );
            """
        )
        # Analysis history snapshots
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_history (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created_at DATETIME DEFAULT (datetime('now')),
              year INTEGER NOT NULL,
              start_issue INTEGER NOT NULL,
              end_issue INTEGER NOT NULL,
              method TEXT NOT NULL,     -- e.g., 'frequency/hot_cold/transition/backtest'
              payload TEXT NOT NULL     -- JSON dump of results/params
            );
            """
        )
        # Test data persistence
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS test_data (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created_at DATETIME DEFAULT (datetime('now')),
              year INTEGER NOT NULL,
              issue INTEGER NOT NULL,
              numbers TEXT NOT NULL,
              source TEXT NOT NULL,
              UNIQUE(year, issue, source)
            );
            """
        )
        conn.commit()


@contextmanager
def get_conn():
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def upsert_zodiac_map(year: int, zodiac_to_numbers: Dict[str, List[int]]) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        for zodiac, numbers in zodiac_to_numbers.items():
            cur.execute(
                "INSERT OR REPLACE INTO zodiac_map (year, zodiac, numbers) VALUES (?, ?, ?)",
                (year, zodiac, ",".join(map(lambda x: f"{x:02d}", numbers))),
            )
        conn.commit()


def upsert_number_attributes(year: int, attrs: Dict[int, Dict[str, Any]]) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        for num, a in attrs.items():
            cur.execute(
                """
                INSERT OR REPLACE INTO number_attributes (year, number, zodiac, color, odd_even, home_wild)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (year, num, a.get("zodiac"), a.get("color"), a.get("odd_even"), a.get("home_wild")),
            )
        conn.commit()


def upsert_draw(year: int, issue: int, numbers: List[int]) -> None:
    numbers_sorted = numbers[:]  # do not sort to preserve order
    code = ",".join(f"{n:02d}" for n in numbers_sorted)
    special = numbers_sorted[-1] if numbers_sorted else None
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO draws (year, issue, code, numbers, special)
            VALUES (?, ?, ?, ?, ?)
            """,
            (year, issue, code, ",".join(str(n) for n in numbers_sorted), special),
        )
        conn.commit()


def bulk_upsert_draws(rows: Iterable[Tuple[int, int, List[int]]]) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        for year, issue, numbers in rows:
            code = ",".join(f"{n:02d}" for n in numbers)
            special = numbers[-1] if numbers else None
            cur.execute(
                "INSERT OR REPLACE INTO draws (year, issue, code, numbers, special) VALUES (?, ?, ?, ?, ?)",
                (year, issue, code, ",".join(map(str, numbers)), special),
            )
        conn.commit()


def save_analysis_history(year: int, start_issue: int, end_issue: int, method: str, payload_json: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO analysis_history (year, start_issue, end_issue, method, payload) VALUES (?, ?, ?, ?, ?)",
            (year, start_issue, end_issue, method, payload_json),
        )
        conn.commit()


def save_test_data(year: int, issue: int, numbers: List[int], source: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO test_data (year, issue, numbers, source) VALUES (?, ?, ?, ?)",
            (year, issue, ",".join(map(str, numbers)), source),
        )
        conn.commit()


def load_draws(year: int, start_issue: int, end_issue: int) -> List[Tuple[int, int, List[int]]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT year, issue, numbers FROM draws WHERE year=? AND issue BETWEEN ? AND ? ORDER BY issue ASC",
            (year, start_issue, end_issue),
        )
        rows = []
        for y, iss, numbers in cur.fetchall():
            nums = [int(x) for x in numbers.split(",") if x]
            rows.append((y, iss, nums))
        return rows
