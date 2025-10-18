# -*- coding: utf-8 -*-
"""Lightweight logger shim.
Tries to use loguru if available; falls back to simple print-based logger.
"""
try:
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover
    class _Logger:
        def info(self, msg):
            print(f"[INFO] {msg}")

        def warning(self, msg):
            print(f"[WARN] {msg}")

        def error(self, msg):
            print(f"[ERROR] {msg}")

        def debug(self, msg):
            print(f"[DEBUG] {msg}")

    logger = _Logger()
