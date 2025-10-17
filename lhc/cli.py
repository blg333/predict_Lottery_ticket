# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from typing import List

from loguru import logger

from .config import CURRENT_YEAR, START_ISSUE, END_ISSUE
from .storage import init_db, upsert_zodiac_map, upsert_number_attributes, bulk_upsert_draws
from .scraper import fetch_zodiac_and_attrs, fetch_draws_for_year
from .analysis import make_recommendation, accuracy_backtest_by_issue, optimize_grid


def cmd_fetch_meta(args):
    logger.info("Fetching zodiac and attribute tables...")
    zodiac_to_numbers, per_number_attrs = fetch_zodiac_and_attrs(args.year)
    upsert_zodiac_map(args.year, zodiac_to_numbers)
    upsert_number_attributes(args.year, per_number_attrs)
    logger.info("Saved zodiac and number attributes for year {}".format(args.year))


def cmd_fetch_draws(args):
    logger.info(f"Fetching draws for year {args.year} issues {args.start:03d}-{args.end:03d}")
    rows_sorted = fetch_draws_for_year(args.year, args.start, args.end)
    bulk_upsert_draws(rows_sorted)
    logger.info(f"Saved {len(rows_sorted)} draw rows")


def cmd_analyze(args):
    result = make_recommendation(args.year, args.start, args.end)
    logger.info("Recommendation summary:\n" + json.dumps(result, ensure_ascii=False, indent=2))


def cmd_backtest(args):
    result = accuracy_backtest_by_issue(args.year, args.start, args.end, window=args.window)
    # show compact view for requested segment
    items = [r for r in result.get('details', []) if r['issue'] >= args.show_from]
    logger.info(
        "Backtest summary (avg_hits={:.3f}, count={}):\n{}".format(
            result.get('avg_hits', 0.0), result.get('count', 0),
            json.dumps(items, ensure_ascii=False, indent=2)
        )
    )


def main():
    parser = argparse.ArgumentParser(description="LHC Macau 2025 crawler and analyzer")
    sub = parser.add_subparsers(dest="cmd")

    p_init = sub.add_parser("initdb")
    p_init.set_defaults(func=lambda a: init_db())

    p_meta = sub.add_parser("fetch-meta")
    p_meta.add_argument("--year", type=int, default=CURRENT_YEAR)
    p_meta.set_defaults(func=cmd_fetch_meta)

    p_draws = sub.add_parser("fetch-draws")
    p_draws.add_argument("--year", type=int, default=CURRENT_YEAR)
    p_draws.add_argument("--start", type=int, default=1)
    p_draws.add_argument("--end", type=int, default=289)
    p_draws.set_defaults(func=cmd_fetch_draws)

    p_an = sub.add_parser("analyze")
    p_an.add_argument("--year", type=int, default=CURRENT_YEAR)
    p_an.add_argument("--start", type=int, default=1)
    p_an.add_argument("--end", type=int, default=289)
    p_an.set_defaults(func=cmd_analyze)

    p_bt = sub.add_parser("backtest")
    p_bt.add_argument("--year", type=int, default=CURRENT_YEAR)
    p_bt.add_argument("--start", type=int, default=1)
    p_bt.add_argument("--end", type=int, default=289)
    p_bt.add_argument("--window", type=int, default=50)
    p_bt.add_argument("--show-from", dest="show_from", type=int, default=280)
    p_bt.set_defaults(func=cmd_backtest)

    def cmd_optimize(args):
        res = optimize_grid(
            args.year, args.start, args.end,
            windows=args.windows, prev_ks=args.prev_ks, decay_list=args.decays, weight_grid=None,
        )
        logger.info("Best config:\n" + json.dumps(res["config"], ensure_ascii=False, indent=2))
        logger.info("Summary avg_hits={:.3f}, count={}".format(res["summary"].get("avg_hits", 0.0), res["summary"].get("count", 0)))

    p_opt = sub.add_parser("optimize")
    p_opt.add_argument("--year", type=int, default=CURRENT_YEAR)
    p_opt.add_argument("--start", type=int, default=220)
    p_opt.add_argument("--end", type=int, default=290)
    p_opt.add_argument("--windows", type=int, nargs="*", default=[40,50,60])
    p_opt.add_argument("--prev-ks", dest="prev_ks", type=int, nargs="*", default=[1,2,3])
    p_opt.add_argument("--decays", type=float, nargs="*", default=[0.95,0.97,0.99])
    p_opt.set_defaults(func=cmd_optimize)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    init_db()
    args.func(args)


if __name__ == "__main__":
    main()
