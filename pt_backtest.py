"""Offline backtesting driver for ``pt_trainer`` using CSV datasets.

The script walks ``<data-dir>/<timeframe>/<coin>/*.csv`` folders, feeds the
candles into ``pt_trainer.load_history`` (CSV mode), and writes isolated
artifacts/metrics into ``--output-dir`` so live neural state stays untouched.

CSV inputs are expected to include headers ``time,open,high,low,close`` using
Unix timestamps for ``time``. A ``volume`` column is optional; if it is
missing, a default of 0.0 is used during parsing.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import pt_trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline backtester for pt_trainer")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root folder containing timeframe/coin CSV data (timeframe/COIN/*.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="backtest_output",
        help="Folder where backtest artifacts are written",
    )
    parser.add_argument(
        "--per-coin-logs",
        action="store_true",
        help="Write detailed per-coin logs alongside the summary report",
    )
    parser.add_argument(
        "--coin",
        action="append",
        help="Limit backtest to specific coin directories (can repeat)",
    )
    return parser.parse_args()


def discover_datasets(root: str, allowed_coins: Iterable[str] | None) -> Dict[str, List[str]]:
    datasets: Dict[str, List[str]] = defaultdict(list)
    for timeframe in sorted(os.listdir(root)):
        tf_path = os.path.join(root, timeframe)
        if not os.path.isdir(tf_path):
            continue
        for coin in sorted(os.listdir(tf_path)):
            if allowed_coins and coin not in allowed_coins:
                continue
            coin_path = os.path.join(tf_path, coin)
            if not os.path.isdir(coin_path):
                continue
            datasets[timeframe].append(coin)
    return datasets


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _parse_history_rows(raw_rows: List[str]) -> List[Tuple[float, float, float, float, float]]:
    parsed = []
    for raw in raw_rows:
        try:
            values = ast.literal_eval(raw)
            # [timestamp, open, close, high, low, volume]
            parsed.append(
                (
                    float(values[0]),
                    float(values[1]),
                    float(values[2]),
                    float(values[3]),
                    float(values[4]),
                )
            )
        except Exception:
            continue
    return parsed


def _calc_metrics(rows: List[Tuple[float, float, float, float, float]]) -> Dict[str, float]:
    if not rows:
        return {
            "samples": 0,
            "win_rate": 0.0,
            "hit_rate": 0.0,
            "avg_threshold": 0.0,
        }
    wins = 0
    hits = 0
    thresholds: List[float] = []
    for _, open_price, close_price, high_price, low_price in rows:
        if close_price > open_price:
            wins += 1
        intraday_move = max(abs(high_price - open_price), abs(open_price - low_price))
        if intraday_move > 0:
            hits += 1
        thresholds.append(abs(close_price - open_price))
    total = len(rows)
    return {
        "samples": total,
        "win_rate": wins / total,
        "hit_rate": hits / total,
        "avg_threshold": sum(thresholds) / total if thresholds else 0.0,
    }


def write_artifacts(
    output_root: str,
    coin: str,
    timeframe: str,
    metrics: Dict[str, float],
    raw_rows: List[str],
    started_at: int,
) -> None:
    coin_dir = os.path.join(output_root, coin, timeframe)
    ensure_dir(coin_dir)
    # Store raw history as backtest "memory" artifacts
    with open(os.path.join(coin_dir, f"memories_{timeframe}.txt"), "w", encoding="utf-8") as fh:
        fh.write("~".join(raw_rows))
    with open(
        os.path.join(coin_dir, f"memory_weights_{timeframe}.txt"), "w", encoding="utf-8"
    ) as fh:
        fh.write(" ".join([str(metrics.get("avg_threshold", 0.0))] * 10))
    with open(
        os.path.join(coin_dir, f"neural_perfect_threshold_{timeframe}.txt"), "w", encoding="utf-8"
    ) as fh:
        fh.write(str(metrics.get("avg_threshold", 0.0)))
    # Training time markers mirror pt_trainer outputs but stay isolated
    with open(os.path.join(coin_dir, "trainer_status.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "coin": coin,
                "state": "FINISHED",
                "started_at": started_at,
                "finished_at": int(time.time()),
                "timestamp": int(time.time()),
                "mode": "backtest",
            },
            fh,
            indent=2,
        )


def write_summary(output_root: str, summary_rows: List[Dict[str, str]]) -> None:
    ensure_dir(output_root)
    report_path = os.path.join(output_root, "backtest_report.csv")
    fieldnames = ["coin", "timeframe", "samples", "win_rate", "hit_rate", "avg_threshold"]
    with open(report_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def write_coin_logs(output_root: str, coin: str, entries: List[Dict[str, str]]) -> None:
    if not entries:
        return
    coin_dir = os.path.join(output_root, coin)
    ensure_dir(coin_dir)
    log_path = os.path.join(coin_dir, "backtest_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["timeframe", "samples", "win_rate", "hit_rate", "avg_threshold"]
        )
        writer.writeheader()
        writer.writerows(entries)


def main() -> None:
    args = parse_args()
    data_root = os.path.abspath(args.data_dir)
    output_root = os.path.abspath(args.output_dir)
    allowed_coins = set(args.coin) if args.coin else None

    datasets = discover_datasets(data_root, allowed_coins)
    if not datasets:
        raise SystemExit("No datasets found in the provided data directory")

    summary_rows: List[Dict[str, str]] = []
    per_coin_entries: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for timeframe, coins in datasets.items():
        for coin in coins:
            started_at = int(time.time())
            raw_rows = pt_trainer.load_history(
                coin,
                timeframe,
                start=float("inf"),
                end=0.0,
                source="csv",
                csv_root=data_root,
            )
            parsed_rows = _parse_history_rows(raw_rows)
            metrics = _calc_metrics(parsed_rows)

            summary_row = {
                "coin": coin,
                "timeframe": timeframe,
                "samples": str(metrics["samples"]),
                "win_rate": f"{metrics['win_rate']:.4f}",
                "hit_rate": f"{metrics['hit_rate']:.4f}",
                "avg_threshold": f"{metrics['avg_threshold']:.8f}",
            }
            summary_rows.append(summary_row)
            per_coin_entries[coin].append({k: summary_row[k] for k in summary_row if k != "coin"})

            write_artifacts(output_root, coin, timeframe, metrics, raw_rows, started_at)

    write_summary(output_root, summary_rows)

    if args.per_coin_logs:
        for coin, entries in per_coin_entries.items():
            write_coin_logs(output_root, coin, entries)

    print(f"Backtest complete. Summary written to {os.path.join(output_root, 'backtest_report.csv')}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
