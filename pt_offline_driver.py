import argparse
import os
import subprocess
import sys
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run thinker + trader in backtest mode from CSV data")
    parser.add_argument("--csv-root", required=True, help="Root folder containing timeframe/coin CSV data")
    parser.add_argument("--timeframe", default="1hour", help="Timeframe folder to replay (default: 1hour)")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_signals"),
        help="Where to write backtest signal outputs",
    )
    parser.add_argument("--gui-settings", help="Optional GUI settings JSON to share coin list across processes")
    parser.add_argument("--no-thinker", action="store_true", help="Do not launch pt_thinker")
    parser.add_argument("--no-trader", action="store_true", help="Do not launch pt_trader monitor")
    return parser.parse_args()


def _launch_process(cmd: List[str], env: dict) -> subprocess.Popen:
    print(f"Starting {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=env)


def main() -> None:
    args = parse_args()

    env = os.environ.copy()
    env["POWERTRADER_MODE"] = "backtest"
    env["POWERTRADER_BACKTEST_CSV_ROOT"] = os.path.abspath(args.csv_root)
    env["POWERTRADER_BACKTEST_TIMEFRAME"] = args.timeframe
    env["POWERTRADER_BACKTEST_OUTPUT"] = os.path.abspath(args.output_dir)
    if args.gui_settings:
        env["POWERTRADER_GUI_SETTINGS"] = os.path.abspath(args.gui_settings)

    procs: List[subprocess.Popen] = []
    try:
        if not args.no_thinker:
            procs.append(_launch_process([sys.executable, "pt_thinker.py"], env))
        if not args.no_trader:
            procs.append(_launch_process([sys.executable, "pt_trader.py"], env))

        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()
