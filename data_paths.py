import os
from typing import Iterable

# Common timeframe folders used by offline CSV workflows
TIMEFRAME_FOLDERS = ["1hour", "2hour", "4hour", "8hour", "12hour", "1day", "1week"]


def repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def default_csv_data_root() -> str:
    """Return the default location for CSV inputs inside the repo."""
    return os.path.join(repo_root(), "data", "csv")


def default_training_data_root() -> str:
    """Return the default location for CSV training inputs inside the repo."""
    return os.path.join(repo_root(), "data", "training")


def default_backtest_data_root() -> str:
    """Return the default location for CSV backtesting inputs inside the repo."""
    return os.path.join(repo_root(), "data", "backtesting")


def ensure_timeframe_dirs(base_root: str, timeframes: Iterable[str] = None) -> None:
    """Create per-timeframe folders (and .gitkeep files) for CSV inputs."""
    timeframes = list(timeframes) if timeframes is not None else TIMEFRAME_FOLDERS
    for timeframe in timeframes:
        tf_dir = os.path.join(base_root, timeframe)
        os.makedirs(tf_dir, exist_ok=True)
        keep_path = os.path.join(tf_dir, ".gitkeep")
        if not os.path.exists(keep_path):
            try:
                with open(keep_path, "a", encoding="utf-8"):
                    pass
            except OSError:
                # If the location is unwritable we still want the code to continue gracefully
                pass
