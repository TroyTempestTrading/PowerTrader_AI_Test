import ast
import os
from typing import Dict, Iterable, Iterator, List, Tuple

import pt_trainer


class BacktestPriceReplay:
    """Simple CSV-backed price iterator for offline/backtest runs.

    Loads candle history for a single timeframe/coin set and yields close prices
    in chronological order. When the stream is exhausted, the last known price is
    repeated.
    """

    def __init__(self, csv_root: str, timeframe: str, coins: Iterable[str]):
        self.csv_root = os.path.abspath(csv_root)
        self.timeframe = timeframe
        self.coins = [str(c).upper().strip() for c in coins if str(c).strip()]
        self._iters: Dict[str, Iterator[float]] = {}
        self._last_price: Dict[str, float] = {}

    @staticmethod
    def _parse_rows(rows: List[str]) -> List[Tuple[float, float]]:
        parsed: List[Tuple[float, float]] = []
        for raw in rows:
            try:
                vals = ast.literal_eval(raw)
                ts = float(vals[0])
                close_price = float(vals[2])
                parsed.append((ts, close_price))
            except Exception:
                continue
        # replay oldest -> newest
        parsed.sort(key=lambda x: x[0])
        return parsed

    def _iter_for(self, sym: str) -> Iterator[float]:
        rows = pt_trainer.load_history(
            sym,
            self.timeframe,
            start=float("inf"),
            end=0.0,
            source="csv",
            csv_root=self.csv_root,
        )
        parsed = self._parse_rows(rows)
        closes = [c for _, c in parsed]
        if not closes:
            closes = [0.0]
        for price in closes:
            self._last_price[sym] = price
            yield price
        # keep yielding last price once data is exhausted
        while True:
            yield self._last_price.get(sym, 0.0)

    def next_price(self, sym: str) -> float:
        sym = str(sym).upper().strip()
        if sym not in self._iters:
            self._iters[sym] = self._iter_for(sym)
        price = next(self._iters[sym])
        self._last_price[sym] = price
        return price
