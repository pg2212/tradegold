from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def bars(rows: list[tuple], start: str = "2026-01-01", freq: str = "h") -> pd.DataFrame:
    """Build an OHLCV frame from (open, high, low, close, volume) tuples."""
    arr = np.asarray(rows, dtype=float)
    idx = pd.date_range(start, periods=len(arr), freq=freq, tz="UTC")
    return pd.DataFrame(
        {"Open": arr[:, 0], "High": arr[:, 1], "Low": arr[:, 2],
         "Close": arr[:, 3], "Volume": arr[:, 4]},
        index=idx,
    )


def candle(direction: str, base: float = 2000.0, body: float = 10.0,
           wick: float = 2.0, volume: float = 100.0) -> tuple:
    """One synthetic candle in the given direction."""
    if direction == "up":
        o, c = base, base + body
    elif direction == "down":
        o, c = base + body, base
    else:
        o = c = base
    return (o, max(o, c) + wick, min(o, c) - wick, c, volume)


@pytest.fixture
def context_bars() -> pd.DataFrame:
    """60 flat low-volume candles: a clean baseline for the volume average."""
    return bars([candle("flat", 2000.0, 0.0, 1.0, 100.0) for _ in range(60)])
