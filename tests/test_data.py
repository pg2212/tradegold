"""Provider normalisation and context resampling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradegold.data import _normalise, resample_bars


def frame(n=8, start="2026-01-01", freq="15min", cols=None):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    data = {
        "Open": np.arange(n, dtype=float) + 100,
        "High": np.arange(n, dtype=float) + 101,
        "Low": np.arange(n, dtype=float) + 99,
        "Close": np.arange(n, dtype=float) + 100.5,
        "Volume": np.full(n, 1000.0),
    }
    df = pd.DataFrame(data, index=idx)
    if cols:
        df.columns = cols
    return df


def test_lowercase_vendor_columns_are_accepted():
    """A vendor CSV using open/high/low/close/volume needs no preprocessing."""
    df = frame(cols=["open", "high", "low", "close", "volume"])
    out = _normalise(df)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert out["Close"].iloc[0] == 100.5


def test_extra_columns_are_dropped_not_rejected():
    df = frame(cols=["open", "high", "low", "close", "volume"])
    df["symbol"] = "XAUUSD"
    out = _normalise(df)
    assert "symbol" not in out.columns
    assert len(out.columns) == 5


def test_missing_column_error_names_what_was_found():
    df = frame(cols=["open", "high", "low", "close", "volume"]).drop(columns=["volume"])
    with pytest.raises(ValueError, match="Found:"):
        _normalise(df)


def test_index_is_made_utc_and_sorted():
    df = frame()
    df.index = df.index.tz_localize(None)[::-1]  # naive and reversed
    out = _normalise(df)
    assert out.index.tz is not None
    assert out.index.is_monotonic_increasing


# ------------------------------------------------------------- resampling ---

def test_resample_aggregates_ohlcv_correctly():
    df = frame(n=8, freq="15min")          # two full hours
    out = resample_bars(df, "60m")

    assert len(out) == 2
    assert out["Open"].iloc[0] == df["Open"].iloc[0]     # first
    assert out["High"].iloc[0] == df["High"].iloc[:4].max()
    assert out["Low"].iloc[0] == df["Low"].iloc[:4].min()
    assert out["Close"].iloc[0] == df["Close"].iloc[3]   # last
    assert out["Volume"].iloc[0] == df["Volume"].iloc[:4].sum()


def test_resampled_bars_are_stamped_by_open_time():
    """The whole alignment scheme assumes open-time stamping."""
    df = frame(n=8, start="2026-01-01 00:00", freq="15min")
    out = resample_bars(df, "60m")
    assert out.index[0] == pd.Timestamp("2026-01-01 00:00", tz="UTC")
    assert out.index[1] == pd.Timestamp("2026-01-01 01:00", tz="UTC")


def test_empty_buckets_are_dropped_not_forward_filled():
    """Weekend gaps must not become phantom candles."""
    df = pd.concat([frame(n=4, start="2026-01-01 00:00"),
                    frame(n=4, start="2026-01-01 05:00")])
    out = resample_bars(df, "60m")
    assert len(out) == 2, "only the two hours with data should survive"
    assert out.isna().sum().sum() == 0


def test_resampling_preserves_total_volume():
    df = frame(n=20)
    out = resample_bars(df, "60m")
    assert out["Volume"].sum() == pytest.approx(df["Volume"].sum())


def test_resampled_range_never_exceeds_the_source():
    rng = np.random.default_rng(5)
    n = 200
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    close = 2000 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame({"Open": close, "High": close + 2, "Low": close - 2,
                       "Close": close, "Volume": 10.0}, index=idx)
    out = resample_bars(df, "60m")
    assert out["High"].max() <= df["High"].max()
    assert out["Low"].min() >= df["Low"].min()
    assert (out["High"] >= out["Low"]).all()
