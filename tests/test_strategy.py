"""Setup detection and the trigger state machine, pinned to the Pine source."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import bars, candle

from tradegold.config import StrategyConfig
from tradegold.strategy import (
    LONG,
    SHORT,
    build_context,
    detect_setups,
    generate_signals,
)


def cfg(**kw) -> StrategyConfig:
    return StrategyConfig(**{"ema_fast": 3, "ema_slow": 4, "ema_trend": 5,
                             "volume_ma_period": 5, **kw})


def rising_market(n: int = 40, volume: float = 100.0) -> list[tuple]:
    """A steadily rising series, so EMA fast > trend and price > trend EMA."""
    return [candle("up", 2000 + i * 5, body=4.0, wick=1.0, volume=volume) for i in range(n)]


def falling_market(n: int = 40, volume: float = 100.0) -> list[tuple]:
    return [candle("down", 2000 - i * 5, body=4.0, wick=1.0, volume=volume) for i in range(n)]


# ---------------------------------------------------------------- context ---

def test_context_indicators_are_causal():
    """A future candle must not change any earlier indicator value."""
    rows = rising_market(40)
    base = build_context(bars(rows), cfg())

    rows[-1] = candle("up", 5000, body=100, volume=99999)
    tampered = build_context(bars(rows), cfg())

    for col in ("ema_fast", "ema_slow", "ema_trend", "vol_ma"):
        pd.testing.assert_series_equal(base[col].iloc[:-1], tampered[col].iloc[:-1])


def test_volume_threshold_is_the_average_times_the_multiplier():
    ctx = build_context(bars(rising_market(20, volume=100.0)), cfg(volume_multiplier=1.5))
    assert ctx["vol_ma"].dropna().iloc[-1] == pytest.approx(100.0)
    assert ctx["vol_threshold"].dropna().iloc[-1] == pytest.approx(150.0)


def test_prev_close_1_is_the_candle_before_the_setup_candle():
    ctx = build_context(bars(rising_market(10)), cfg())
    closes = bars(rising_market(10))["Close"]
    pd.testing.assert_series_equal(
        ctx["prev_close_1"], closes.shift(1), check_names=False
    )


# ------------------------------------------------------------------ setups ---

def test_bull_setup_needs_volume_above_the_threshold():
    quiet = detect_setups(build_context(bars(rising_market(30, 100.0)), cfg()), cfg())
    assert not quiet["bull_setup"].any(), "flat volume must not produce setups"

    rows = rising_market(30, 100.0)
    rows[-2] = candle("down", 2140, body=3.0, wick=1.0, volume=100.0)  # satisfy pullback
    rows[-1] = candle("up", 2145, body=4.0, wick=1.0, volume=400.0)    # volume spike
    loud = detect_setups(build_context(bars(rows), cfg()), cfg())
    assert bool(loud["bull_setup"].iloc[-1])


def test_pullback_filter_rejects_a_fourth_consecutive_candle():
    """Pine: not (the three preceding candles all closed the same way)."""
    rows = rising_market(30, 100.0)
    rows[-1] = candle("up", 2145, body=4.0, wick=1.0, volume=400.0)

    with_filter = detect_setups(build_context(bars(rows), cfg()), cfg(require_pullback=True))
    without = detect_setups(build_context(bars(rows), cfg()), cfg(require_pullback=False))

    # Every prior candle is bullish here, so the filter must veto the setup.
    assert not bool(with_filter["bull_setup"].iloc[-1])
    assert bool(without["bull_setup"].iloc[-1])


def test_bull_setup_requires_close_above_the_trend_ema():
    rows = falling_market(30, 100.0)
    rows[-1] = candle("up", 1860, body=4.0, wick=1.0, volume=400.0)
    out = detect_setups(build_context(bars(rows), cfg()), cfg())
    assert not bool(out["bull_setup"].iloc[-1]), "below the trend EMA is not a long setup"


def test_long_level_maths_match_the_source_formula():
    rows = rising_market(30, 100.0)
    rows[-3] = candle("down", 2135, body=6.0, wick=1.0, volume=100.0)  # break the run
    rows[-1] = (2140.0, 2160.0, 2130.0, 2155.0, 400.0)
    frame = bars(rows)
    out = detect_setups(build_context(frame, cfg()), cfg(long_retracement=0.40))
    row = out.iloc[-1]
    assert bool(row["bull_setup"])

    high = 2160.0
    low = min(2130.0, frame["Close"].iloc[-2])   # min(lowC, closeC2)
    assert row["setup_high"] == pytest.approx(high)
    assert row["setup_low"] == pytest.approx(low)
    assert row["setup_trigger"] == pytest.approx(high - (high - low) * 0.40)


def test_short_level_maths_use_the_sixty_percent_retracement():
    rows = falling_market(30, 100.0)
    rows[-3] = candle("up", 1865, body=6.0, wick=1.0, volume=100.0)
    rows[-1] = (1875.0, 1880.0, 1850.0, 1860.0, 400.0)  # bearish body, volume spike
    frame = bars(rows)
    out = detect_setups(build_context(frame, cfg()), cfg(short_retracement=0.60))
    row = out.iloc[-1]
    assert bool(row["bear_setup"])

    low = 1850.0
    high = max(1880.0, frame["Close"].iloc[-2])  # max(highC, closeC2)
    assert row["setup_trigger"] == pytest.approx(high - (high - low) * 0.60)
    # 60% down from the high is 40% up from the low.
    assert row["setup_trigger"] == pytest.approx(low + (high - low) * 0.40)


def test_disabling_a_side_removes_its_setups():
    rows = rising_market(30, 100.0)
    rows[-3] = candle("down", 2135, body=6.0, wick=1.0, volume=100.0)
    rows[-1] = (2140.0, 2160.0, 2130.0, 2155.0, 400.0)
    out = detect_setups(build_context(bars(rows), cfg()), cfg(allow_long=False))
    assert not out["bull_setup"].any()


# ---------------------------------------------------------- state machine ---

def _armed(direction: int, trigger: float, n: int = 10, high=2100.0, low=2000.0):
    """An entry frame with one setup armed from the first bar."""
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {"Open": 2050.0, "High": 2060.0, "Low": 2040.0, "Close": 2050.0, "Volume": 10.0,
         "context_open_time": pd.Timestamp("2026-01-01", tz="UTC"),
         "setup_dir": 0, "setup_high": np.nan, "setup_low": np.nan,
         "setup_trigger": np.nan},
        index=idx,
    )
    df.iloc[0, df.columns.get_loc("setup_dir")] = direction
    df.iloc[0, df.columns.get_loc("setup_high")] = high
    df.iloc[0, df.columns.get_loc("setup_low")] = low
    df.iloc[0, df.columns.get_loc("setup_trigger")] = trigger
    return df


def test_long_triggers_when_the_low_touches_the_level():
    df = _armed(LONG, trigger=2030.0)
    df.iloc[4, df.columns.get_loc("Low")] = 2029.0

    out = generate_signals(df, cfg())
    assert out["signal"].iloc[4] == LONG
    assert (out["signal"].drop(out.index[4]) == 0).all()


def test_short_triggers_when_the_high_touches_the_level():
    df = _armed(SHORT, trigger=2070.0)
    df.iloc[3, df.columns.get_loc("High")] = 2071.0

    out = generate_signals(df, cfg())
    assert out["signal"].iloc[3] == SHORT


def test_a_setup_fires_only_once():
    df = _armed(LONG, trigger=2030.0)
    df.iloc[[4, 5, 6], df.columns.get_loc("Low")] = 2029.0

    out = generate_signals(df, cfg())
    assert (out["signal"] != 0).sum() == 1


def test_setup_stays_armed_until_it_triggers():
    df = _armed(LONG, trigger=2000.0)  # never touched
    out = generate_signals(df, cfg())
    assert (out["signal"] == 0).all()
    assert (out["active_dir"] == LONG).all(), "setup must persist, as in the original"


def test_opposite_setup_replaces_a_pending_one():
    n = 12
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    first = pd.Timestamp("2026-01-01", tz="UTC")
    later = pd.Timestamp("2026-01-01 01:00", tz="UTC")

    df = pd.DataFrame(
        {
            "Open": 2050.0, "High": 2060.0, "Low": 2040.0, "Close": 2050.0, "Volume": 10.0,
            "context_open_time": [first] * 6 + [later] * (n - 6),
            "setup_dir": [LONG] + [0] * 5 + [SHORT] + [0] * (n - 7),
            "setup_high": [2100.0] + [np.nan] * 5 + [2100.0] + [np.nan] * (n - 7),
            "setup_low": [2000.0] + [np.nan] * 5 + [2000.0] + [np.nan] * (n - 7),
            # Neither level is reachable, so both setups stay pending.
            "setup_trigger": [2000.0] + [np.nan] * 5 + [2100.0] + [np.nan] * (n - 7),
        },
        index=idx,
    )

    out = generate_signals(df, cfg())
    assert out["active_dir"].iloc[5] == LONG
    assert out["active_dir"].iloc[7] == SHORT
    assert (out["signal"] == 0).all()


def test_expiry_disarms_a_stale_setup():
    df = _armed(LONG, trigger=2000.0, n=12)
    for k, hour in enumerate([1, 2, 3], start=1):
        stamp = pd.Timestamp(f"2026-01-01 0{hour}:00", tz="UTC")
        df.iloc[2 * k + 2:, df.columns.get_loc("context_open_time")] = stamp

    kept = generate_signals(df, cfg(setup_expiry_bars=0))
    dropped = generate_signals(df, cfg(setup_expiry_bars=1))
    assert (kept["active_dir"] != 0).sum() > (dropped["active_dir"] != 0).sum()


def test_immediate_trigger_is_flagged():
    """Price already through the level when the setup arms: no retracement."""
    df = _armed(LONG, trigger=2045.0)  # Low is 2040 on every bar
    out = generate_signals(df, cfg())
    assert out["signal"].iloc[0] == LONG
    assert bool(out["trigger_immediate"].iloc[0])


def test_a_real_retracement_is_not_flagged_immediate():
    df = _armed(LONG, trigger=2030.0)
    df.iloc[5, df.columns.get_loc("Low")] = 2029.0
    out = generate_signals(df, cfg())
    assert not bool(out["trigger_immediate"].iloc[5])


# ------------------------------------------------------------ swing points ---

def test_swing_low_is_only_published_after_confirmation():
    """A pivot needs `right` bars after it, so it cannot be used before then."""
    from tradegold.strategy import confirmed_swings

    low  = np.array([10, 9, 8, 5, 8, 9, 10, 11, 12], dtype=float)
    high = np.array([12, 11, 10, 7, 10, 11, 12, 13, 14], dtype=float)
    sl, _ = confirmed_swings(high, low, left=2, right=2)

    assert np.isnan(sl[:5]).all(), "pivot at index 3 must not exist before index 5"
    assert sl[5] == pytest.approx(5.0), "confirmed at 3 + right(2) = 5"
    assert sl[6:].min() == pytest.approx(5.0), "and carried forward"


def test_swing_detection_is_causal():
    rng = np.random.default_rng(9)
    n = 200
    low = 100 + np.cumsum(rng.normal(0, 1, n))
    high = low + 2
    from tradegold.strategy import confirmed_swings

    base_l, base_h = confirmed_swings(high, low, 3, 3)
    low2, high2 = low.copy(), high.copy()
    low2[-1] = -999                       # a dramatic future low
    tampered_l, _ = confirmed_swings(high2, low2, 3, 3)

    np.testing.assert_array_equal(
        np.nan_to_num(base_l[:-4], nan=-1), np.nan_to_num(tampered_l[:-4], nan=-1)
    )


def test_swing_high_mirrors_swing_low():
    from tradegold.strategy import confirmed_swings

    high = np.array([10, 11, 12, 15, 12, 11, 10, 9, 8], dtype=float)
    low = high - 2
    _, sh = confirmed_swings(high, low, left=2, right=2)
    assert np.isnan(sh[:5]).all()
    assert sh[5] == pytest.approx(15.0)


def test_flat_series_produces_no_pivots():
    """Ties are not pivots -- otherwise a flat market invents levels."""
    from tradegold.strategy import confirmed_swings

    flat = np.full(50, 100.0)
    sl, sh = confirmed_swings(flat, flat, 3, 3)
    assert np.isnan(sl).all() and np.isnan(sh).all()


def test_short_series_returns_all_nan_without_error():
    from tradegold.strategy import confirmed_swings

    sl, sh = confirmed_swings(np.arange(4.0), np.arange(4.0), 3, 3)
    assert len(sl) == 4 and np.isnan(sl).all() and np.isnan(sh).all()


def test_out_of_session_triggers_are_discarded_not_queued():
    """Blocking an hour must not stockpile setups that flush when it reopens."""
    n = 24
    idx = pd.date_range("2026-01-01 00:00", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"Open": 2050.0, "High": 2060.0, "Low": 2020.0, "Close": 2050.0, "Volume": 10.0,
         "context_open_time": pd.Timestamp("2026-01-01", tz="UTC"),
         "setup_dir": 0, "setup_high": np.nan, "setup_low": np.nan,
         "setup_trigger": np.nan},
        index=idx,
    )
    # A setup arms at 02:00; its level is touched on every subsequent bar.
    df.iloc[2, df.columns.get_loc("setup_dir")] = LONG
    df.iloc[2, df.columns.get_loc("setup_high")] = 2100.0
    df.iloc[2, df.columns.get_loc("setup_low")] = 2000.0
    df.iloc[2, df.columns.get_loc("setup_trigger")] = 2030.0
    df.iloc[2:, df.columns.get_loc("context_open_time")] = pd.Timestamp(
        "2026-01-01 02:00", tz="UTC")

    blocked = generate_signals(df, cfg(allowed_hours=[10, 11, 12]))
    assert (blocked["signal"] != 0).sum() == 0, "must be discarded, not held"
    assert (blocked.loc["2026-01-01 10:00":, "signal"] != 0).sum() == 0, \
        "and must not fire when the session reopens"

    allowed = generate_signals(df, cfg(allowed_hours=list(range(24))))
    assert (allowed["signal"] != 0).sum() == 1
