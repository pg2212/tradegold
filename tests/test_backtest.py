"""Fill model and trade accounting for both sides."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradegold.backtest import run_backtest
from tradegold.config import ExecutionConfig
from tradegold.metrics import trade_metrics
from tradegold.strategy import LONG, SHORT


def frame(n=20, direction=LONG, trigger=2030.0, high=2100.0, low=2000.0,
          signal_at=3, immediate=False) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {"Open": 2050.0, "High": 2055.0, "Low": 2045.0, "Close": 2050.0, "Volume": 10.0,
         "signal": 0, "trigger_immediate": immediate,
         "active_high": high, "active_low": low, "active_trigger": trigger,
         # The last completed context candle, which `prev_candle` exits trail.
         "highC": high, "lowC": low,
         "context_open_time": pd.Timestamp("2026-01-01", tz="UTC")},
        index=idx,
    )
    df.iloc[signal_at, df.columns.get_loc("signal")] = direction
    return df


def cfg(**kw) -> ExecutionConfig:
    return ExecutionConfig(**{"cost_per_side": 0.25, **kw})


def test_next_open_fill_uses_the_bar_after_the_signal():
    df = frame(signal_at=3)
    df.iloc[4, df.columns.get_loc("Open")] = 2031.0

    t = run_backtest(df, cfg(entry_fill="next_open")).trades.iloc[0]
    assert t["entry_time"] == df.index[4]
    assert t["entry_price"] == pytest.approx(2031.0)


def test_level_fill_cannot_beat_the_level_without_a_gap():
    df = frame(signal_at=3)
    df.iloc[3, df.columns.get_loc("Open")] = 2050.0  # opened above the 2030 level

    t = run_backtest(df, cfg(entry_fill="level")).trades.iloc[0]
    assert t["entry_price"] == pytest.approx(2030.0), "must fill at the level, not the open"


def test_level_fill_uses_the_open_when_the_bar_gapped_through():
    df = frame(signal_at=3)
    df.iloc[3, df.columns.get_loc("Open")] = 2020.0  # gapped below the level

    t = run_backtest(df, cfg(entry_fill="level")).trades.iloc[0]
    assert t["entry_price"] == pytest.approx(2020.0)


def test_immediate_triggers_are_skipped_by_default():
    df = frame(signal_at=3, immediate=True)
    assert run_backtest(df, cfg()).n_trades == 0
    assert run_backtest(df, cfg(skip_immediate_triggers=False)).n_trades == 1


def test_long_target_and_stop_come_from_the_setup_extremes():
    df = frame(direction=LONG, high=2100.0, low=2000.0, signal_at=2)
    df.iloc[3, df.columns.get_loc("Open")] = 2030.0
    df.iloc[6, df.columns.get_loc("High")] = 2101.0  # touch the target

    t = run_backtest(df, cfg(entry_fill="next_open")).trades.iloc[0]
    assert t["stop_price"] == pytest.approx(2000.0)
    assert t["target_price"] == pytest.approx(2100.0)
    assert t["exit_reason"] == "target"
    assert t["net_pnl"] == pytest.approx(2100.0 - 2030.0 - 0.5)


def test_short_pnl_is_positive_when_price_falls():
    df = frame(direction=SHORT, trigger=2070.0, high=2100.0, low=2000.0, signal_at=2)
    df.iloc[3, df.columns.get_loc("Open")] = 2070.0
    df.iloc[6, df.columns.get_loc("Low")] = 1999.0  # touch the target

    t = run_backtest(df, cfg(entry_fill="next_open")).trades.iloc[0]
    assert t["side"] == "short"
    assert t["stop_price"] == pytest.approx(2100.0)
    assert t["target_price"] == pytest.approx(2000.0)
    assert t["net_pnl"] == pytest.approx(2070.0 - 2000.0 - 0.5)


def test_both_barriers_in_one_bar_takes_the_loss():
    df = frame(direction=LONG, signal_at=2)
    df.iloc[3, df.columns.get_loc("Open")] = 2030.0
    df.iloc[5, df.columns.get_loc("High")] = 2101.0
    df.iloc[5, df.columns.get_loc("Low")] = 1999.0

    t = run_backtest(df, cfg(entry_fill="next_open")).trades.iloc[0]
    assert t["exit_reason"] == "stop_loss"
    assert t["net_pnl"] < 0


def test_r_multiple_target_sizes_from_the_stop():
    df = frame(direction=LONG, signal_at=2, low=2000.0)
    df.iloc[3, df.columns.get_loc("Open")] = 2030.0

    exec_cfg = cfg(entry_fill="next_open", target="r_multiple", r_multiple=2.0)
    t = run_backtest(df, exec_cfg).trades.iloc[0]
    risk = 2030.0 - 2000.0
    assert t["target_price"] == pytest.approx(2030.0 + 2 * risk)


def test_max_holding_forces_a_time_exit():
    df = frame(direction=LONG, signal_at=2, high=9999.0, low=1.0)
    df.iloc[3, df.columns.get_loc("Open")] = 2030.0

    t = run_backtest(df, cfg(entry_fill="next_open", max_holding_bars=4)).trades.iloc[0]
    assert t["bars_held"] == 4
    assert t["exit_reason"] == "time_exit"


def test_equity_curve_reconciles_with_the_ledger():
    rng = np.random.default_rng(2)
    n = 400
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    close = 2050 + np.cumsum(rng.normal(0, 1.5, n))
    df = pd.DataFrame(
        {"Open": close, "High": close + 3, "Low": close - 3, "Close": close, "Volume": 10.0,
         "signal": 0, "trigger_immediate": False,
         "active_high": close + 25, "active_low": close - 25, "active_trigger": close,
         "highC": close + 25, "lowC": close - 25,
         "context_open_time": pd.Timestamp("2026-01-01", tz="UTC")},
        index=idx,
    )
    fire = rng.choice(n - 2, 25, replace=False)
    df.iloc[fire, df.columns.get_loc("signal")] = rng.choice([LONG, SHORT], 25)

    result = run_backtest(df, cfg(entry_fill="next_open"))
    assert result.trades["net_pnl"].sum() == pytest.approx(result.equity["equity"].iloc[-1])


def test_no_signals_gives_an_empty_but_valid_result():
    df = frame(signal_at=3)
    df["signal"] = 0
    result = run_backtest(df, cfg())
    assert result.n_trades == 0
    assert result.equity["equity"].iloc[-1] == 0.0
    assert trade_metrics(result.trades)["n_trades"] == 0



def test_prev_candle_stop_trails_in_the_trades_favour():
    """As new context candles close, a long's stop should ratchet upward only."""
    df = frame(direction=LONG, signal_at=2, high=2200.0, low=1900.0, trigger=2050.0)
    df.iloc[3, df.columns.get_loc("Open")] = 2050.0
    # Later bars see a context candle whose low has risen well above the setup's.
    df.iloc[5:, df.columns.get_loc("lowC")] = 2040.0
    df.iloc[7, df.columns.get_loc("Low")] = 2039.0     # breaks the trailed stop

    t = run_backtest(df, cfg(entry_fill="next_open", stop="prev_candle")).trades.iloc[0]
    assert t["stop_price"] == pytest.approx(2040.0), "stop must follow the newer candle"
    assert t["exit_reason"] == "stop_loss"
    assert t["exit_idx"] == 7


def test_prev_candle_stop_never_loosens():
    """A context candle with a lower low must not widen an existing stop."""
    df = frame(direction=LONG, signal_at=2, high=2200.0, low=2000.0, trigger=2050.0)
    df.iloc[3, df.columns.get_loc("Open")] = 2050.0
    df.iloc[5:, df.columns.get_loc("lowC")] = 1500.0   # much looser
    df.iloc[7, df.columns.get_loc("Low")] = 1999.0     # breaks the ORIGINAL stop

    t = run_backtest(df, cfg(entry_fill="next_open", stop="prev_candle")).trades.iloc[0]
    assert t["stop_price"] == pytest.approx(2000.0), "stop must not move against the trade"
    assert t["exit_reason"] == "stop_loss"


def test_prev_candle_stop_mirrors_for_shorts():
    df = frame(direction=SHORT, signal_at=2, high=2100.0, low=1900.0, trigger=2050.0)
    df.iloc[3, df.columns.get_loc("Open")] = 2050.0
    df.iloc[5:, df.columns.get_loc("highC")] = 2060.0  # tighter than the setup high
    df.iloc[7, df.columns.get_loc("High")] = 2061.0

    t = run_backtest(df, cfg(entry_fill="next_open", stop="prev_candle")).trades.iloc[0]
    assert t["stop_price"] == pytest.approx(2060.0)
    assert t["exit_reason"] == "stop_loss"


def test_prev_candle_target_tracks_the_latest_candle():
    df = frame(direction=LONG, signal_at=2, high=2200.0, low=2000.0, trigger=2050.0)
    df.iloc[3, df.columns.get_loc("Open")] = 2050.0
    df.iloc[5:, df.columns.get_loc("highC")] = 2070.0  # nearer than the setup high
    df.iloc[6, df.columns.get_loc("High")] = 2071.0

    t = run_backtest(df, cfg(entry_fill="next_open", target="prev_candle")).trades.iloc[0]
    assert t["target_price"] == pytest.approx(2070.0)
    assert t["exit_reason"] == "target"


def test_fixed_modes_are_unaffected_by_later_candles():
    """setup_extreme must ignore context candles that close after entry."""
    df = frame(direction=LONG, signal_at=2, high=2200.0, low=2000.0, trigger=2050.0)
    df.iloc[3, df.columns.get_loc("Open")] = 2050.0
    df.iloc[5:, [df.columns.get_loc("highC"), df.columns.get_loc("lowC")]] = [2060.0, 2040.0]
    df.iloc[7, df.columns.get_loc("Low")] = 2039.0     # would break a trailed stop

    t = run_backtest(df, cfg(entry_fill="next_open")).trades.iloc[0]
    assert t["stop_price"] == pytest.approx(2000.0)
    assert t["exit_reason"] != "stop_loss"


def frame_sw(**kw):
    """A frame carrying swing levels for structural-stop tests."""
    df = frame(**kw)
    df["swing_low"] = np.nan
    df["swing_high"] = np.nan
    return df


def test_swing_stop_uses_the_confirmed_pivot_with_a_buffer():
    df = frame_sw(direction=LONG, signal_at=2, high=2100.0, low=2000.0, trigger=2050.0)
    df.iloc[3, df.columns.get_loc("Open")] = 2050.0
    df["swing_low"] = 2030.0                      # tighter than the setup low

    t = run_backtest(df, cfg(entry_fill="next_open", stop="swing",
                             swing_buffer=0.05)).trades.iloc[0]
    # 2030 minus 5% of the 100-wide setup range
    assert t["stop_price"] == pytest.approx(2025.0)


def test_swing_stop_falls_back_when_no_pivot_is_confirmed():
    df = frame_sw(direction=LONG, signal_at=2, high=2100.0, low=2000.0, trigger=2050.0)
    df.iloc[3, df.columns.get_loc("Open")] = 2050.0   # swing_low stays NaN

    t = run_backtest(df, cfg(entry_fill="next_open", stop="swing",
                             swing_buffer=0.0)).trades.iloc[0]
    assert t["stop_price"] == pytest.approx(2000.0), "must fall back to the setup low"


def test_swing_stop_mirrors_for_shorts():
    df = frame_sw(direction=SHORT, signal_at=2, high=2100.0, low=2000.0, trigger=2050.0)
    df.iloc[3, df.columns.get_loc("Open")] = 2050.0
    df["swing_high"] = 2070.0

    t = run_backtest(df, cfg(entry_fill="next_open", stop="swing",
                             swing_buffer=0.0)).trades.iloc[0]
    assert t["stop_price"] == pytest.approx(2070.0)


def test_min_reward_risk_rejects_poor_geometry():
    """Entering 40% down from the high pays 0.67R, so a 1.5 floor must veto it."""
    # high 2100, low 2000, entry 2060 -> risk 60, reward 40, R:R 0.667
    df = frame_sw(direction=LONG, signal_at=2, high=2100.0, low=2000.0, trigger=2060.0)
    df.iloc[3, df.columns.get_loc("Open")] = 2060.0

    assert run_backtest(df, cfg(entry_fill="next_open")).n_trades == 1
    assert run_backtest(df, cfg(entry_fill="next_open", min_reward_risk=1.5)).n_trades == 0
    # A floor below the actual geometry lets it through.
    assert run_backtest(df, cfg(entry_fill="next_open", min_reward_risk=0.5)).n_trades == 1


def test_min_reward_risk_allows_good_geometry():
    df = frame_sw(direction=LONG, signal_at=2, high=2100.0, low=2000.0, trigger=2060.0)
    df.iloc[3, df.columns.get_loc("Open")] = 2060.0
    df["swing_low"] = 2040.0        # risk 20, reward to 2100 is 40 -> 2R

    t = run_backtest(df, cfg(entry_fill="next_open", stop="swing", swing_buffer=0.0,
                             min_reward_risk=1.5)).trades.iloc[0]
    assert t["stop_price"] == pytest.approx(2040.0)
