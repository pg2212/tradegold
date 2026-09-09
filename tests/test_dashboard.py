"""Dashboard plumbing, against a synthetic run directory."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tradegold.dashboard import context_chart, discover_runs, entry_chart, load_run, setup_id
from tradegold.strategy import LONG, SHORT


@pytest.fixture
def fake_run(tmp_path):
    d = tmp_path / "20260101T000000Z__demo__abc123def456"
    d.mkdir()
    n = 120
    rng = np.random.default_rng(4)
    close = 2000 + np.cumsum(rng.normal(0, 1.0, n))
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")

    signals = pd.DataFrame(
        {"Open": close, "High": close + 2, "Low": close - 2, "Close": close,
         "Volume": 100.0, "ema_fast": close, "ema_slow": close - 1, "ema_trend": close - 2,
         "signal": 0, "trigger_immediate": False, "active_dir": 0,
         "active_high": np.nan, "active_low": np.nan, "active_trigger": np.nan,
         "context_open_time": idx[0]},
        index=idx,
    )
    # Two distinct setups so the segment logic has something to separate.
    for lo, hi, direction, trig in ((10, 40, LONG, 1995.0), (60, 90, SHORT, 2010.0)):
        signals.iloc[lo:hi, signals.columns.get_loc("active_dir")] = direction
        signals.iloc[lo:hi, signals.columns.get_loc("active_high")] = 2020.0
        signals.iloc[lo:hi, signals.columns.get_loc("active_low")] = 1990.0
        signals.iloc[lo:hi, signals.columns.get_loc("active_trigger")] = trig
    signals.iloc[39, signals.columns.get_loc("signal")] = LONG
    signals.iloc[89, signals.columns.get_loc("signal")] = SHORT
    signals.iloc[89, signals.columns.get_loc("trigger_immediate")] = True
    signals.to_parquet(d / "signals.parquet")

    ctx_idx = pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC")
    context = pd.DataFrame(
        {"openC": 2000.0, "highC": 2010.0, "lowC": 1990.0, "closeC": 2005.0,
         "volumeC": 500.0, "vol_ma": 200.0, "vol_threshold": 300.0,
         "ema_fast": 2004.0, "ema_slow": 2003.0, "ema_trend": 2002.0,
         "bull_setup": [True] + [False] * 11, "bear_setup": [False] * 11 + [True],
         "setup_trigger": 1998.0, "setup_high": 2010.0, "setup_low": 1990.0},
        index=ctx_idx,
    )
    context.to_parquet(d / "context.parquet")

    trades = pd.DataFrame({
        "direction": [LONG], "side": ["long"], "signal_time": [idx[39]],
        "entry_time": [idx[40]], "exit_time": [idx[50]], "entry_idx": [40], "exit_idx": [50],
        "entry_price": [1995.0], "exit_price": [2020.0], "stop_price": [1990.0],
        "target_price": [2020.0], "setup_time": [ctx_idx[0]], "trigger_level": [1995.0],
        "bars_held": [11], "gross_pnl": [25.0], "costs": [0.5], "net_pnl": [24.5],
        "r_multiple": [4.9], "exit_reason": ["target"],
    })
    trades.to_csv(d / "trades.csv")

    pd.DataFrame({"close": close, "bar_pnl": 0.0, "equity": np.linspace(0, 24.5, n),
                  "buy_hold": close - close[0], "in_position": False}, index=idx
                 ).to_csv(d / "equity.csv")

    (d / "config.yaml").write_text("run:\n  name: demo\n")
    (d / "metrics.json").write_text(json.dumps({
        "setups": {"setups_armed": 2, "setups_triggered": 2, "fill_rate": 1.0,
                   "bars_armed_pct": 0.5},
        "trades": {"n_trades": 1, "win_rate": 1.0, "profit_factor": None},
        "equity": {"net_pnl": 24.5, "max_drawdown": 0.0},
        "data": {"entry_interval": "5m", "context_interval": "60m"},
    }))
    return tmp_path, d


def test_discover_runs_requires_signal_artifacts(fake_run):
    root, _ = fake_run
    (root / "20260101T000000Z__empty__deadbeef0000").mkdir()
    runs = discover_runs(root)
    assert len(runs) == 1 and runs[0].name == "demo"


def test_setup_id_separates_consecutive_setups(fake_run):
    _, d = fake_run
    signals, *_ = load_run(str(d))
    groups = setup_id(signals)
    assert groups.dropna().nunique() == 2
    assert groups.isna().sum() > 0  # unarmed bars carry no group


def test_entry_chart_draws_candles_emas_and_levels(fake_run):
    _, d = fake_run
    signals, _, trades, equity, _ = load_run(str(d))
    fig = entry_chart(signals, trades, equity, show_emas=True, show_levels=True,
                      show_volume=True, dark=True)

    names = [t.name for t in fig.data]
    assert any(t.type == "candlestick" for t in fig.data)
    assert "EMA 10 (60m)" in names and "EMA 50 (60m)" in names
    assert "Setup high" in names and "Entry level (retracement)" in names
    # The immediate trigger must be surfaced, not silently dropped.
    assert any("no retracement" in str(n) for n in names)


def test_entry_chart_respects_toggles(fake_run):
    _, d = fake_run
    signals, _, trades, equity, _ = load_run(str(d))
    bare = entry_chart(signals, trades, equity, show_emas=False, show_levels=False,
                       show_volume=False, dark=False)
    names = [t.name for t in bare.data]
    assert not any("EMA" in str(n) for n in names)
    assert not any(n == "Setup high" for n in names)


def test_level_segments_are_broken_between_setups(fake_run):
    _, d = fake_run
    signals, _, trades, equity, _ = load_run(str(d))
    fig = entry_chart(signals, trades, equity, show_emas=False, show_levels=True,
                      show_volume=False, dark=True)
    trace = next(t for t in fig.data if t.name == "Setup high")
    assert any(v is None for v in trace.y), "setups must not be joined by a line"


def test_context_chart_marks_setups_and_the_volume_gate(fake_run):
    _, d = fake_run
    _, context, *_ = load_run(str(d))
    fig = context_chart(context, dark=True)
    names = [str(t.name) for t in fig.data]
    assert any("Bullish setup" in n for n in names)
    assert any("Bearish setup" in n for n in names)
    assert "Setup threshold" in names and "Volume average" in names


def test_export_figure_normalises_every_timestamp(fake_run, tmp_path):
    """Static export serialises via orjson, which rejects pandas Timestamps.

    Checking only the shapes is not enough -- trace x-arrays carry them too.
    """
    import copy

    import pandas as pd

    from tradegold.dashboard import export_figure

    _, d = fake_run
    signals, _, trades, equity, _ = load_run(str(d))
    fig = entry_chart(signals, trades, equity, show_emas=True, show_levels=True,
                      show_volume=True, dark=True)

    # The interactive figure legitimately holds Timestamps...
    assert any(
        isinstance(v, pd.Timestamp)
        for t in fig.data if t.x is not None for v in list(t.x)[:5]
    )

    # ...and export_figure must leave none behind on its copy.
    captured = {}

    def fake_write(self, path, **kwargs):
        captured["fig"] = self

    monkey = copy.copy(type(fig).write_image)
    type(fig).write_image = fake_write
    try:
        export_figure(fig, tmp_path / "out.png")
    finally:
        type(fig).write_image = monkey

    out = captured["fig"]
    for t in out.data:
        if t.x is not None:
            assert not any(isinstance(v, pd.Timestamp) for v in t.x)
    for shape in out.layout.shapes:
        assert not isinstance(shape.x0, pd.Timestamp)
        assert not isinstance(shape.x1, pd.Timestamp)


def test_export_figure_makes_room_for_a_title(fake_run, tmp_path):
    """A figure title sits in the same band as the legend; both must fit."""
    import copy

    from tradegold.dashboard import export_figure

    _, d = fake_run
    signals, _, trades, equity, _ = load_run(str(d))
    fig = entry_chart(signals, trades, equity, show_emas=True, show_levels=True,
                      show_volume=False, dark=True)
    fig.update_layout(title=dict(text="A title"))
    original_top = fig.layout.margin.t

    captured = {}
    keep = copy.copy(type(fig).write_image)
    type(fig).write_image = lambda self, path, **kw: captured.__setitem__("fig", self)
    try:
        export_figure(fig, tmp_path / "out.png")
    finally:
        type(fig).write_image = keep

    out = captured["fig"]
    assert out.layout.margin.t >= 118
    assert out.layout.legend.y > 1.0
    assert fig.layout.margin.t == original_top, "the source figure must not be mutated"


def test_metric_values_stay_short_enough_not_to_truncate(fake_run):
    """Streamlit truncates a metric value that overflows its column."""
    from tradegold.dashboard import money

    # Widest realistic figures: five-digit PnL with a sign and a currency mark.
    assert len(money(-12345.0, 0)) <= 8
    assert len(money(156.3, 0)) <= 6


def test_chart_labels_follow_the_run_not_a_hardcoded_timeframe(fake_run):
    """A 15m run must not be labelled 5m."""
    _, d = fake_run
    signals, _, trades, equity, _ = load_run(str(d))
    fig = entry_chart(signals, trades, equity, show_emas=True, show_levels=False,
                      show_volume=True, dark=True,
                      entry_interval="15m", context_interval="4h")

    names = [str(t.name) for t in fig.data]
    assert "EMA 10 (4h)" in names and "EMA 50 (4h)" in names
    titles = [a.text for a in fig.layout.annotations]
    assert any("Volume (15m)" in t for t in titles)
    assert not any("5m" in n and "15m" not in n for n in names)


def test_walk_forward_chart_colours_by_sign(fake_run):
    import pandas as pd

    from tradegold.dashboard import walk_forward_chart

    rows = pd.DataFrame({
        "fold": ["fold_1", "fold_2"],
        "test_start": pd.to_datetime(["2025-01-01", "2025-04-01"], utc=True),
        "test_end": pd.to_datetime(["2025-03-31", "2025-06-30"], utc=True),
        "test_n": [10, 12], "test_net": [25.0, -18.0], "test_win": [0.6, 0.4],
    })
    fig = walk_forward_chart(rows, dark=True)
    colors = list(fig.data[0].marker.color)
    assert colors[0] != colors[1], "gain and loss must not share a colour"
