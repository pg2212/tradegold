"""Context alignment is the one place this project could acquire lookahead bias."""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import bars, candle

from tradegold.data import align_context


def _context(n: int = 5) -> pd.DataFrame:
    ctx = bars([candle("up", 2000 + i, volume=100) for i in range(n)], freq="h")
    ctx["marker"] = range(n)  # identifies which context bar a row is showing
    return ctx[["marker"]]


def test_entry_bar_sees_only_the_previous_closed_context_bar():
    ctx = _context(4)
    entry = bars([candle("flat") for _ in range(48)], freq="5min")

    aligned = align_context(entry, ctx, "60m")

    # The 60m bar opening at 00:00 closes at 01:00, so 01:00-01:55 must see
    # marker 0 -- never marker 1, whose candle is still forming.
    window = aligned.loc["2026-01-01 01:00":"2026-01-01 01:55", "marker"]
    assert (window == 0).all()
    later = aligned.loc["2026-01-01 02:00":"2026-01-01 02:55", "marker"]
    assert (later == 1).all()


def test_bars_before_the_first_closed_context_bar_are_dropped():
    ctx = _context(3)
    entry = bars([candle("flat") for _ in range(36)], freq="5min")

    aligned = align_context(entry, ctx, "60m")
    # 00:00-00:55 have no fully closed context candle yet.
    assert aligned.index[0] == pd.Timestamp("2026-01-01 01:00", tz="UTC")


def test_context_value_never_precedes_its_own_close():
    ctx = _context(6)
    entry = bars([candle("flat") for _ in range(72)], freq="5min")

    aligned = align_context(entry, ctx, "60m")
    available_from = aligned["context_open_time"] + pd.Timedelta(1, unit="h")
    assert (available_from <= aligned.index).all()


def test_a_future_context_bar_cannot_change_the_past():
    """Mutating the last context row must not alter any earlier alignment."""
    ctx = _context(5)
    entry = bars([candle("flat") for _ in range(60)], freq="5min")
    baseline = align_context(entry, ctx, "60m")

    tampered = ctx.copy()
    tampered.iloc[-1, 0] = 999
    shifted = align_context(entry, tampered, "60m")

    common = baseline.index.intersection(shifted.index)[:-12]
    pd.testing.assert_series_equal(
        baseline.loc[common, "marker"], shifted.loc[common, "marker"]
    )


def test_alignment_rejects_a_finer_context_than_entry():
    from tradegold.config import DataConfig

    with pytest.raises(ValueError, match="must be finer"):
        DataConfig(entry_interval="60m", context_interval="5m")
