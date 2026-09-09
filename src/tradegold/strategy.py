"""GOLD STRATEGY: volume-expansion setups with a retracement entry.

A faithful port of the TradingView indicator. The shape of it:

1.  On the higher timeframe, a *setup* candle is one that closed with volume
    well above its own recent average, in the direction of its trend EMA, and
    which is not merely the fourth leg of a move already three candles old.
2.  That candle's range defines three levels: its extremes, and a retracement
    level inside it.
3.  On the lower timeframe the strategy then waits, possibly for hours, for
    price to pull back into that retracement level. Touching it is the entry.
4.  A setup is consumed by its trigger, or replaced by an opposite setup. It is
    never re-armed.

Two faithfulness notes, both of which change results if "corrected":

  * The source comment claims the volume gate is `volMA90` (3x the average) but
    the code tests `volMA45` (1.5x). The code is what ran, so the code is what
    is ported; `strategy.volume_multiplier` exposes it.
  * Longs retrace 40% down from the setup high, shorts 60% down from it
    (= 40% up from the low). The asymmetry is in the original and is preserved
    as two separate settings rather than silently symmetrised.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .logging_utils import get_logger

logger = get_logger(__name__)

LONG, SHORT, FLAT = 1, -1, 0

CONTEXT_COLUMNS = [
    "openC", "highC", "lowC", "closeC", "volumeC", "efficiency", "atr_pctile",
    "ema_fast", "ema_slow", "ema_trend", "vol_ma", "vol_threshold",
    "prev_close_1", "bull_1", "bull_2", "bull_3",
    "bull_setup", "bear_setup",
    "setup_high", "setup_low", "setup_trigger", "setup_dir",
]


@dataclass(frozen=True)
class Setup:
    """An armed setup waiting for its retracement."""

    direction: int
    context_time: pd.Timestamp
    armed_at: pd.Timestamp
    high: float
    low: float
    trigger: float

    @property
    def stop(self) -> float:
        # Far side of the setup range: below a long, above a short.
        return self.low if self.direction == LONG else self.high

    @property
    def target(self) -> float:
        return self.high if self.direction == LONG else self.low


def build_context(bars: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Indicator state on the context timeframe.

    Row `c` holds everything knowable once candle `c` has closed: its own OHLCV,
    indicators computed through it, and the direction of the three candles
    before it. Nothing here reads forward.
    """
    out = pd.DataFrame(index=bars.index)
    close, volume = bars["Close"], bars["Volume"]

    out["openC"] = bars["Open"]
    out["highC"] = bars["High"]
    out["lowC"] = bars["Low"]
    out["closeC"] = close
    out["volumeC"] = volume

    out["ema_fast"] = close.ewm(span=cfg.ema_fast, adjust=False, min_periods=cfg.ema_fast).mean()
    out["ema_slow"] = close.ewm(span=cfg.ema_slow, adjust=False, min_periods=cfg.ema_slow).mean()
    out["ema_trend"] = close.ewm(span=cfg.ema_trend, adjust=False, min_periods=cfg.ema_trend).mean()

    # Efficiency ratio: how much of the distance travelled became net progress.
    net = (close - close.shift(cfg.regime_lookback)).abs()
    path = close.diff().abs().rolling(cfg.regime_lookback,
                                      min_periods=cfg.regime_lookback).sum()
    out["efficiency"] = net / path.replace(0, np.nan)

    # ATR and its trailing percentile rank, both causal.
    prev_close = close.shift(1)
    tr = pd.concat([bars["High"] - bars["Low"],
                    (bars["High"] - prev_close).abs(),
                    (bars["Low"] - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    out["atr_pctile"] = atr.rolling(
        cfg.atr_percentile_window, min_periods=cfg.atr_percentile_window // 5
    ).rank(pct=True)

    out["vol_ma"] = volume.rolling(cfg.volume_ma_period, min_periods=cfg.volume_ma_period).mean()
    out["vol_threshold"] = out["vol_ma"] * cfg.volume_multiplier

    # Pine's closeC2 -- the candle immediately before the setup candle. It
    # widens the setup range when it closed beyond the setup candle's extreme.
    out["prev_close_1"] = close.shift(1)

    bullish = close > bars["Open"]
    bearish = close < bars["Open"]
    for i in (1, 2, 3):
        out[f"bull_{i}"] = bullish.shift(i)
        out[f"bear_{i}"] = bearish.shift(i)
    return out


def detect_setups(context: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Flag setup candles and compute their three levels."""
    out = context.copy()

    volume_ok = out["volumeC"] > out["vol_threshold"]
    # "Don't chase": skip when the three preceding candles already ran one way.
    if cfg.require_pullback:
        def _run_of_three(prefix: str) -> pd.Series:
            cols = [out[f"{prefix}_{i}"].astype("boolean").fillna(False) for i in (1, 2, 3)]
            return (cols[0] & cols[1] & cols[2]).astype(bool)

        not_extended_up = ~_run_of_three("bull")
        not_extended_down = ~_run_of_three("bear")
    else:
        not_extended_up = not_extended_down = pd.Series(True, index=out.index)

    regime_ok = pd.Series(True, index=out.index)
    if cfg.min_efficiency is not None:
        regime_ok &= out["efficiency"] >= cfg.min_efficiency
    if cfg.max_atr_percentile is not None:
        regime_ok &= out["atr_pctile"] <= cfg.max_atr_percentile
    regime_ok = regime_ok.fillna(False)

    out["bull_setup"] = (
        regime_ok
        & volume_ok
        & (out["closeC"] > out["openC"])
        & not_extended_up
        & (out["ema_trend"] <= out["closeC"])
        & ((out["ema_trend"] < out["ema_fast"]) | (out["ema_slow"] < out["ema_fast"]))
    ).fillna(False).astype(bool)

    out["bear_setup"] = (
        regime_ok
        & volume_ok
        & (out["closeC"] < out["openC"])
        & not_extended_down
        & (out["ema_trend"] >= out["closeC"])
        & ((out["ema_trend"] > out["ema_fast"]) | (out["ema_slow"] > out["ema_fast"]))
    ).fillna(False).astype(bool)

    if not cfg.allow_long:
        out["bull_setup"] = False
    if not cfg.allow_short:
        out["bear_setup"] = False

    bull_high = out["highC"]
    bull_low = np.minimum(out["lowC"], out["prev_close_1"])
    bull_trigger = bull_high - (bull_high - bull_low) * cfg.long_retracement

    bear_low = out["lowC"]
    bear_high = np.maximum(out["highC"], out["prev_close_1"])
    bear_trigger = bear_high - (bear_high - bear_low) * cfg.short_retracement

    out["setup_dir"] = np.where(out["bull_setup"], LONG,
                                np.where(out["bear_setup"], SHORT, FLAT))
    out["setup_high"] = np.where(out["bull_setup"], bull_high,
                                 np.where(out["bear_setup"], bear_high, np.nan))
    out["setup_low"] = np.where(out["bull_setup"], bull_low,
                                np.where(out["bear_setup"], bear_low, np.nan))
    out["setup_trigger"] = np.where(out["bull_setup"], bull_trigger,
                                    np.where(out["bear_setup"], bear_trigger, np.nan))

    n_bull, n_bear = int(out["bull_setup"].sum()), int(out["bear_setup"].sum())
    logger.info(
        "Context: %d candles -> %d bullish setups, %d bearish setups (%.1f%% of candles)",
        len(out), n_bull, n_bear, (n_bull + n_bear) / max(len(out), 1) * 100,
    )
    return out



def confirmed_swings(
    high: np.ndarray, low: np.ndarray, left: int = 3, right: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Most recent CONFIRMED swing low and high visible at each bar.

    A pivot low at bar p needs `left` higher lows before it and `right` after.
    Those `right` bars are the catch: the pivot is not knowable until bar
    p + right, so using it any earlier is lookahead. Each pivot is therefore
    published at p + right and forward-filled from there.

    Returns (last_swing_low, last_swing_high), NaN until the first confirmation.
    """
    n = len(low)
    swing_low = np.full(n, np.nan)
    swing_high = np.full(n, np.nan)
    if n < left + right + 1:
        return swing_low, swing_high

    for p in range(left, n - right):
        window_lo = low[p - left : p + right + 1]
        if low[p] == window_lo.min() and (window_lo == low[p]).sum() == 1:
            swing_low[p + right] = low[p]
        window_hi = high[p - left : p + right + 1]
        if high[p] == window_hi.max() and (window_hi == high[p]).sum() == 1:
            swing_high[p + right] = high[p]

    return (pd.Series(swing_low).ffill().to_numpy(),
            pd.Series(swing_high).ffill().to_numpy())


def generate_signals(aligned: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Walk the entry timeframe, arming setups and firing retracement triggers.

    A setup arms on the first entry bar after its context candle closes, and can
    trigger on that very bar -- the original checks the trigger immediately
    after creating the setup, so a violent pullback inside the first minutes is
    a valid fill.
    """
    n = len(aligned)
    low = aligned["Low"].to_numpy(float)
    high = aligned["High"].to_numpy(float)
    ctx_time = aligned["context_open_time"].to_numpy()
    setup_dir = aligned["setup_dir"].to_numpy()
    s_high = aligned["setup_high"].to_numpy(float)
    s_low = aligned["setup_low"].to_numpy(float)
    s_trig = aligned["setup_trigger"].to_numpy(float)
    index = aligned.index

    signal = np.zeros(n, dtype=int)
    active_dir = np.zeros(n, dtype=int)
    active_high = np.full(n, np.nan)
    active_low = np.full(n, np.nan)
    active_trigger = np.full(n, np.nan)
    setup_age = np.full(n, np.nan)
    immediate = np.zeros(n, dtype=bool)

    allowed = set(cfg.allowed_hours) if cfg.allowed_hours else None
    current: Setup | None = None
    armed_bar = -1
    last_ctx = None
    context_bars_seen = 0
    armed_at_count = 0
    expired = 0
    missed_session = 0

    for i in range(n):
        if ctx_time[i] != last_ctx:
            last_ctx = ctx_time[i]
            context_bars_seen += 1

            direction = int(setup_dir[i])
            if direction != FLAT and np.isfinite(s_trig[i]):
                current = Setup(
                    direction=direction,
                    context_time=pd.Timestamp(ctx_time[i]),
                    armed_at=index[i],
                    high=float(s_high[i]),
                    low=float(s_low[i]),
                    trigger=float(s_trig[i]),
                )
                armed_at_count = context_bars_seen
                armed_bar = i
            elif (
                current is not None
                and cfg.setup_expiry_bars > 0
                and context_bars_seen - armed_at_count > cfg.setup_expiry_bars
            ):
                current = None
                expired += 1

        if current is None:
            continue

        active_dir[i] = current.direction
        active_high[i] = current.high
        active_low[i] = current.low
        active_trigger[i] = current.trigger
        setup_age[i] = context_bars_seen - armed_at_count

        touched = (
            low[i] <= current.trigger if current.direction == LONG
            else high[i] >= current.trigger
        )
        if touched:
            if allowed is not None and index[i].hour not in allowed:
                # Outside the traded session the opportunity is missed, not
                # deferred. Holding it would queue setups up and flush them all
                # at the session boundary, far past their entry level.
                current = None
                missed_session += 1
                continue
            signal[i] = current.direction
            # No retracement occurred: price was already through the level
            # when the setup became known.
            immediate[i] = i == armed_bar
            current = None  # consumed; never re-arms

    out = aligned.copy()
    swing_low, swing_high = confirmed_swings(
        aligned["High"].to_numpy(float), aligned["Low"].to_numpy(float),
        cfg.swing_left, cfg.swing_right,
    )
    out["swing_low"] = swing_low
    out["swing_high"] = swing_high
    out["signal"] = signal
    out["active_dir"] = active_dir
    out["active_high"] = active_high
    out["active_low"] = active_low
    out["active_trigger"] = active_trigger
    out["setup_age"] = setup_age
    out["trigger_immediate"] = immediate

    n_buy = int((signal == LONG).sum())
    n_sell = int((signal == SHORT).sum())
    if missed_session:
        logger.info(
            "Discarded %d triggers that fired outside the traded session", missed_session
        )
    n_immediate = int(immediate.sum())
    logger.info(
        "Entry timeframe: %d bars -> %d BUY, %d SELL triggers "
        "(%d fired with no retracement, %d setups expired)",
        n, n_buy, n_sell, n_immediate, expired,
    )
    if n_buy + n_sell == 0:
        logger.warning(
            "No triggers fired. Either no setup formed, or price never retraced to "
            "the entry level. Check strategy.volume_multiplier and the retracement depths."
        )
    return out


def run_strategy(
    entry_bars: pd.DataFrame, context_bars: pd.DataFrame, cfg: StrategyConfig,
    *, context_interval: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full strategy pass. Returns (signals on entry bars, context frame)."""
    from .data import align_context

    context = detect_setups(build_context(context_bars, cfg), cfg)
    aligned = align_context(entry_bars, context[CONTEXT_COLUMNS], context_interval)
    return generate_signals(aligned, cfg), context
