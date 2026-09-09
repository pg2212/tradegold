"""Event-driven backtester for setup-triggered entries, long and short.

The source indicator emits entries and plots the setup's extremes, but names no
exit. `execution.stop`/`target` default to `setup_extreme`, the reading those
plots imply: stop at the far side of the setup candle, target at the near side.
That is roughly 0.67R, so it needs a hit rate above ~60% to break even before
costs -- a fact the metrics report makes visible rather than hiding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ExecutionConfig
from .logging_utils import get_logger
from .strategy import LONG, SHORT

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame

    @property
    def n_trades(self) -> int:
        return len(self.trades)


TRADE_COLUMNS = [
    "direction", "side", "signal_time", "entry_time", "exit_time",
    "entry_idx", "exit_idx", "entry_price", "exit_price",
    "stop_price", "initial_stop", "trailing_stop", "target_price",
    "setup_time", "trigger_level",
    "bars_held", "gross_pnl", "costs", "net_pnl", "r_multiple", "exit_reason",
]


def _true_range(bars: pd.DataFrame, period: int) -> np.ndarray:
    prev_close = bars["Close"].shift(1)
    tr = pd.concat(
        [bars["High"] - bars["Low"],
         (bars["High"] - prev_close).abs(),
         (bars["Low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().to_numpy(float)


def _levels(
    direction: int, entry: float, setup_high: float, setup_low: float,
    atr: float, cfg: ExecutionConfig, *, ref_high: float, ref_low: float,
    swing_low: float = np.nan, swing_high: float = np.nan,
) -> tuple[float, float]:
    """Stop and target for one bar.

    `ref_high`/`ref_low` are the last completed context candle at that bar. For
    the fixed modes they equal the setup candle and never change; for
    `prev_candle` they advance, which is what makes the exit trail.
    """
    if cfg.stop == "setup_extreme":
        stop = setup_low if direction == LONG else setup_high
    elif cfg.stop == "prev_candle":
        stop = ref_low if direction == LONG else ref_high
    elif cfg.stop == "swing":
        level = swing_low if direction == LONG else swing_high
        if not np.isfinite(level):
            # No pivot confirmed yet; fall back to the setup range.
            level = setup_low if direction == LONG else setup_high
        pad = abs(setup_high - setup_low) * cfg.swing_buffer
        stop = level - pad if direction == LONG else level + pad
    else:
        offset = atr * cfg.atr_stop_multiplier
        stop = entry - offset if direction == LONG else entry + offset

    if cfg.target == "setup_extreme":
        target = setup_high if direction == LONG else setup_low
    elif cfg.target == "prev_candle":
        target = ref_high if direction == LONG else ref_low
    else:
        risk = abs(entry - stop)
        target = entry + risk * cfg.r_multiple * (1 if direction == LONG else -1)
    return float(stop), float(target)


def _is_dynamic(cfg: ExecutionConfig) -> bool:
    return "prev_candle" in (cfg.stop, cfg.target)


def run_backtest(signals: pd.DataFrame, cfg: ExecutionConfig) -> BacktestResult:
    """Simulate one position at a time from the strategy's trigger signals."""
    open_ = signals["Open"].to_numpy(float)
    high = signals["High"].to_numpy(float)
    low = signals["Low"].to_numpy(float)
    close = signals["Close"].to_numpy(float)
    signal = signals["signal"].to_numpy(int)
    immediate = signals["trigger_immediate"].to_numpy(bool)
    s_high = signals["active_high"].to_numpy(float)
    s_low = signals["active_low"].to_numpy(float)
    trigger = signals["active_trigger"].to_numpy(float)
    # The last completed context candle at each entry bar -- what `prev_candle`
    # exits trail. Already lag-corrected by the context alignment.
    ctx_high = signals["highC"].to_numpy(float)
    ctx_low = signals["lowC"].to_numpy(float)
    sw_low = signals["swing_low"].to_numpy(float) if "swing_low" in signals else None
    sw_high = signals["swing_high"].to_numpy(float) if "swing_high" in signals else None
    is_roll = (signals["is_roll"].to_numpy(bool) if "is_roll" in signals
               else np.zeros(len(signals), dtype=bool))
    ctx_time = signals["context_open_time"].to_numpy()
    index = signals.index
    n = len(signals)

    atr = _true_range(signals, cfg.atr_period) if cfg.stop == "atr" else np.zeros(n)
    cost = cfg.cost_per_side + cfg.slippage_per_side

    rows: list[dict] = []
    busy_until = -1
    skipped_immediate = 0
    degenerate = 0
    poor_geometry = 0
    rolled = 0

    for i in range(n):
        direction = int(signal[i])
        if direction == 0 or i <= busy_until:
            continue
        if is_roll[i] or (i + 1 < n and is_roll[i + 1]):
            rolled += 1
            continue
        if cfg.skip_immediate_triggers and immediate[i]:
            skipped_immediate += 1
            continue

        level = trigger[i]
        if cfg.entry_fill == "next_open":
            # The alert fires on this bar's close; the fill is the next open.
            if i + 1 >= n:
                continue
            entry_i, entry_price = i + 1, open_[i + 1]
        elif cfg.entry_fill == "close":
            entry_i, entry_price = i, close[i]
        else:
            # A resting limit order. Filling better than the level is only
            # possible when the bar opened through it.
            entry_i = i
            entry_price = min(open_[i], level) if direction == LONG else max(open_[i], level)

        stop, target = _levels(
            direction, entry_price, s_high[i], s_low[i], atr[i], cfg,
            ref_high=ctx_high[i], ref_low=ctx_low[i],
            swing_low=sw_low[i] if sw_low is not None else np.nan,
            swing_high=sw_high[i] if sw_high is not None else np.nan,
        )
        # A trailing stop moves, so the ledger's `stop_price` is where it ended.
        # Keep where it began, or a chart cannot show what actually happened.
        initial_stop = stop

        # A stop the wrong side of entry means the setup range degenerated.
        if (direction == LONG and stop >= entry_price) or (
            direction == SHORT and stop <= entry_price
        ):
            degenerate += 1
            continue
        # Price ran so far past the level that the fill sits on top of the stop.
        setup_range = abs(s_high[i] - s_low[i])
        if setup_range > 0 and abs(entry_price - stop) < setup_range * cfg.min_risk_fraction:
            degenerate += 1
            continue
        # Geometry filter: skip trades that cannot pay for their own risk.
        if cfg.min_reward_risk > 0:
            risk_now = abs(entry_price - stop)
            reward_now = abs(target - entry_price)
            if risk_now <= 0 or reward_now / risk_now < cfg.min_reward_risk:
                poor_geometry += 1
                continue

        last = (n - 1 if cfg.max_holding_bars == 0
                else min(entry_i + cfg.max_holding_bars - 1, n - 1))
        exit_i, exit_price, reason = None, None, None
        dynamic = _is_dynamic(cfg)
        for j in range(entry_i, last + 1):
            if is_roll[j] and j > entry_i:
                # The contract rolls on this bar. An unadjusted series prices
                # the new contract differently, so close on the prior bar rather
                # than book a gap that no position actually experienced.
                exit_i, exit_price, reason = j - 1, close[j - 1], "contract_roll"
                break
            if dynamic and j > entry_i:
                # Re-read the levels off whichever context candle has closed by
                # now. A trailing stop only ever moves in the trade's favour.
                new_stop, new_target = _levels(
                    direction, entry_price, s_high[i], s_low[i], atr[i], cfg,
                    ref_high=ctx_high[j], ref_low=ctx_low[j],
                )
                if cfg.stop == "prev_candle":
                    stop = max(stop, new_stop) if direction == LONG else min(stop, new_stop)
                if cfg.target == "prev_candle":
                    target = new_target
            if direction == LONG:
                hit_target, hit_stop = high[j] >= target, low[j] <= stop
            else:
                hit_target, hit_stop = low[j] <= target, high[j] >= stop
            if hit_stop and hit_target:
                # Intrabar order is unknowable from OHLC; take the loss.
                exit_i, exit_price, reason = j, stop, "stop_loss"
                break
            if hit_target:
                exit_i, exit_price, reason = j, target, "target"
                break
            if hit_stop:
                exit_i, exit_price, reason = j, stop, "stop_loss"
                break
        if exit_i is None:
            exit_i = last
            exit_price = close[last]
            reason = "time_exit" if cfg.max_holding_bars and last < n - 1 else "end_of_data"

        gross = (exit_price - entry_price) * (1 if direction == LONG else -1)
        risk = abs(entry_price - stop)
        net = gross - 2.0 * cost
        rows.append({
            "direction": direction,
            "side": "long" if direction == LONG else "short",
            "signal_time": index[i],
            "entry_time": index[entry_i],
            "exit_time": index[exit_i],
            "entry_idx": entry_i,
            "exit_idx": exit_i,
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "stop_price": stop,
            "initial_stop": initial_stop,
            "trailing_stop": cfg.stop == "prev_candle",
            "target_price": target,
            "setup_time": pd.Timestamp(ctx_time[i]),
            "trigger_level": float(level),
            "bars_held": int(exit_i - entry_i + 1),
            "gross_pnl": float(gross),
            "costs": 2.0 * cost,
            "net_pnl": float(net),
            "r_multiple": float(net / risk) if risk > 0 else np.nan,
            "exit_reason": reason,
        })
        busy_until = exit_i

    trades = pd.DataFrame(rows, columns=TRADE_COLUMNS)
    equity = _mark_to_market(signals, trades, cost)

    if degenerate:
        logger.info(
            "Skipped %d triggers whose fill landed too close to the stop "
            "(execution.min_risk_fraction)", degenerate,
        )
    if rolled:
        logger.info("Skipped %d triggers landing on a contract roll", rolled)
    if poor_geometry:
        logger.info(
            "Skipped %d triggers whose reward:risk was below %.2f "
            "(execution.min_reward_risk)", poor_geometry, cfg.min_reward_risk,
        )
    if skipped_immediate:
        logger.info(
            "Skipped %d triggers that fired with no retracement "
            "(execution.skip_immediate_triggers)", skipped_immediate,
        )
    if len(trades):
        longs = int((trades["direction"] == LONG).sum())
        logger.info(
            "Backtest: %d trades (%d long, %d short) over %d bars, %.1f%% time in market",
            len(trades), longs, len(trades) - longs, n,
            float(equity["in_position"].mean()) * 100,
        )
    else:
        logger.warning("Backtest produced no trades")
    return BacktestResult(trades=trades, equity=equity)


def _mark_to_market(bars: pd.DataFrame, trades: pd.DataFrame, cost: float) -> pd.DataFrame:
    """Per-bar equity, so drawdown reflects open risk rather than closed trades."""
    close = bars["Close"].to_numpy(float)
    n = len(bars)
    bar_pnl = np.zeros(n)
    exposure = np.zeros(n, dtype=bool)

    for t in trades.itertuples():
        sign = 1.0 if t.direction == LONG else -1.0
        for j in range(t.entry_idx, t.exit_idx + 1):
            prev = t.entry_price if j == t.entry_idx else close[j - 1]
            cur = t.exit_price if j == t.exit_idx else close[j]
            bar_pnl[j] += (cur - prev) * sign
            exposure[j] = True
        bar_pnl[t.entry_idx] -= cost
        bar_pnl[t.exit_idx] -= cost

    return pd.DataFrame(
        {
            "close": close,
            "bar_pnl": bar_pnl,
            "equity": np.cumsum(bar_pnl),
            "buy_hold": close - close[0],
            "in_position": exposure,
        },
        index=bars.index,
    )
