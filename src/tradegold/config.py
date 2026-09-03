"""Typed, validated configuration for the GOLD STRATEGY engine.

Every parameter of the strategy lives here. Configs load from YAML, can be
layered, and are hashed so a run directory traces back to exact settings.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Bars per year by interval, used to annualise Sharpe/Sortino.
# Gold futures trade ~23h/day, ~252 sessions/year.
BARS_PER_YEAR: dict[str, float] = {
    "1m": 252 * 23 * 60,
    "5m": 252 * 23 * 12,
    "15m": 252 * 23 * 4,
    "30m": 252 * 23 * 2,
    "60m": 252 * 23,
    "1h": 252 * 23,
    "1d": 252,
}

# Pandas offset per interval, for resampling and gap checks.
INTERVAL_MINUTES: dict[str, int] = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "1h": 60, "1d": 1440,
}


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RunConfig(_Base):
    name: str = "gold_strategy"
    output_dir: Path = Path("runs")
    log_level: str = "INFO"


class DataConfig(_Base):
    provider: str = "yfinance"
    ticker: str = "GC=F"
    # Entry timeframe: where retracement triggers are detected.
    entry_interval: str = "5m"
    entry_period: str = "60d"
    # Context timeframe: where setups are defined. Pine's `timeframeC`.
    context_interval: str = "60m"
    context_period: str = "730d"
    # fetch    -- download the context timeframe separately (two provider calls)
    # resample -- aggregate it from the entry bars. Correct when both timeframes
    #             must come from one file, and it removes any chance of the two
    #             series disagreeing about a bar's OHLC.
    context_source: Literal["fetch", "resample"] = "fetch"
    # Column marking contract rolls in a continuous futures series. An
    # unadjusted series shows the roll as a price gap; holding through one books
    # a move that never happened, so positions are closed before it.
    roll_column: str | None = None
    cache_dir: Path = Path("data/raw")
    use_cache: bool = True
    refresh: bool = False

    @model_validator(mode="after")
    def _check_intervals(self) -> DataConfig:
        for field in ("entry_interval", "context_interval"):
            value = getattr(self, field)
            if value not in INTERVAL_MINUTES:
                raise ValueError(
                    f"data.{field}='{value}' is not supported. "
                    f"Choose from: {', '.join(INTERVAL_MINUTES)}"
                )
        if INTERVAL_MINUTES[self.entry_interval] >= INTERVAL_MINUTES[self.context_interval]:
            raise ValueError(
                f"entry_interval ({self.entry_interval}) must be finer than "
                f"context_interval ({self.context_interval}); the strategy defines setups "
                "on the higher timeframe and triggers them on the lower one"
            )
        return self


class StrategyConfig(_Base):
    """Parameters of the GOLD STRATEGY setup rules."""

    ema_fast: int = Field(10, ge=1)     # Pine: emaCLen1
    ema_slow: int = Field(20, ge=1)     # Pine: emaCLen2
    ema_trend: int = Field(50, ge=1)    # Pine: emaCLen3

    volume_ma_period: int = Field(30, ge=2)
    # Pine computes volMA45 = SMA(volume,30) * 1.5 and tests `volumeC > volMA45`.
    # (The source comment says volMA90; the code uses volMA45. Code wins.)
    volume_multiplier: float = Field(1.5, gt=0)

    # Retracement into the setup candle's range, as a fraction measured down
    # from its high. Long fills at high - 0.40*range; short fills at
    # high - 0.60*range (= low + 0.40*range). The asymmetry is in the original.
    long_retracement: float = Field(0.40, gt=0, lt=1)
    short_retracement: float = Field(0.60, gt=0, lt=1)

    # Pine: not (three preceding context candles all closed the same direction).
    require_pullback: bool = True
    # Context bars a pending setup stays armed. 0 reproduces the original, where
    # a setup persists until it triggers or the opposite setup replaces it.
    setup_expiry_bars: int = Field(0, ge=0)

    # Pivot detection for structural stops. `swing_right` bars must pass before
    # a pivot is confirmed, so larger values mean later but firmer levels.
    swing_left: int = Field(3, ge=1)
    swing_right: int = Field(3, ge=1)

    # Entry hours (UTC). None trades around the clock. A session filter is a
    # strong claim -- it must be justified out of sample, not by slicing history.
    allowed_hours: list[int] | None = None

    # Regime gate on the context timeframe, both causal rolling statistics.
    # efficiency = |net move| / total path over `regime_lookback` candles.
    # High means trending, low means chop. A 2R target needs follow-through, so
    # the hypothesis is that chop is unprofitable regardless of the calendar.
    regime_lookback: int = Field(24, ge=4)
    min_efficiency: float | None = Field(None, ge=0, le=1)
    # Skip when ATR sits above this percentile of its own trailing distribution.
    max_atr_percentile: float | None = Field(None, gt=0, le=1)
    atr_percentile_window: int = Field(500, ge=50)

    allow_long: bool = True
    allow_short: bool = True

    @model_validator(mode="after")
    def _check_hours(self) -> StrategyConfig:
        if self.allowed_hours is not None:
            bad = [h for h in self.allowed_hours if not 0 <= h <= 23]
            if bad:
                raise ValueError(f"allowed_hours must be 0-23, got {bad}")
            if not self.allowed_hours:
                raise ValueError("allowed_hours cannot be an empty list; use null")
        return self

    @model_validator(mode="after")
    def _check_sides(self) -> StrategyConfig:
        if not (self.allow_long or self.allow_short):
            raise ValueError("strategy.allow_long and allow_short cannot both be false")
        if self.ema_fast >= self.ema_trend:
            raise ValueError(
                f"ema_fast ({self.ema_fast}) must be shorter than ema_trend ({self.ema_trend})"
            )
        return self


class ExecutionConfig(_Base):
    """How a triggered setup becomes a closed trade.

    The source is a TradingView *indicator*: it emits entries and plots the
    setup high/low, but specifies no exit. `setup_extreme` is the reading its
    own plots imply -- stop at the far side of the setup candle, target at the
    near side. Note this gives roughly 0.67R, so it needs a >60% hit rate.
    """

    # How a trigger becomes a fill.
    #   next_open  -- signal at bar close, fill at the next bar's open. Matches
    #                 what a trader acting on the alert actually gets, and is
    #                 the only option with no free edge baked in.
    #   level      -- assume a resting limit order fills exactly at the level.
    #   close      -- fill at the close of the trigger bar.
    entry_fill: Literal["next_open", "level", "close"] = "next_open"
    # A setup that arms with price ALREADY beyond its trigger never offered a
    # retracement to wait for; the "entry" is just wherever price happens to be.
    # Skipping these removes the single largest source of illusory profit.
    skip_immediate_triggers: bool = True

    # A next-open fill can land a hair from the stop, producing a trade whose
    # risk is smaller than the spread and an R multiple in the tens. Reject any
    # fill whose distance to the stop is below this fraction of the setup range.
    min_risk_fraction: float = Field(0.10, ge=0, lt=1)
    # Padding beyond the swing level, as a fraction of the setup range, so a
    # stop does not sit exactly on an obvious pivot.
    swing_buffer: float = Field(0.05, ge=0)
    # Refuse trades whose geometry is not worth taking. 0 disables the filter.
    min_reward_risk: float = Field(0.0, ge=0)

    # setup_extreme -- fixed at the setup candle's far/near side
    # prev_candle   -- tracks the LAST COMPLETED context candle, so both levels
    #                  move forward as new candles close. On the entry bar the
    #                  last completed candle IS the setup candle, so this starts
    #                  identical to setup_extreme and then trails.
    # atr / r_multiple -- volatility- or risk-scaled, fixed at entry.
    # swing -- the last confirmed swing low (long) / high (short) on the entry
    #          timeframe. Structural rather than tied to the setup candle's range,
    #          so the reward:risk is no longer pinned at 0.67 by construction.
    stop: Literal["setup_extreme", "prev_candle", "swing", "atr"] = "setup_extreme"
    target: Literal["setup_extreme", "prev_candle", "r_multiple"] = "setup_extreme"
    r_multiple: float = Field(1.5, gt=0)
    atr_period: int = Field(14, ge=2)
    atr_stop_multiplier: float = Field(1.5, gt=0)
    # Entry bars before an open trade is closed at market. 0 = no time limit.
    max_holding_bars: int = Field(0, ge=0)
    cost_per_side: float = Field(0.25, ge=0)
    slippage_per_side: float = Field(0.0, ge=0)
    annualization_bars: float | None = None


class CapitalConfig(_Base):
    """Turns per-unit price results into an account.

    Without this every trade is one unit regardless of its stop distance, so a
    trade risking $4 and one risking $78 count equally. Sizing by risk is both
    how people actually trade and what stops wide-stop trades dominating.
    """

    enabled: bool = False
    initial: float = Field(100_000.0, gt=0)
    # Fraction of equity risked per trade, in percent.
    risk_per_trade_pct: float = Field(1.0, gt=0, le=100)
    # Units of the underlying per contract: 100 for GC, 10 for micro gold,
    # 1 for spot traded in ounces.
    contract_size: float = Field(100.0, gt=0)
    # Smallest tradeable increment. 1 means whole contracts only.
    lot_step: float = Field(1.0, gt=0)
    # Reinvest gains. False sizes every trade off the starting balance.
    compound: bool = True
    # Refuse a trade whose notional would exceed this multiple of equity.
    max_leverage: float = Field(20.0, gt=0)


class ReportConfig(_Base):
    plot: bool = True
    # Chart window written to the run directory, in entry bars.
    plot_bars: int = Field(600, ge=50)


class Config(_Base):
    run: RunConfig = Field(default_factory=RunConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    capital: CapitalConfig = Field(default_factory=CapitalConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    @model_validator(mode="after")
    def _resolve_defaults(self) -> Config:
        if self.execution.annualization_bars is None:
            self.execution.annualization_bars = BARS_PER_YEAR[self.data.entry_interval]
        return self

    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce_scalar(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply `--set a.b.c=value` style overrides onto a raw config dict."""
    out = dict(data)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be in key.path=value form")
        path, _, raw = item.partition("=")
        keys = path.strip().split(".")
        cursor = out
        for key in keys[:-1]:
            nxt = cursor.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[key] = nxt
            cursor = nxt
        cursor[keys[-1]] = _coerce_scalar(raw.strip())
    return out


def load_config(
    path: str | Path,
    *,
    base: str | Path | None = None,
    overrides: list[str] | None = None,
) -> Config:
    data: dict[str, Any] = {}
    if base is not None:
        data = yaml.safe_load(Path(base).read_text()) or {}
    variant = yaml.safe_load(Path(path).read_text()) or {}
    data = _deep_merge(data, variant)
    if overrides:
        data = apply_overrides(data, overrides)
    return Config.model_validate(data)
