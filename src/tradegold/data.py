"""Dual-timeframe market data with strict as-of alignment.

The strategy defines setups on a higher timeframe (60m) and triggers them on a
lower one (5m). Pine expresses this as `request.security(..., close[1],
lookahead_on)`: the `[1]` means each chart bar sees the *previous completed*
context candle, and it is that offset -- not the lookahead flag -- that keeps
the script honest.

Reproducing it here is the single place this project can silently acquire
lookahead bias, so alignment is done once, explicitly, and asserted in tests:
every context bar is stamped with its **close** time and merged backward onto
the entry bars, so a 5m bar at 10:05 sees the 60m candle that closed at 10:00
and nothing newer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

from .config import INTERVAL_MINUTES, DataConfig
from .logging_utils import get_logger
from .registry import Registry

logger = get_logger(__name__)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


class DataProvider(Protocol):
    def __call__(self, ticker: str, interval: str, period: str) -> pd.DataFrame: ...


PROVIDERS: Registry[DataProvider] = Registry("data provider")


@PROVIDERS.register("yfinance")
def _yfinance_provider(ticker: str, interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    logger.info("Downloading %s | interval=%s period=%s", ticker, interval, period)
    df = yf.download(
        tickers=ticker, interval=interval, period=period,
        auto_adjust=False, progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError(
            f"No rows returned for {ticker} at {interval}/{period}. "
            "Note yfinance caps intraday history: 7d for 1m, 60d for 2m-90m, 730d for 60m."
        )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@PROVIDERS.register("csv")
def _csv_provider(ticker: str, interval: str, period: str) -> pd.DataFrame:
    """Read a local CSV (offline / CI). `ticker` is treated as a path template
    with `{interval}` substituted, e.g. `data/gold_{interval}.csv`."""
    path = Path(ticker.format(interval=interval))
    if not path.exists():
        raise FileNotFoundError(f"csv provider: {path} not found")
    logger.info("Reading %s", path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _normalise(df: pd.DataFrame, roll_column: str | None = None) -> pd.DataFrame:
    """Coerce any provider's frame into a UTC-indexed OHLCV table.

    Column names are matched case-insensitively so a vendor CSV using
    `open,high,low,close,volume` needs no preprocessing; extra columns such as
    `symbol` are dropped rather than tripping the check.
    """
    lookup = {str(c).strip().lower(): c for c in df.columns}
    missing = [c for c in OHLCV if c.lower() not in lookup]
    if missing:
        raise ValueError(
            f"Provider data is missing required columns: {missing}. "
            f"Found: {sorted(df.columns)}"
        )
    out = df.loc[:, [lookup[c.lower()] for c in OHLCV]].copy()
    out.columns = OHLCV
    out.index = pd.to_datetime(out.index, utc=True)
    out.index.name = "timestamp"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out["Volume"] = out["Volume"].fillna(0.0).astype(float)

    if roll_column:
        match = next((c for c in df.columns if str(c).lower() == roll_column.lower()), None)
        if match is None:
            raise ValueError(
                f"data.roll_column '{roll_column}' not found. Available: {sorted(df.columns)}"
            )
        flag = df[match].reindex(out.index)
        out["is_roll"] = (
            flag.astype(str).str.strip().str.lower().isin(["true", "1", "yes"]).fillna(False)
        )
    return out


def _fetch(cfg: DataConfig, interval: str, period: str) -> pd.DataFrame:
    cache = cfg.cache_dir / f"{cfg.ticker.replace('=', '_')}_{interval}.parquet"
    if cfg.use_cache and not cfg.refresh and cache.exists():
        logger.info("Loading cached %s bars from %s", interval, cache)
        return _normalise(pd.read_parquet(cache), cfg.roll_column)

    df = _normalise(PROVIDERS.get(cfg.provider)(cfg.ticker, interval, period), cfg.roll_column)
    if cfg.use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
        logger.info("Cached %d %s bars to %s", len(df), interval, cache)
    return df


def resample_bars(bars: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Aggregate finer bars into a coarser timeframe, indexed by OPEN time.

    `label="left", closed="left"` keeps the convention used everywhere else here:
    a 60m bar stamped 10:00 covers 10:00-10:59 and is only complete at 11:00.
    Empty buckets (weekends, holidays) are dropped rather than carried forward.
    """
    rule = f"{INTERVAL_MINUTES[interval]}min"
    agg_roll = {"is_roll": ("is_roll", "max")} if "is_roll" in bars.columns else {}
    out = bars.resample(rule, label="left", closed="left").agg(
        **agg_roll,
        Open=("Open", "first"), High=("High", "max"),
        Low=("Low", "min"), Close=("Close", "last"), Volume=("Volume", "sum"),
    )
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def load_timeframes(cfg: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (entry_bars, context_bars), both UTC-indexed by bar OPEN time."""
    entry = _fetch(cfg, cfg.entry_interval, cfg.entry_period)
    if cfg.context_source == "resample":
        context = resample_bars(entry, cfg.context_interval)
        logger.info(
            "Derived %d %s context candles by resampling the %s entry bars",
            len(context), cfg.context_interval, cfg.entry_interval,
        )
    else:
        context = _fetch(cfg, cfg.context_interval, cfg.context_period)
    logger.info(
        "Entry %s: %d bars %s -> %s | Context %s: %d bars %s -> %s",
        cfg.entry_interval, len(entry), entry.index[0].date(), entry.index[-1].date(),
        cfg.context_interval, len(context), context.index[0].date(), context.index[-1].date(),
    )
    if entry.empty or context.empty:
        raise RuntimeError("One of the timeframes came back empty")
    return entry, context


def align_context(
    entry: pd.DataFrame, context_view: pd.DataFrame, context_interval: str
) -> pd.DataFrame:
    """Attach each entry bar to the last context bar that had FULLY CLOSED.

    `context_view` is indexed by context bar open time. A bar opening at 09:00
    on a 60m timeframe is only knowable from 10:00 onward, so it is stamped with
    its close time before the backward merge.
    """
    step = pd.Timedelta(int(INTERVAL_MINUTES[context_interval]), unit="m")
    stamped = context_view.copy()
    stamped["context_open_time"] = stamped.index
    stamped.index = stamped.index + step
    stamped.index.name = "available_from"

    # Name the join keys here rather than relying on the caller's index name.
    left = pd.DataFrame({"entry_time": entry.index})
    merged = pd.merge_asof(
        left=left,
        right=stamped.reset_index(),
        left_on="entry_time",
        right_on="available_from",
        direction="backward",
    ).set_index("entry_time")
    merged.index = entry.index

    aligned = entry.join(merged.drop(columns=["available_from"]))

    # A context bar must never be visible before it closes.
    late = aligned["context_open_time"].notna()
    if late.any():
        latest_visible = aligned.loc[late, "context_open_time"] + step
        if (latest_visible > aligned.index[late]).any():
            raise AssertionError("Context alignment leaked a bar before its close time")

    n_warm = int(aligned["context_open_time"].isna().sum())
    if n_warm:
        logger.info("Dropping %d entry bars before the first closed context bar", n_warm)
    return aligned[aligned["context_open_time"].notna()].copy()
