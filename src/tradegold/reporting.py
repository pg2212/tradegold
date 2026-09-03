"""Static chart for the run directory. Headless, so runs never block on a GUI."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .backtest import BacktestResult  # noqa: E402
from .logging_utils import get_logger  # noqa: E402
from .strategy import LONG  # noqa: E402

logger = get_logger(__name__)

UP, DOWN = "#0ca30c", "#d03b3b"
# EMAs and setup levels must never share a hue -- they overlay the same panel.
EMA_STYLE = {
    "ema_fast": ("#2a78d6", "EMA fast (context)"),
    "ema_slow": ("#eb6834", "EMA slow (context)"),
    "ema_trend": ("#4a3aa7", "EMA trend (context)"),
}
LEVEL_HIGH, LEVEL_ENTRY, LEVEL_STOP = "#e87ba4", "#eda100", "#d03b3b"


def plot_strategy(
    signals: pd.DataFrame, result: BacktestResult, path: Path, title: str, bars: int = 600
) -> Path:
    """Candles + context EMAs + setup levels + triggers, over the last `bars`."""
    # Centre the window on the last trade so the chart shows the strategy
    # working, not an arbitrary quiet stretch at the end of the sample.
    trades_all = result.trades
    if not trades_all.empty:
        anchor = signals.index.searchsorted(trades_all["exit_time"].iloc[-1])
        lo = max(0, anchor - int(bars * 0.75))
        view = signals.iloc[lo: lo + bars]
    else:
        view = signals.iloc[-bars:]
    fig, (ax, ax_eq) = plt.subplots(
        2, 1, figsize=(15, 9), sharex=True, height_ratios=[3, 1], constrained_layout=True
    )

    x = mdates.date2num(view.index.to_pydatetime())
    width = (x[1] - x[0]) * 0.7 if len(x) > 1 else 0.001
    up = view["Close"] >= view["Open"]
    for mask, color in ((up, UP), (~up, DOWN)):
        sub, xs = view[mask], x[mask.to_numpy()]
        if sub.empty:
            continue
        ax.vlines(xs, sub["Low"], sub["High"], color=color, linewidth=0.6)
        ax.bar(xs, (sub["Close"] - sub["Open"]).abs().clip(lower=1e-6),
               bottom=np.minimum(sub["Open"], sub["Close"]),
               width=width, color=color, edgecolor=color, linewidth=0.4)

    for col, (color, label) in EMA_STYLE.items():
        if col in view.columns:
            ax.step(x, view[col], where="post", color=color, linewidth=1.3,
                    alpha=0.9, label=label)

    # Setup levels, drawn only while a setup is armed (matches the source plots).
    for col, color, style, label in (
        ("active_high", LEVEL_HIGH, "-", "Setup high"),
        ("active_trigger", LEVEL_ENTRY, "-", "Entry (retracement)"),
        ("active_low", LEVEL_STOP, ":", "Stop side"),
    ):
        series = view[col].where(view["active_dir"] != 0)
        ax.plot(x, series, color=color, linewidth=1.6, linestyle=style, label=label, alpha=0.95)

    trades = result.trades
    lo, hi = view.index[0], view.index[-1]
    in_view = trades[(trades["entry_time"] >= lo) & (trades["entry_time"] <= hi)]
    for t in in_view.itertuples():
        marker = "^" if t.direction == LONG else "v"
        color = UP if t.net_pnl > 0 else DOWN
        ax.plot(mdates.date2num(t.entry_time), t.entry_price, marker=marker,
                color=color, markersize=10, markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        ax.plot(mdates.date2num(t.exit_time), t.exit_price, marker="x",
                color=color, markersize=7, zorder=5)

    ax.set_title(f"{title}\nlast {len(view):,} bars · {len(in_view)} trades shown", fontsize=12)
    ax.set_ylabel("Price ($/oz)")
    ax.legend(loc="upper left", fontsize=9, ncol=3, framealpha=0.9)
    ax.grid(alpha=0.2)

    eq = result.equity.loc[view.index[0]:view.index[-1]]
    ax_eq.plot(mdates.date2num(eq.index.to_pydatetime()), eq["equity"],
               color="#eb6834", linewidth=1.6)
    ax_eq.axhline(0, color="#888", linewidth=0.8)
    ax_eq.set_ylabel("Net PnL ($/oz)")
    ax_eq.grid(alpha=0.2)
    ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info("Chart written to %s", path)
    return path
