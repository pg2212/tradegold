"""Performance metrics computed from the trade ledger."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .backtest import BacktestResult
from .strategy import LONG


def _max_drawdown(equity: np.ndarray) -> tuple[float, int]:
    if equity.size == 0:
        return 0.0, 0
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(dd.min()), int(np.argmin(dd))


def _side_stats(trades: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, subset in (("long", trades[trades["direction"] == LONG]),
                         ("short", trades[trades["direction"] != LONG])):
        if subset.empty:
            out[name] = {"n": 0}
            continue
        pnl = subset["net_pnl"].to_numpy(float)
        out[name] = {
            "n": len(subset),
            "win_rate": float((pnl > 0).mean()),
            "net_pnl": float(pnl.sum()),
            "expectancy": float(pnl.mean()),
        }
    return out


def trade_metrics(trades: pd.DataFrame) -> dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {"n_trades": 0, "win_rate": None, "profit_factor": None, "expectancy": None}

    pnl = trades["net_pnl"].to_numpy(float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())
    risk = (trades["entry_price"] - trades["stop_price"]).abs()
    reward = (trades["target_price"] - trades["entry_price"]).abs()

    return {
        "n_trades": n,
        "win_rate": float(len(wins) / n),
        "avg_win": float(wins.mean()) if wins.size else 0.0,
        "avg_loss": float(losses.mean()) if losses.size else 0.0,
        "largest_win": float(pnl.max()),
        "largest_loss": float(pnl.min()),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else None,
        "expectancy": float(pnl.mean()),
        "avg_r_multiple": float(trades["r_multiple"].mean()),
        # The planned reward:risk the setup geometry hands you before any skill.
        # Reported as a median: a single fill landing beside its stop produces an
        # R in the tens and makes the mean meaningless.
        "median_planned_rr": float((reward / risk.replace(0, np.nan)).median()),
        "avg_planned_rr": float((reward / risk.replace(0, np.nan)).mean()),
        "breakeven_win_rate": float(
            1.0 / (1.0 + (reward / risk.replace(0, np.nan)).median())
        ),
        "total_net_pnl": float(pnl.sum()),
        "total_costs": float(trades["costs"].sum()),
        "avg_bars_held": float(trades["bars_held"].mean()),
        "exit_reasons": trades["exit_reason"].value_counts().to_dict(),
        "by_side": _side_stats(trades),
    }


def equity_metrics(equity: pd.DataFrame, annualization_bars: float) -> dict[str, Any]:
    bar_pnl = equity["bar_pnl"].to_numpy(float)
    curve = equity["equity"].to_numpy(float)
    mdd, trough = _max_drawdown(curve)

    std = float(bar_pnl.std(ddof=1)) if bar_pnl.size > 1 else 0.0
    downside = bar_pnl[bar_pnl < 0]
    dstd = float(downside.std(ddof=1)) if downside.size > 1 else 0.0
    scale = float(np.sqrt(annualization_bars))
    mean = float(bar_pnl.mean()) if bar_pnl.size else 0.0
    net = float(curve[-1]) if curve.size else 0.0

    return {
        "net_pnl": net,
        "buy_hold_pnl": float(equity["buy_hold"].iloc[-1]) if len(equity) else 0.0,
        "max_drawdown": mdd,
        "max_drawdown_time": str(equity.index[trough]) if len(equity) else None,
        "return_over_max_drawdown": float(net / abs(mdd)) if mdd < 0 else None,
        "sharpe": float(mean / std * scale) if std > 0 else None,
        "sortino": float(mean / dstd * scale) if dstd > 0 else None,
        "exposure": float(equity["in_position"].mean()),
        "n_bars": int(len(equity)),
        "start": str(equity.index[0]) if len(equity) else None,
        "end": str(equity.index[-1]) if len(equity) else None,
    }


def setup_metrics(signals: pd.DataFrame) -> dict[str, Any]:
    """How many setups formed, and how many ever got their retracement."""
    armed = signals["active_dir"].to_numpy()
    changes = int((pd.Series(armed).ne(pd.Series(armed).shift()) & (armed != 0)).sum())
    triggered = int((signals["signal"] != 0).sum())
    return {
        "setups_armed": changes,
        "setups_triggered": triggered,
        "fill_rate": float(triggered / changes) if changes else None,
        "bars_armed_pct": float((armed != 0).mean()),
    }


def summarise(
    result: BacktestResult, signals: pd.DataFrame, annualization_bars: float
) -> dict[str, Any]:
    return {
        "setups": setup_metrics(signals),
        "trades": trade_metrics(result.trades),
        "equity": equity_metrics(result.equity, annualization_bars),
    }


def format_report(metrics: dict[str, Any], title: str) -> str:
    s, t, e = metrics["setups"], metrics["trades"], metrics["equity"]

    def num(value: Any, spec: str = ".2f", suffix: str = "") -> str:
        return "n/a" if value is None or (isinstance(value, float) and np.isnan(value)) \
            else f"{value:{spec}}{suffix}"

    lines = [
        "=" * 64,
        title.center(64),
        "=" * 64,
        "  Setups",
        f"    Armed / triggered  : {s['setups_armed']} / {s['setups_triggered']}"
        f"   (fill rate {num((s['fill_rate'] or 0) * 100, '.1f', '%')})",
        f"    Time with a setup  : {num(s['bars_armed_pct'] * 100, '.1f', '%')} of bars",
        "",
        "  Trades",
        f"    Closed trades      : {t['n_trades']}",
    ]
    if t["n_trades"]:
        by = t["by_side"]
        lines += [
            f"    Long / short       : {by['long']['n']} / {by['short']['n']}",
            f"    Win rate           : {num(t['win_rate'] * 100, '.1f', '%')}"
            f"   (break-even needs {num(t['breakeven_win_rate'] * 100, '.1f', '%')})",
            f"    Planned reward:risk: {num(t['median_planned_rr'], '.2f')} median"
            f"  ({num(t['avg_planned_rr'], '.2f')} mean)",
            f"    Avg win / avg loss : {num(t['avg_win'])} / {num(t['avg_loss'])} $/oz",
            f"    Expectancy         : {num(t['expectancy'], '.3f')} $/oz"
            f"   ({num(t['avg_r_multiple'], '.3f')} R)",
            f"    Profit factor      : {num(t['profit_factor'])}",
            f"    Avg bars held      : {num(t['avg_bars_held'], '.1f')}",
            f"    Costs paid         : {num(t['total_costs'])} $/oz",
        ]
    lines += [
        "",
        "  Equity",
        f"    Net PnL            : {num(e['net_pnl'])} $/oz",
        f"    Buy & hold         : {num(e['buy_hold_pnl'])} $/oz",
        f"    Max drawdown       : {num(e['max_drawdown'])} $/oz",
        f"    Return / max DD    : {num(e['return_over_max_drawdown'])}",
        f"    Sharpe / Sortino   : {num(e['sharpe'])} / {num(e['sortino'])}",
        f"    Time in market     : {num(e['exposure'] * 100, '.1f', '%')}",
        f"    Period             : {str(e['start'])[:16]} -> {str(e['end'])[:16]}",
        "=" * 64,
    ]
    if t["n_trades"] and t["n_trades"] < 30:
        lines += [
            f"  NOTE: {t['n_trades']} trades is too few for the win rate or Sharpe",
            "        to be statistically meaningful. Directional read only.",
            "=" * 64,
        ]
    if t.get("exit_reasons"):
        reasons = ", ".join(f"{k}={v}" for k, v in t["exit_reasons"].items())
        lines += ["  Exit reasons: " + reasons, "=" * 64]
    return "\n".join(lines)
