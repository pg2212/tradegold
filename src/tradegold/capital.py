"""Position sizing: converts per-unit price results into an account.

The backtest decides *which* trades happen from price alone, so sizing can be
applied afterwards without changing entries or exits. What it does change is how
much each trade matters. Unsized, a trade risking $4/oz and one risking $78/oz
count equally — which is neither how anyone trades nor a fair weighting, since
the wide-stop trades quietly dominate the result.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import CapitalConfig
from .logging_utils import get_logger

logger = get_logger(__name__)


def size_trades(trades: pd.DataFrame, cfg: CapitalConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Walk the ledger chronologically, sizing each trade off equity at the time.

    Returns the ledger with `lots`, `pnl_usd` and `equity` columns, plus summary
    statistics in account terms. Trades too large for one minimum lot are
    recorded with `lots = 0` and skipped -- an account that cannot afford the
    smallest position simply does not take the trade.
    """
    out = trades.copy()
    if out.empty:
        return out, {"enabled": True, "final_equity": cfg.initial, "return_pct": 0.0,
                     "trades_taken": 0, "trades_skipped": 0}

    out = out.sort_values("entry_time").reset_index(drop=True)
    risk_per_unit = (out["entry_price"] - out["stop_price"]).abs().to_numpy(float)
    pnl_per_unit = out["net_pnl"].to_numpy(float)
    entry_price = out["entry_price"].to_numpy(float)

    equity = cfg.initial
    lots, pnl_usd, equity_after, skipped = [], [], [], 0

    for i in range(len(out)):
        base = equity if cfg.compound else cfg.initial
        budget = base * cfg.risk_per_trade_pct / 100.0
        risk_per_lot = risk_per_unit[i] * cfg.contract_size

        raw = budget / risk_per_lot if risk_per_lot > 0 else 0.0
        n_lots = np.floor(raw / cfg.lot_step) * cfg.lot_step

        # Refuse positions the account cannot carry.
        notional = n_lots * cfg.contract_size * entry_price[i]
        if notional > base * cfg.max_leverage:
            n_lots = np.floor(
                (base * cfg.max_leverage) / (cfg.contract_size * entry_price[i]) / cfg.lot_step
            ) * cfg.lot_step

        if n_lots <= 0:
            skipped += 1
            lots.append(0.0)
            pnl_usd.append(0.0)
            equity_after.append(equity)
            continue

        trade_pnl = pnl_per_unit[i] * n_lots * cfg.contract_size
        equity += trade_pnl
        lots.append(float(n_lots))
        pnl_usd.append(float(trade_pnl))
        equity_after.append(equity)

    out["lots"] = lots
    out["pnl_usd"] = pnl_usd
    out["equity"] = equity_after

    taken = out[out["lots"] > 0]
    curve = np.concatenate([[cfg.initial], out["equity"].to_numpy(float)])
    peak = np.maximum.accumulate(curve)
    dd = curve - peak
    dd_pct = dd / peak

    stats = {
        "enabled": True,
        "initial": cfg.initial,
        "final_equity": float(equity),
        "profit_usd": float(equity - cfg.initial),
        "return_pct": float((equity / cfg.initial - 1) * 100),
        "max_drawdown_usd": float(dd.min()),
        "max_drawdown_pct": float(dd_pct.min() * 100),
        "trades_taken": int(len(taken)),
        "trades_skipped": skipped,
        "median_lots": float(taken["lots"].median()) if len(taken) else 0.0,
        "max_lots": float(taken["lots"].max()) if len(taken) else 0.0,
        "risk_per_trade_pct": cfg.risk_per_trade_pct,
        "contract_size": cfg.contract_size,
    }
    if skipped:
        logger.warning(
            "%d/%d trades skipped: one %g-unit lot risks more than %.2f%% of equity. "
            "Use a smaller contract (e.g. micro), raise risk_per_trade_pct, or add capital.",
            skipped, len(out), cfg.contract_size, cfg.risk_per_trade_pct,
        )
    return out, stats


def format_capital(stats: dict[str, Any]) -> str:
    if not stats.get("enabled"):
        return ""
    lines = [
        "  Account",
        f"    Starting capital   : ${stats['initial']:,.0f}",
        f"    Final equity       : ${stats['final_equity']:,.0f}",
        f"    Profit / loss      : ${stats['profit_usd']:+,.0f}  "
        f"({stats['return_pct']:+.2f}%)",
        f"    Max drawdown       : ${stats['max_drawdown_usd']:,.0f}  "
        f"({stats['max_drawdown_pct']:.2f}%)",
        f"    Risk per trade     : {stats['risk_per_trade_pct']:.2f}% of equity",
        f"    Position size      : median {stats['median_lots']:g} lots, "
        f"max {stats['max_lots']:g}  ({stats['contract_size']:g} units each)",
        f"    Trades taken       : {stats['trades_taken']}"
        + (f"   ({stats['trades_skipped']} too large to afford)"
           if stats["trades_skipped"] else ""),
    ]
    return "\n".join(lines)
