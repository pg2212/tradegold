"""Orchestration: data -> context -> setups -> triggers -> trades -> metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .artifacts import RunDirectory
from .backtest import BacktestResult, run_backtest
from .capital import format_capital, size_trades
from .config import Config
from .data import load_timeframes
from .logging_utils import get_logger
from .metrics import format_report, summarise
from .reporting import plot_strategy
from .strategy import run_strategy

logger = get_logger(__name__)


@dataclass
class Outcome:
    signals: pd.DataFrame
    context: pd.DataFrame
    result: BacktestResult
    metrics: dict[str, Any] = field(default_factory=dict)
    report: str = ""
    run_dir: RunDirectory | None = None


def run_pipeline(cfg: Config, *, write_artifacts: bool = True) -> Outcome:
    entry_bars, context_bars = load_timeframes(cfg.data)

    signals, context = run_strategy(
        entry_bars, context_bars, cfg.strategy, context_interval=cfg.data.context_interval
    )
    result = run_backtest(signals, cfg.execution)
    metrics = summarise(result, signals, float(cfg.execution.annualization_bars))

    if cfg.capital.enabled:
        sized, cap_stats = size_trades(result.trades, cfg.capital)
        result.trades = sized
        metrics["capital"] = cap_stats
    metrics["config_fingerprint"] = cfg.fingerprint()
    metrics["data"] = {
        "ticker": cfg.data.ticker,
        "entry_interval": cfg.data.entry_interval,
        "context_interval": cfg.data.context_interval,
        "entry_bars": len(entry_bars),
        "context_bars": len(context_bars),
    }

    title = (
        f"GOLD STRATEGY | {cfg.data.ticker} | "
        f"{cfg.data.entry_interval} entries on {cfg.data.context_interval} setups | "
        f"{cfg.run.name}"
    )
    report = format_report(metrics, title)
    if cfg.capital.enabled:
        report += "\n" + format_capital(metrics["capital"]) + "\n" + "=" * 64
    outcome = Outcome(signals=signals, context=context, result=result,
                      metrics=metrics, report=report)

    if write_artifacts:
        run_dir = RunDirectory(cfg)
        run_dir.write_json("metrics.json", metrics)
        run_dir.write_text("report.txt", report + "\n")
        run_dir.write_frame("trades.csv", result.trades)
        run_dir.write_frame("equity.csv", result.equity)
        run_dir.write_frame("signals.parquet", signals)
        run_dir.write_frame("context.parquet", context)
        if cfg.report.plot:
            plot_strategy(
                signals, result, run_dir.path / "strategy_chart.png", title,
                bars=cfg.report.plot_bars,
            )
        run_dir.finalise({"n_trades": int(len(result.trades))})
        outcome.run_dir = run_dir

    return outcome
