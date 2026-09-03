"""Expanding-window walk-forward evaluation.

This strategy fits no model, so a walk-forward split is not about leakage from a
training set. It answers a different and more awkward question: **would a choice
you made using only the data available at the time have worked afterwards?**

The depth sweep is the motivating case. Optimising entry depth over a whole year
picked 0.55 and showed +$61/oz — but that value was chosen with knowledge of the
whole year. Here each fold selects its parameter on everything *before* the fold
and is scored only on the fold itself, so a parameter that only ever worked in
hindsight has nowhere to hide.

Two modes:

  * `fixed`  -- one parameter set throughout; the folds are pure consistency
    checks, showing whether results hold across successive regimes.
  * `select` -- each fold re-picks from candidates using prior data only. The
    aggregate is then a genuine out-of-sample record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Fold:
    """One expanding-window step. `train` is everything before `test`."""

    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def describe(self) -> str:
        return (
            f"{self.name}: train {self.train_start:%Y-%m-%d} -> {self.train_end:%Y-%m-%d}"
            f" | test {self.test_start:%Y-%m-%d} -> {self.test_end:%Y-%m-%d}"
        )


def expanding_folds(
    index: pd.DatetimeIndex, n_folds: int = 4, min_train_frac: float = 0.30
) -> list[Fold]:
    """Split a time index into expanding-window folds.

    Fold k trains on everything from the start up to its test window, so later
    folds see strictly more history -- the shape a live system actually has.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    if not 0 < min_train_frac < 1:
        raise ValueError(f"min_train_frac must be in (0, 1), got {min_train_frac}")

    n = len(index)
    first_test = int(n * min_train_frac)
    remaining = n - first_test
    if remaining < n_folds:
        raise ValueError(
            f"Not enough bars after the warm-up window ({remaining}) for {n_folds} folds"
        )

    size = remaining // n_folds
    folds: list[Fold] = []
    for k in range(n_folds):
        start = first_test + k * size
        end = first_test + (k + 1) * size if k < n_folds - 1 else n
        folds.append(Fold(
            name=f"fold_{k + 1}",
            train_start=index[0], train_end=index[start - 1],
            test_start=index[start], test_end=index[end - 1],
        ))
    return folds


def _window(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if trades.empty:
        return trades
    entry = trades["entry_time"]
    return trades[(entry >= start) & (entry <= end)]


def _stats(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {"n": 0, "net": 0.0, "win_rate": None, "profit_factor": None}
    pnl = trades["net_pnl"].to_numpy(float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gross_loss = float(-losses.sum())
    return {
        "n": len(trades),
        "net": float(pnl.sum()),
        "win_rate": float(len(wins) / len(pnl)),
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else None,
        "expectancy": float(pnl.mean()),
    }


@dataclass
class WalkForwardResult:
    folds: list[Fold]
    rows: list[dict[str, Any]] = field(default_factory=list)
    mode: str = "fixed"

    @property
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def summary(self) -> dict[str, Any]:
        df = self.frame
        if df.empty:
            return {"folds": 0, "total_net": 0.0, "positive_folds": 0}
        return {
            "mode": self.mode,
            "folds": len(df),
            "total_net": float(df["test_net"].sum()),
            "positive_folds": int((df["test_net"] > 0).sum()),
            "total_trades": int(df["test_n"].sum()),
            "worst_fold": float(df["test_net"].min()),
            "best_fold": float(df["test_net"].max()),
        }


def evaluate_fixed(trades: pd.DataFrame, folds: list[Fold]) -> WalkForwardResult:
    """Score one fixed parameter set across every fold."""
    result = WalkForwardResult(folds=folds, mode="fixed")
    for f in folds:
        test = _stats(_window(trades, f.test_start, f.test_end))
        result.rows.append({
            "fold": f.name,
            "test_start": f.test_start, "test_end": f.test_end,
            "test_n": test["n"], "test_net": test["net"],
            "test_win": test["win_rate"], "test_pf": test["profit_factor"],
        })
    return result


def evaluate_select(
    candidates: dict[Any, pd.DataFrame], folds: list[Fold], *, label: str = "param"
) -> WalkForwardResult:
    """Pick the best candidate on each fold's history, score it on the fold.

    `candidates` maps a parameter value to the full-period trade ledger produced
    by that value. Selection reads only the training window; the score comes only
    from the test window. A parameter that wins purely in hindsight will be
    chosen in one fold and fail in the next, which is the point.
    """
    result = WalkForwardResult(folds=folds, mode="select")
    for f in folds:
        scored = {
            value: _stats(_window(trades, f.train_start, f.train_end))
            for value, trades in candidates.items()
        }
        # Require the choice to have actually traded in-sample.
        usable = {v: s for v, s in scored.items() if s["n"] >= 5}
        if not usable:
            logger.warning("%s: no candidate traded enough in-sample; skipping", f.name)
            continue
        best = max(usable, key=lambda v: usable[v]["net"])

        test = _stats(_window(candidates[best], f.test_start, f.test_end))
        result.rows.append({
            "fold": f.name,
            "test_start": f.test_start, "test_end": f.test_end,
            label: best,
            "train_n": usable[best]["n"], "train_net": usable[best]["net"],
            "test_n": test["n"], "test_net": test["net"],
            "test_win": test["win_rate"], "test_pf": test["profit_factor"],
        })
    return result


def format_walk_forward(result: WalkForwardResult, title: str, label: str = "param") -> str:
    df = result.frame
    lines = ["=" * 78, title.center(78), "=" * 78]
    if df.empty:
        return "\n".join(lines + ["  no folds produced results", "=" * 78])

    has_sel = label in df.columns
    head = f"  {'fold':8s} {'test window':23s}"
    if has_sel:
        head += f" {label:>7s} {'train':>9s}"
    head += f" {'trades':>7s} {'net $/oz':>10s} {'win%':>6s} {'PF':>6s}"
    lines += [head, "  " + "-" * 74]

    for r in df.itertuples():
        row = (f"  {r.fold:8s} {r.test_start:%Y-%m-%d} -> {r.test_end:%Y-%m-%d}  ")
        if has_sel:
            row += f"{getattr(r, label):>7} {r.train_net:>+9.2f}"
        win = f"{r.test_win * 100:.1f}%" if r.test_win is not None else "n/a"
        pf = f"{r.test_pf:.2f}" if r.test_pf is not None else "n/a"
        row += f" {r.test_n:>7d} {r.test_net:>+10.2f} {win:>6s} {pf:>6s}"
        lines.append(row)

    s = result.summary()
    lines += [
        "  " + "-" * 74,
        f"  {'TOTAL':8s} {'':23s}" + (f" {'':7s} {'':9s}" if has_sel else "")
        + f" {s['total_trades']:>7d} {s['total_net']:>+10.2f}",
        "=" * 78,
        f"  Folds in profit: {s['positive_folds']}/{s['folds']}"
        f"   best {s['best_fold']:+.2f}   worst {s['worst_fold']:+.2f}",
        "=" * 78,
    ]
    return "\n".join(lines)
