"""Walk-forward folds and out-of-sample parameter selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradegold.evaluation import (
    evaluate_fixed,
    evaluate_select,
    expanding_folds,
    format_walk_forward,
)


def index(n=1000):
    return pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")


def ledger(times, pnls):
    return pd.DataFrame({"entry_time": pd.DatetimeIndex(times), "net_pnl": pnls})


# ------------------------------------------------------------------ folds ---

def test_folds_expand_and_tile_the_test_period():
    folds = expanding_folds(index(1000), n_folds=4, min_train_frac=0.30)

    assert len(folds) == 4
    for a, b in zip(folds, folds[1:], strict=False):
        assert b.train_end > a.train_end, "training window must expand"
        assert b.test_start > a.test_end, "test windows must not overlap"
    assert all(f.train_start == folds[0].train_start for f in folds)


def test_first_fold_respects_the_warm_up_fraction():
    idx = index(1000)
    folds = expanding_folds(idx, n_folds=4, min_train_frac=0.40)
    assert folds[0].test_start == idx[400]


def test_training_window_always_precedes_its_test_window():
    for f in expanding_folds(index(800), n_folds=5, min_train_frac=0.25):
        assert f.train_end < f.test_start


def test_last_fold_reaches_the_end_of_the_data():
    idx = index(997)          # deliberately not divisible
    folds = expanding_folds(idx, n_folds=4, min_train_frac=0.30)
    assert folds[-1].test_end == idx[-1], "no bars may be silently dropped"


def test_invalid_fold_settings_are_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        expanding_folds(index(), n_folds=1)
    with pytest.raises(ValueError, match="min_train_frac"):
        expanding_folds(index(), n_folds=3, min_train_frac=1.5)
    with pytest.raises(ValueError, match="Not enough bars"):
        expanding_folds(index(10), n_folds=8, min_train_frac=0.9)


# ------------------------------------------------------------ evaluation ---

def test_fixed_evaluation_assigns_each_trade_to_one_fold():
    idx = index(1000)
    folds = expanding_folds(idx, n_folds=4, min_train_frac=0.30)
    times = [f.test_start for f in folds] * 2
    result = evaluate_fixed(ledger(times, [1.0] * 8), folds)

    assert len(result.rows) == 4
    assert all(r["test_n"] == 2 for r in result.rows)
    assert result.summary()["total_trades"] == 8


def test_trades_before_the_first_fold_are_excluded():
    idx = index(1000)
    folds = expanding_folds(idx, n_folds=4, min_train_frac=0.30)
    early = ledger([idx[0], idx[10]], [99.0, 99.0])   # inside the warm-up
    result = evaluate_fixed(early, folds)
    assert result.summary()["total_trades"] == 0


def test_selection_uses_only_prior_data():
    """A candidate that only wins AFTER the fold must never be picked."""
    idx = index(1000)
    folds = expanding_folds(idx, n_folds=2, min_train_frac=0.40)
    f = folds[0]

    honest = ledger([f.train_start, f.train_end, f.test_start], [10.0, 10.0, 1.0])
    hindsight = ledger([f.train_start, f.train_end, f.test_start], [-10.0, -10.0, 500.0])
    # Pad both so each clears the minimum in-sample trade count.
    pad = [f.train_start] * 5
    honest = pd.concat([honest, ledger(pad, [1.0] * 5)], ignore_index=True)
    hindsight = pd.concat([hindsight, ledger(pad, [-1.0] * 5)], ignore_index=True)

    result = evaluate_select({"good": honest, "hindsight": hindsight}, folds, label="p")
    assert result.rows[0]["p"] == "good", "selection must not see the test window"


def test_selection_skips_candidates_that_barely_traded():
    idx = index(1000)
    folds = expanding_folds(idx, n_folds=2, min_train_frac=0.40)
    f = folds[0]
    thin = ledger([f.train_start], [999.0])                    # 1 trade in-sample
    thick = ledger([f.train_start] * 10, [1.0] * 10)           # 10 trades

    result = evaluate_select({"thin": thin, "thick": thick}, folds, label="p")
    assert result.rows[0]["p"] == "thick"


def test_summary_counts_profitable_folds():
    idx = index(1000)
    folds = expanding_folds(idx, n_folds=4, min_train_frac=0.30)
    times = [f.test_start for f in folds]
    result = evaluate_fixed(ledger(times, [5.0, -5.0, 5.0, -5.0]), folds)

    s = result.summary()
    assert s["positive_folds"] == 2
    assert s["total_net"] == pytest.approx(0.0)
    assert s["best_fold"] == pytest.approx(5.0)
    assert s["worst_fold"] == pytest.approx(-5.0)


def test_report_renders_without_a_selection_column():
    idx = index(1000)
    folds = expanding_folds(idx, n_folds=3, min_train_frac=0.30)
    text = format_walk_forward(
        evaluate_fixed(ledger([f.test_start for f in folds], [1.0, 2.0, 3.0]), folds),
        "TITLE",
    )
    assert "TITLE" in text and "Folds in profit: 3/3" in text


def test_report_handles_an_empty_result():
    folds = expanding_folds(index(1000), n_folds=3, min_train_frac=0.30)
    empty = ledger(pd.DatetimeIndex([]), np.array([], dtype=float))
    text = format_walk_forward(evaluate_fixed(empty, folds), "EMPTY")
    assert "EMPTY" in text
