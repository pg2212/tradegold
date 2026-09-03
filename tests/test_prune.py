"""Prune decides what gets deleted, so its rules need to be exact."""

from __future__ import annotations

import json

import pytest

from tradegold.prune import describe, execute, plan


def make_run(root, stamp, name, fingerprint, *, readable=True, trades=5):
    d = root / f"{stamp}__{name}__{fingerprint}"
    d.mkdir(parents=True)
    if readable:
        (d / "signals.parquet").write_bytes(b"x" * 100)
    (d / "metrics.json").write_text(json.dumps({
        "trades": {"n_trades": trades}, "equity": {"net_pnl": 1.0}
    }))
    return d


@pytest.fixture
def runs(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    return root


def names(candidates):
    return {c.path.name for c in candidates}


def test_directories_without_signal_artifacts_are_removed(runs):
    old = make_run(runs, "20260101T000000Z", "legacy", "aaa", readable=False)
    new = make_run(runs, "20260102T000000Z", "current", "bbb")

    remove, keep = plan(runs)
    assert names(remove) == {old.name}
    assert names(keep) == {new.name}
    assert "retired pipeline" in remove[0].reason


def test_duplicate_fingerprints_keep_only_the_newest(runs):
    older = make_run(runs, "20260101T000000Z", "run", "same")
    newer = make_run(runs, "20260102T000000Z", "run", "same")

    remove, keep = plan(runs, keep_per_config=1)
    assert names(keep) == {newer.name}
    assert names(remove) == {older.name}


def test_keep_per_config_can_retain_more_than_one(runs):
    make_run(runs, "20260101T000000Z", "run", "same")
    make_run(runs, "20260102T000000Z", "run", "same")
    make_run(runs, "20260103T000000Z", "run", "same")

    remove, keep = plan(runs, keep_per_config=2)
    assert len(keep) == 2 and len(remove) == 1


def test_keep_per_name_drops_superseded_revisions(runs):
    """A changed config gets a new fingerprint, which must not protect it."""
    old_rev = make_run(runs, "20260101T000000Z", "strategy", "oldhash")
    new_rev = make_run(runs, "20260102T000000Z", "strategy", "newhash")
    other = make_run(runs, "20260103T000000Z", "variant", "otherhash")

    remove, keep = plan(runs, keep_per_name=1)
    assert names(keep) == {new_rev.name, other.name}
    assert names(remove) == {old_rev.name}
    assert "superseded" in remove[0].reason


def test_keep_names_filters_by_run_name(runs):
    keeper = make_run(runs, "20260102T000000Z", "wanted", "aaa")
    make_run(runs, "20260101T000000Z", "unwanted", "bbb")

    remove, keep = plan(runs, keep_names={"wanted"})
    assert names(keep) == {keeper.name}
    assert "not in the keep list" in remove[0].reason


def test_malformed_directory_names_are_ignored(runs):
    (runs / "not-a-run").mkdir()
    kept = make_run(runs, "20260101T000000Z", "run", "aaa")

    remove, keep = plan(runs)
    assert names(keep) == {kept.name}
    assert remove == []
    assert (runs / "not-a-run").exists(), "unrecognised directories must be left alone"


def test_execute_deletes_only_the_planned_directories(runs):
    doomed = make_run(runs, "20260101T000000Z", "run", "same")
    survivor = make_run(runs, "20260102T000000Z", "run", "same")

    remove, keep = plan(runs)
    freed = execute(remove)

    assert not doomed.exists()
    assert survivor.exists()
    assert freed > 0


def test_describe_is_explicit_about_whether_anything_was_deleted(runs):
    make_run(runs, "20260101T000000Z", "run", "same")
    make_run(runs, "20260102T000000Z", "run", "same")
    remove, keep = plan(runs)

    dry = describe(remove, keep, dry_run=True)
    assert "Would remove" in dry and "Nothing was deleted" in dry
    wet = describe(remove, keep, dry_run=False)
    assert wet.startswith("Removed") and "Nothing was deleted" not in wet
