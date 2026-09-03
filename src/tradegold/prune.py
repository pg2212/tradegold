"""Run-directory housekeeping.

Runs accumulate fast -- every chart tweak produces another directory with
identical settings, and old pipelines leave behind directories the dashboard
cannot even read. Both clutter the run picker, and a stale run showing
superseded numbers is worse than clutter: it invites reading the wrong result.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .logging_utils import get_logger

logger = get_logger(__name__)

# A run directory must contain this to be readable by the current dashboard.
SIGNAL_ARTIFACT = "signals.parquet"


@dataclass
class Candidate:
    path: Path
    name: str
    fingerprint: str
    stamp: str
    reason: str
    n_trades: int | None = None
    size_bytes: int = 0


def _size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _trades(path: Path) -> int | None:
    try:
        return json.loads((path / "metrics.json").read_text())["trades"]["n_trades"]
    except (OSError, ValueError, KeyError):
        return None


def plan(
    root: Path,
    *,
    keep_per_config: int = 1,
    keep_per_name: int | None = None,
    keep_names: set[str] | None = None,
) -> tuple[list[Candidate], list[Candidate]]:
    """Decide what to remove. Returns (remove, keep), newest first.

    Three independent rules, applied newest-first:

      * a directory without signal artifacts belongs to a retired pipeline;
      * beyond `keep_per_config` runs sharing a config *fingerprint*, the older
        ones are re-runs of identical settings;
      * beyond `keep_per_name` runs sharing a run *name*, the older ones are
        superseded revisions of that configuration. This is the rule that drops
        a run whose settings have since changed -- a new fingerprint alone does
        not make an old result worth keeping.

    `keep_names` additionally restricts survivors to those run names.
    """
    remove: list[Candidate] = []
    keep: list[Candidate] = []
    seen: dict[str, int] = {}
    seen_names: dict[str, int] = {}

    for path in sorted(root.glob("*__*__*"), reverse=True):
        if not path.is_dir():
            continue
        try:
            stamp, name, fingerprint = path.name.split("__", 2)
        except ValueError:
            continue

        base = dict(path=path, name=name, fingerprint=fingerprint, stamp=stamp,
                    n_trades=_trades(path), size_bytes=_size(path))

        if not (path / SIGNAL_ARTIFACT).exists():
            remove.append(Candidate(**base, reason="no signal artifacts (retired pipeline)"))
            continue

        count = seen.get(fingerprint, 0)
        seen[fingerprint] = count + 1
        name_count = seen_names.get(name, 0)
        seen_names[name] = name_count + 1

        if count >= keep_per_config:
            remove.append(Candidate(**base, reason="duplicate config fingerprint"))
        elif keep_names is not None and name not in keep_names:
            remove.append(Candidate(**base, reason="run name not in the keep list"))
        elif keep_per_name is not None and name_count >= keep_per_name:
            remove.append(Candidate(**base, reason="superseded revision of this config"))
        else:
            keep.append(Candidate(**base, reason="kept"))
    return remove, keep


def execute(remove: list[Candidate]) -> int:
    freed = 0
    for c in remove:
        shutil.rmtree(c.path)
        freed += c.size_bytes
        logger.info("Removed %s (%s)", c.path.name, c.reason)
    return freed


def describe(remove: list[Candidate], keep: list[Candidate], *, dry_run: bool) -> str:
    lines: list[str] = []
    verb = "Would remove" if dry_run else "Removed"
    freed = sum(c.size_bytes for c in remove)

    lines.append(f"{verb} {len(remove)} run(s), {freed / 1e6:.1f} MB:")
    for c in remove:
        trades = "" if c.n_trades is None else f"{c.n_trades} trades, "
        lines.append(f"    {c.path.name}  ({trades}{c.reason})")
    lines.append("")
    lines.append(f"Keeping {len(keep)}:")
    for c in keep:
        trades = "" if c.n_trades is None else f"{c.n_trades} trades"
        lines.append(f"    {c.path.name}  ({trades})")
    if dry_run:
        lines += ["", "Nothing was deleted. Re-run with --yes to apply."]
    return "\n".join(lines)
