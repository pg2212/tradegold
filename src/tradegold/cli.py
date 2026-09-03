"""Command-line interface.

    tradegold run       -c configs/default.yaml --set model.name=logistic
    tradegold fetch     -c configs/default.yaml --refresh
    tradegold registry
    tradegold show-config -c configs/aggressive.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import Config, load_config
from .logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_CONFIG = Path("configs/default.yaml")


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c", "--config", type=Path, default=DEFAULT_CONFIG, help="Path to a YAML config"
    )
    parser.add_argument(
        "-b", "--base", type=Path, default=None,
        help="Base config that --config is layered over (deep merge)",
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="Override any config value, e.g. --set model.params.max_depth=6 (repeatable)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tradegold", description=__doc__)
    parser.add_argument("--version", action="version", version=f"tradegold {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the full pipeline and write a run directory")
    _add_config_args(run)
    run.add_argument("--no-artifacts", action="store_true", help="Skip writing a run directory")
    run.add_argument("--no-plot", action="store_true", help="Skip the strategy chart")

    fetch = sub.add_parser("fetch", help="Download and cache market data only")
    _add_config_args(fetch)
    fetch.add_argument("--refresh", action="store_true", help="Ignore the cache")

    show = sub.add_parser("show-config", help="Print the fully resolved config")
    _add_config_args(show)

    wf = sub.add_parser(
        "walkforward",
        help="Expanding-window evaluation; optionally re-select a parameter per fold",
    )
    _add_config_args(wf)
    wf.add_argument("--folds", type=int, default=4, help="Number of expanding folds")
    wf.add_argument("--min-train-frac", type=float, default=0.30,
                    help="Fraction of history reserved before the first test fold")
    wf.add_argument("--select", choices=["none", "depth", "volume"], default="none",
                    help="Re-select this parameter on each fold's history")

    sub.add_parser("registry", help="List every pluggable component")

    prune = sub.add_parser("prune", help="Remove stale and duplicate run directories")
    prune.add_argument("--runs-dir", type=Path, default=Path("runs"))
    prune.add_argument("--keep-per-config", type=int, default=1, metavar="N",
                       help="Runs to keep per config fingerprint (default 1)")
    prune.add_argument("--keep-per-name", type=int, default=None, metavar="N",
                       help="Runs to keep per run name, dropping superseded revisions")
    prune.add_argument("--keep-name", action="append", default=None, metavar="NAME",
                       help="Only keep runs with these names (repeatable)")
    prune.add_argument("--yes", action="store_true",
                       help="Actually delete. Without this the command only reports.")
    return parser


def _load(args: argparse.Namespace) -> Config:
    if not args.config.exists():
        raise SystemExit(f"Config not found: {args.config}")
    return load_config(args.config, base=args.base, overrides=args.overrides)


def _cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run_pipeline

    cfg = _load(args)
    if args.no_plot:
        cfg.report.plot = False
    setup_logging(cfg.run.log_level if not args.verbose else "DEBUG")
    logger.info("Config fingerprint %s", cfg.fingerprint())

    outcome = run_pipeline(cfg, write_artifacts=not args.no_artifacts)
    print("\n" + outcome.report)
    if outcome.run_dir:
        print(f"\nArtifacts: {outcome.run_dir.path}")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    from .data import load_timeframes

    cfg = _load(args)
    setup_logging(cfg.run.log_level)
    if args.refresh:
        cfg.data.refresh = True
    entry, context = load_timeframes(cfg.data)
    for name, df, interval in (("entry", entry, cfg.data.entry_interval),
                               ("context", context, cfg.data.context_interval)):
        print(f"{name:8s} {interval:4s} {len(df):7,d} bars | {df.index[0]} -> {df.index[-1]}")
    return 0


def _cmd_show_config(args: argparse.Namespace) -> int:
    print(_load(args).to_yaml())
    return 0


CANDIDATES = {
    "depth": [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
    "volume": [1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
}


def _apply_candidate(cfg, which: str, value: float):
    """Return a copy of cfg with one swept parameter set to `value`."""
    data = cfg.model_dump()
    if which == "depth":
        data["strategy"]["long_retracement"] = value
        data["strategy"]["short_retracement"] = round(1 - value, 4)
    else:
        data["strategy"]["volume_multiplier"] = value
    return type(cfg).model_validate(data)


def _cmd_walkforward(args: argparse.Namespace) -> int:
    from .evaluation import (
        evaluate_fixed,
        evaluate_select,
        expanding_folds,
        format_walk_forward,
    )
    from .pipeline import run_pipeline

    cfg = _load(args)
    cfg.report.plot = False
    setup_logging(cfg.run.log_level if not args.verbose else "DEBUG")

    base = run_pipeline(cfg, write_artifacts=False)
    folds = expanding_folds(base.signals.index, args.folds, args.min_train_frac)
    for f in folds:
        logger.info(f.describe())

    if args.select == "none":
        result = evaluate_fixed(base.result.trades, folds)
        title = f"WALK-FORWARD (fixed parameters) | {cfg.run.name}"
        label = "param"
    else:
        label = args.select
        candidates = {}
        for value in CANDIDATES[label]:
            out = run_pipeline(_apply_candidate(cfg, label, value), write_artifacts=False)
            candidates[value] = out.result.trades
            logger.info("  candidate %s=%s -> %d trades over the full period",
                        label, value, len(out.result.trades))
        result = evaluate_select(candidates, folds, label=label)
        title = f"WALK-FORWARD (re-selecting {label} per fold) | {cfg.run.name}"

    print("\n" + format_walk_forward(result, title, label))

    if args.select != "none":
        fixed = evaluate_fixed(base.result.trades, folds)
        print("\n" + format_walk_forward(
            fixed, f"SAME FOLDS, PARAMETER LEFT ALONE | {cfg.run.name}", label))
        chosen = result.summary()["total_net"]
        left = fixed.summary()["total_net"]
        verdict = ("selection helped" if chosen > left
                   else "selection did NOT beat leaving it alone")
        print(f"\n  Out-of-sample: re-selecting {label} {chosen:+.2f} $/oz "
              f"vs fixed {left:+.2f} $/oz  ->  {verdict}.")
    return 0


def _cmd_prune(args: argparse.Namespace) -> int:
    from .prune import describe, execute, plan

    setup_logging("INFO")
    if not args.runs_dir.exists():
        raise SystemExit(f"No such directory: {args.runs_dir}")

    keep_names = set(args.keep_name) if args.keep_name else None
    remove, keep = plan(
        args.runs_dir,
        keep_per_config=args.keep_per_config,
        keep_per_name=args.keep_per_name,
        keep_names=keep_names,
    )
    if not remove:
        print("Nothing to prune.")
        return 0
    if args.yes:
        execute(remove)
    print(describe(remove, keep, dry_run=not args.yes))
    return 0


def _cmd_registry(_: argparse.Namespace) -> int:
    from .data import PROVIDERS

    print(f"{'data providers':20s}: {', '.join(PROVIDERS.names())}")
    print(f"{'execution stops':20s}: setup_extreme, atr")
    print(f"{'execution targets':20s}: setup_extreme, r_multiple")
    return 0


COMMANDS = {
    "run": _cmd_run,
    "fetch": _cmd_fetch,
    "show-config": _cmd_show_config,
    "registry": _cmd_registry,
    "prune": _cmd_prune,
    "walkforward": _cmd_walkforward,
}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return 130
    except Exception as exc:  # surfaced as a clean CLI error, full trace at -v
        logger.error("%s: %s", type(exc).__name__, exc)
        if getattr(args, "verbose", False):
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
