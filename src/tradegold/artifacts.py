"""Versioned run directories.

Every run writes its config, metrics, trade ledger, equity curve and logs to a
directory stamped with the config fingerprint, so a result can always be traced
back to the exact settings that produced it.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config
from .logging_utils import get_logger

logger = get_logger(__name__)


def _git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


class RunDirectory:
    def __init__(self, cfg: Config) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.fingerprint = cfg.fingerprint()
        self.path = Path(cfg.run.output_dir) / f"{stamp}__{cfg.run.name}__{self.fingerprint}"
        self.path.mkdir(parents=True, exist_ok=True)
        self._cfg = cfg
        (self.path / "config.yaml").write_text(cfg.to_yaml())
        logger.info("Run directory: %s", self.path)

    @property
    def log_file(self) -> Path:
        return self.path / "run.log"

    def write_json(self, name: str, payload: Any) -> Path:
        target = self.path / name
        target.write_text(json.dumps(payload, indent=2, default=str))
        return target

    def write_frame(self, name: str, frame: pd.DataFrame) -> Path:
        target = self.path / name
        if target.suffix == ".parquet":
            frame.to_parquet(target)
        else:
            frame.to_csv(target)
        return target

    def write_text(self, name: str, text: str) -> Path:
        target = self.path / name
        target.write_text(text)
        return target

    def finalise(self, extra: dict[str, Any] | None = None) -> Path:
        manifest = {
            "run_name": self._cfg.run.name,
            "config_fingerprint": self.fingerprint,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_revision": _git_revision(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "artifacts": sorted(p.name for p in self.path.iterdir()),
        }
        if extra:
            manifest.update(extra)
        return self.write_json("manifest.json", manifest)
