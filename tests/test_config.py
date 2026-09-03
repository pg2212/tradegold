from __future__ import annotations

import pytest
from pydantic import ValidationError

from tradegold.config import Config, apply_overrides, load_config


def test_annualization_derives_from_the_entry_interval():
    assert Config.model_validate({"data": {"entry_interval": "5m"}}) \
        .execution.annualization_bars == 252 * 23 * 12
    assert Config.model_validate({"data": {"entry_interval": "15m"}}) \
        .execution.annualization_bars == 252 * 23 * 4


def test_entry_timeframe_must_be_finer_than_context():
    with pytest.raises(ValidationError, match="must be finer"):
        Config.model_validate({"data": {"entry_interval": "60m", "context_interval": "60m"}})


def test_unknown_interval_is_rejected_with_the_valid_list():
    with pytest.raises(ValidationError, match="not supported"):
        Config.model_validate({"data": {"entry_interval": "3m"}})


def test_both_sides_disabled_is_rejected():
    with pytest.raises(ValidationError, match="cannot both be false"):
        Config.model_validate({"strategy": {"allow_long": False, "allow_short": False}})


def test_ema_fast_must_be_shorter_than_trend():
    with pytest.raises(ValidationError, match="shorter than"):
        Config.model_validate({"strategy": {"ema_fast": 50, "ema_trend": 10}})


def test_unknown_keys_are_rejected():
    with pytest.raises(ValidationError):
        Config.model_validate({"strategey": {"ema_fast": 5}})


def test_overrides_are_type_coerced():
    data = apply_overrides({}, ["strategy.volume_multiplier=3.0", "report.plot=false"])
    assert data["strategy"]["volume_multiplier"] == 3.0
    assert data["report"]["plot"] is False


def test_fingerprint_is_stable_and_sensitive():
    a = Config.model_validate({})
    b = Config.model_validate({})
    c = Config.model_validate({"strategy": {"volume_multiplier": 3.0}})
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


def test_layered_config_deep_merges(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("strategy:\n  ema_fast: 10\n  volume_multiplier: 1.5\n")
    variant = tmp_path / "v.yaml"
    variant.write_text("strategy:\n  volume_multiplier: 3.0\n")

    cfg = load_config(variant, base=base)
    assert cfg.strategy.ema_fast == 10        # inherited
    assert cfg.strategy.volume_multiplier == 3.0  # overridden


def test_defaults_are_the_faithful_port():
    cfg = Config.model_validate({})
    assert (cfg.strategy.ema_fast, cfg.strategy.ema_slow, cfg.strategy.ema_trend) == (10, 20, 50)
    assert cfg.strategy.volume_multiplier == 1.5   # code, not the volMA90 comment
    assert cfg.strategy.long_retracement == 0.40
    assert cfg.strategy.short_retracement == 0.60  # asymmetry preserved
