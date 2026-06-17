"""CLI plumbing for run.py: --motion / --segments / --torque-scale -> WBCConfig."""
import pytest
from t1_wbc.run import _parse_segments, _build_cfg, _parser


def test_parse_segments():
    assert _parse_segments("0,1") == (0, 1)
    assert _parse_segments("2") == (2,)
    assert _parse_segments("0, 1 ,2") == (0, 1, 2)
    assert _parse_segments(None) is None


def test_build_cfg_applies_motion_segments_and_torque():
    args = _parser().parse_args(
        ["--mode", "hw", "--motion", "/x/y.pkl", "--segments", "0,1",
         "--torque-scale", "0.5", "--control-period", "0.04"])
    cfg = _build_cfg(args)
    assert cfg.motion == "/x/y.pkl"
    assert cfg.segments == (0, 1)
    assert cfg.torque_limit_scale == 0.5
    assert cfg.control_period == 0.04


def test_build_cfg_defaults_are_unchanged():
    args = _parser().parse_args(["--mode", "track"])
    cfg = _build_cfg(args)
    assert cfg.segments is None        # full motion by default
    assert cfg.torque_limit_scale == 1.0
