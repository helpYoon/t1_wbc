from t1_wbc.run import run_track_estimated_safe
from t1_wbc.config import WBCConfig

def test_safety_wrapped_loop_stays_upright():
    cfg = WBCConfig()                       # nonzero servo gains applied by SafetyLayer
    out = run_track_estimated_safe(cfg, seconds=5.0)
    assert out["upright"] is True
    assert out["infeasible"] == 0
    assert out["min_base_z"] > 0.55
    assert out["lh_rms"] < 0.05 and out["rh_rms"] < 0.05   # looser: ramp + servo PD perturb early ticks
