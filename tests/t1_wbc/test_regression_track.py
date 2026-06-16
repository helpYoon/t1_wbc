from t1_wbc.run import run_track
from t1_wbc.config import WBCConfig

def test_track_reproduces_baseline():
    out = run_track(WBCConfig(), seconds=5.0)        # headless, ~2500 ticks
    assert out["upright"] is True
    assert out["infeasible"] == 0
    assert out["min_base_z"] > 0.55                  # leans to ~0.58-0.64, never falls
    assert out["lh_rms"] < 0.03 and out["rh_rms"] < 0.03   # ~1-2 cm
