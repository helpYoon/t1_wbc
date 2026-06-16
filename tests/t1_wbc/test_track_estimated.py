import numpy as np, mujoco
from t1_wbc.model import load_t1_model, build_index_maps
from t1_wbc.config import WBCConfig
from t1_wbc.controller import WBController
from t1_wbc.reference import ReferenceTrajectory
from t1_wbc.estimator import StateEstimator
from t1_wbc.transport import SimTransport

def test_one_estimated_track_tick():
    cfg = WBCConfig(); model, data = load_t1_model(cfg.xml)
    ctrl = WBController(model, cfg); ctrl.reset(data); ctrl.settle(data)
    maps = build_index_maps(model)
    ctrl.attach_reference(ReferenceTrajectory(model, maps, ctrl.q_home, cfg, 0.0, 0.0, 0.0))
    ctrl.attach_estimator(StateEstimator(model, maps))
    tr = SimTransport(model, data)
    cmd, diag = ctrl.step_track_estimated(tr.read_lowstate(), 0.0)
    assert isinstance(cmd.tau_ff, np.ndarray) and cmd.tau_ff.shape == (model.nu,)
    assert diag["ok"] is True


from t1_wbc.run import run_track_estimated, run_track

def test_estimated_track_stays_upright():
    cfg = WBCConfig()
    out = run_track_estimated(cfg, seconds=5.0)
    assert out["upright"] is True
    assert out["infeasible"] == 0
    assert out["min_base_z"] > 0.55
    base = run_track(cfg, seconds=5.0)
    # hand tracking on estimated state within 1.5x of the ground-truth baseline
    assert out["lh_rms"] < 1.5 * base["lh_rms"] + 0.005
    assert out["rh_rms"] < 1.5 * base["rh_rms"] + 0.005
