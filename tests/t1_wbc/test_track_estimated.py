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
