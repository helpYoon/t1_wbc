import pytest, mujoco
from t1_wbc.model import load_t1_model, build_index_maps
from t1_wbc.config import WBCConfig
from t1_wbc.controller import WBController
from t1_wbc.reference import ReferenceTrajectory

@pytest.fixture
def tracking_controller():
    cfg = WBCConfig(); model, data = load_t1_model(cfg.xml)
    ctrl = WBController(model, cfg); ctrl.reset(data); ctrl.settle(data)
    ref = ReferenceTrajectory(model, build_index_maps(model), ctrl.q_home, cfg, 0.0, 0.0, 0.0)
    ctrl.attach_reference(ref)
    return cfg, model, data, ctrl
