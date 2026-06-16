import numpy as np, mujoco
from t1_wbc.model import load_t1_model, build_index_maps
from t1_wbc.config import WBCConfig
from t1_wbc.controller import WBController
from t1_wbc.dynamics import CpuDynamics
from t1_wbc.estimator import StateEstimator
from t1_wbc.transport import SimTransport

def test_assembled_state_reproduces_dynamics():
    cfg = WBCConfig(); model, data = load_t1_model(cfg.xml)
    mujoco.mj_resetDataKeyframe(model, data, 0); mujoco.mj_forward(model, data)
    maps = build_index_maps(model)
    direct = CpuDynamics(model, maps).extract(data)            # ground-truth dynamics

    ctrl = WBController(model, cfg)
    est = StateEstimator(model, maps); ctrl.attach_estimator(est)
    ls = SimTransport(model, data).read_lowstate()
    # drive estimator, then assemble+extract, copying TRUE base twist to isolate the POSE path
    # (lin-vel estimation quality is covered by the in-the-loop test, not here):
    d_est = ctrl._assemble_est_dynamics(ls, base_twist=data.qvel[0:6].copy())
    for k in ("M", "com", "Jfoot_L", "Jfoot_R", "Jcom"):
        np.testing.assert_allclose(d_est[k], direct[k], atol=1e-3, err_msg=k)
    np.testing.assert_allclose(d_est["h"], direct["h"], atol=1e-2)
