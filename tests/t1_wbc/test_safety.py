import numpy as np
from t1_wbc.model import load_t1_model, build_index_maps
from t1_wbc.config import WBCConfig, servo_gains_for
from t1_wbc.controller import WBController
from t1_wbc.safety import clamp_torque, slew_limit

def test_servo_gains_table_per_group():
    model, _ = load_t1_model()
    maps = build_index_maps(model)
    kp, kd = servo_gains_for(maps)            # (nu,), (nu,)
    n2i = maps["name_to_act_index"]
    assert kp[n2i["Left_Shoulder_Pitch"]] == 20.0 and kd[n2i["Left_Shoulder_Pitch"]] == 0.5
    assert kp[n2i["Left_Hip_Pitch"]] == 200.0 and kd[n2i["Left_Hip_Pitch"]] == 5.0
    assert kp[n2i["Left_Ankle_Pitch"]] == 50.0 and kd[n2i["Left_Ankle_Pitch"]] == 3.0

def test_torque_limit_scale_tightens_qp_ctrlrange():
    model, data = load_t1_model()
    full = WBController(model, WBCConfig(torque_limit_scale=1.0, tau_pd_margin=0.0))
    half = WBController(model, WBCConfig(torque_limit_scale=0.5, tau_pd_margin=0.0))
    np.testing.assert_allclose(half.ctrlrange[:, 1], 0.5 * full.ctrlrange[:, 1])
    np.testing.assert_allclose(half.ctrlrange[:, 0], 0.5 * full.ctrlrange[:, 0])

def test_pd_margin_shrinks_limits_symmetrically():
    model, _ = load_t1_model()
    base = WBController(model, WBCConfig(torque_limit_scale=1.0, tau_pd_margin=0.0)).ctrlrange.copy()
    m = WBController(model, WBCConfig(torque_limit_scale=1.0, tau_pd_margin=2.0)).ctrlrange
    np.testing.assert_allclose(m[:, 1], base[:, 1] - 2.0)
    np.testing.assert_allclose(m[:, 0], base[:, 0] + 2.0)

def test_clamp_torque_bounds_per_joint():
    tau = np.array([100.0, -100.0, 5.0])
    lo = np.array([-30.0, -30.0, -30.0]); hi = np.array([30.0, 30.0, 30.0])
    np.testing.assert_allclose(clamp_torque(tau, lo, hi), [30.0, -30.0, 5.0])

def test_slew_limit_caps_delta():
    prev = np.array([0.0, 0.0])
    tau = np.array([200.0, -5.0])
    out = slew_limit(tau, prev, max_delta=10.0)
    np.testing.assert_allclose(out, [10.0, -5.0])   # first capped to +10, second within
