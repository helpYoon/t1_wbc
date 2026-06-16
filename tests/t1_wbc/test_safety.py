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

from t1_wbc.safety import SafetyLayer
from t1_wbc.transport import LowCmd

def _sl(model, cfg, maps):
    return SafetyLayer(model, cfg, maps)

def test_hold_when_infeasible_zeros_tau_ff():
    model, _ = load_t1_model(); maps = build_index_maps(model); cfg = WBCConfig()
    sl = _sl(model, cfg, maps); hold_q = np.zeros(model.nu); sl.begin(hold_q)
    raw = LowCmd(q_des=np.ones(model.nu), qd_des=np.zeros(model.nu),
                 kp=np.zeros(model.nu), kd=np.zeros(model.nu), tau_ff=np.full(model.nu, 50.0))
    out = sl.wrap(raw, ok=False, t=10.0, lowstate_age=0.0)   # past ramp, but infeasible
    np.testing.assert_allclose(out.tau_ff, 0.0)               # hold: no feedforward torque
    np.testing.assert_allclose(out.q_des, hold_q)             # PD to the hold pose
    assert np.all(out.kp > 0)                                 # servo gains engaged

def test_watchdog_stale_state_holds():
    model, _ = load_t1_model(); maps = build_index_maps(model); cfg = WBCConfig()
    sl = _sl(model, cfg, maps); sl.begin(np.zeros(model.nu))
    raw = LowCmd(q_des=np.ones(model.nu), qd_des=np.zeros(model.nu),
                 kp=np.zeros(model.nu), kd=np.zeros(model.nu), tau_ff=np.full(model.nu, 50.0))
    out = sl.wrap(raw, ok=True, t=10.0, lowstate_age=0.2)     # stale beyond watchdog_timeout_s
    np.testing.assert_allclose(out.tau_ff, 0.0)

def test_weight_ramp_blends_in_tau_ff():
    model, _ = load_t1_model(); maps = build_index_maps(model)
    cfg = WBCConfig(ramp_seconds=2.0)
    sl = _sl(model, cfg, maps); sl.begin(np.zeros(model.nu))
    # tau_ff=5.0 stays within every actuator limit (min |limit| is 7 Nm) so the clamp
    # does not bind and the test exercises only the ramp blend + slew, not the clamp.
    raw = LowCmd(q_des=np.zeros(model.nu), qd_des=np.zeros(model.nu),
                 kp=np.zeros(model.nu), kd=np.zeros(model.nu), tau_ff=np.full(model.nu, 5.0))
    half = sl.wrap(raw, ok=True, t=1.0, lowstate_age=0.0)     # 50% through the ramp
    assert np.all(half.tau_ff < raw.tau_ff) and np.all(half.tau_ff > 0)
    full = sl.wrap(raw, ok=True, t=5.0, lowstate_age=0.0)     # past ramp
    np.testing.assert_allclose(full.tau_ff, raw.tau_ff)
