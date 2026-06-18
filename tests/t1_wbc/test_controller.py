import numpy as np
def test_one_track_tick_is_numpy(tracking_controller):
    cfg, model, data, ctrl = tracking_controller
    cmd, diag = ctrl.step_track(data, 0.0)
    assert isinstance(cmd.tau_ff, np.ndarray) and cmd.tau_ff.shape == (model.nu,)
    assert isinstance(cmd.q_des, np.ndarray) and cmd.q_des.shape == (model.nu,)
    assert diag["ok"] is True
    assert "torch" not in __import__("sys").modules

def test_one_balance_tick(tracking_controller):
    cfg, model, data, ctrl = tracking_controller
    cmd, diag = ctrl.step_balance(data)
    assert isinstance(cmd.tau_ff, np.ndarray) and diag["ok"] is True

def test_base_twist_filter_emas(tracking_controller):
    # The estimator base twist must be EMA-smoothed before the QP (it drives the highest-weight
    # CoM/base-ori tasks; raw it injected sensor noise into leg tau_ff). alpha=1.0 disables.
    cfg, model, data, ctrl = tracking_controller
    ctrl.cfg.base_vel_filter_alpha = 0.5
    ctrl._base_twist_filt = None
    np.testing.assert_allclose(ctrl._filter_base_twist(np.ones(6)), np.ones(6))   # seed
    np.testing.assert_allclose(ctrl._filter_base_twist(np.zeros(6)), 0.5 * np.ones(6))  # EMA step
    ctrl.cfg.base_vel_filter_alpha = 1.0
    ctrl._base_twist_filt = None
    np.testing.assert_allclose(ctrl._filter_base_twist(np.full(6, 7.0)), np.full(6, 7.0))  # off

def test_cmd_tracks_reference_not_measured_state(tracking_controller):
    # q_des/qd_des must be the REFERENCE (computed-torque + PD form). The old measured-state +
    # integrated-accel form cancelled the firmware kd damping and added velocity anti-damping
    # -> whole-body buzz on hardware.
    cfg, model, data, ctrl = tracking_controller
    t = 0.5
    rs = ctrl.ref.sample(t)
    cmd, _ = ctrl.step_track(data, t)
    np.testing.assert_allclose(cmd.q_des, np.asarray(rs.q_ref).reshape(-1))
    np.testing.assert_allclose(cmd.qd_des, np.asarray(rs.qd_ref).reshape(-1))
