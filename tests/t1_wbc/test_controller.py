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
