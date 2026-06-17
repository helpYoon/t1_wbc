"""run_hw_loop real-time pacing: fixed control period, overrun accounting.
Timing is deterministic via injected clock/sleep (no real wall-clock waiting)."""
import numpy as np
import mujoco
from t1_wbc.model import load_t1_model, build_index_maps
from t1_wbc.config import WBCConfig
from t1_wbc.run import run_hw_loop
from t1_wbc.transport import LowState
from t1_wbc._assets import asset

H50 = asset("motion_plan_h50cm.pkl")


class MockTransport:
    """Returns a steady home-pose LowState every tick; records written commands."""
    def __init__(self, model):
        d = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, d, 0)
        self.q = d.qpos[7:7 + model.nu].copy()
        self.nu = model.nu
        self.written = []

    def read_lowstate(self):
        return LowState(imu_rpy=np.zeros(3), imu_gyro=np.zeros(3),
                        imu_acc=np.array([0.0, 0.0, 9.81]),
                        joint_q=self.q.copy(), joint_dq=np.zeros(self.nu),
                        odom_xytheta=np.zeros(3))

    def write_lowcmd(self, cmd):
        self.written.append(cmd)

    def state_age(self):
        return 0.0


def _fixtures():
    model, data = load_t1_model()
    return model, data, build_index_maps(model)


def test_run_hw_loop_paces_at_control_period():
    model, data, maps = _fixtures()
    tr = MockTransport(model)
    cfg = WBCConfig(motion=H50, segments=(0, 1))
    cfg.control_period = 0.02
    clk = {"t": 0.0}
    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)
        clk["t"] += max(s, 0.0)

    res = run_hw_loop(cfg, model, data, maps, tr, ticks=5,
                      clock=lambda: clk["t"], sleep=fake_sleep)
    assert res["n"] == 5
    assert res["overruns"] == 0
    assert len(tr.written) == 5
    assert len(sleeps) == 5
    assert abs(sum(sleeps) - 5 * cfg.control_period) < 1e-9


def test_run_hw_loop_counts_overruns_and_never_sleeps_negative():
    model, data, maps = _fixtures()
    tr = MockTransport(model)
    cfg = WBCConfig(motion=H50, segments=(0, 1))
    cfg.control_period = 0.02
    state = {"t": 0.0}

    def clock():  # advances 0.05s (> period) per call -> every tick overruns
        v = state["t"]
        state["t"] += 0.05
        return v

    sleeps = []
    res = run_hw_loop(cfg, model, data, maps, tr, ticks=3,
                      clock=clock, sleep=lambda s: sleeps.append(s))
    assert res["n"] == 3
    assert res["overruns"] == 3
    assert sleeps == []


def test_run_hw_loop_horizon_derives_from_control_period():
    # With ticks=None the horizon spans the selected segments at the control period,
    # NOT the 500Hz physics step -> ~ duration / control_period ticks.
    model, data, maps = _fixtures()
    tr = MockTransport(model)
    cfg = WBCConfig(motion=H50, segments=(0, 1))
    cfg.control_period = 0.02
    res = run_hw_loop(cfg, model, data, maps, tr, ticks=None,
                      clock=lambda: 0.0, sleep=lambda s: None)
    # seg0+1 ~2.094s ref-time * time_scale(5) ~= 10.47s wall / 0.02 ~= 520 ticks
    assert 400 < res["n"] < 700
