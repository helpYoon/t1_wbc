"""C1 (decoupled publish/solve) + C2 (real-pose seed) for the hardware tracking path.

Threads are NOT spawned here — the publish/control loops take an injectable stop()/clock()/
sleep() so the decoupling and pacing logic is tested deterministically single-threaded."""
import numpy as np
import mujoco
from t1_wbc.model import load_t1_model, build_index_maps
from t1_wbc.config import WBCConfig
from t1_wbc.run import (_CmdBox, _publish_loop, _track_control_loop,
                        _build_track_controller, build_hold_cmd)
from t1_wbc.transport import LowState
from t1_wbc._assets import asset

H50 = asset("motion_plan_h50cm.pkl")


class MockTransport:
    """Returns a steady LowState at pose `q`; records every published command."""
    def __init__(self, model, q=None):
        if q is None:
            d = mujoco.MjData(model); mujoco.mj_resetDataKeyframe(model, d, 0)
            q = d.qpos[7:7 + model.nu].copy()
        self.q = q; self.nu = model.nu; self.written = []
    def read_lowstate(self):
        return LowState(imu_rpy=np.zeros(3), imu_gyro=np.zeros(3), imu_acc=np.array([0.0, 0.0, 9.81]),
                        joint_q=self.q.copy(), joint_dq=np.zeros(self.nu), odom_xytheta=np.zeros(3))
    def write_lowcmd(self, cmd): self.written.append(cmd)
    def state_age(self): return 0.0


def _stop_after(n):
    c = {"i": 0}
    def stop():
        c["i"] += 1
        return c["i"] > n
    return stop


def _cfg():
    return WBCConfig(motion=H50, segments=(0, 1))


# ---- C1: publish loop -------------------------------------------------------

def test_publish_loop_streams_box_every_tick_paced():
    model, _ = load_t1_model(); maps = build_index_maps(model)
    tr = MockTransport(model)
    hold = build_hold_cmd(model, maps, _cfg(), tr.q)
    box = _CmdBox(hold)
    clk = {"t": 0.0}; sleeps = []
    def slp(s): sleeps.append(s); clk["t"] += max(s, 0.0)
    n = _publish_loop(tr, box, 0.005, _stop_after(5), clock=lambda: clk["t"], sleep=slp)
    assert n == 5 and len(tr.written) == 5
    assert all(w is hold for w in tr.written)
    assert abs(sum(sleeps) - 5 * 0.005) < 1e-9


def test_publish_loop_keeps_streaming_when_control_stalls():
    # C1 GUARANTEE: if the control loop never updates the box (slow/blocked solve), the
    # publish loop still emits a command every tick — the firmware stream never gaps.
    model, _ = load_t1_model(); maps = build_index_maps(model)
    tr = MockTransport(model)
    hold = build_hold_cmd(model, maps, _cfg(), tr.q)
    box = _CmdBox(hold)
    n = _publish_loop(tr, box, 0.005, _stop_after(8), clock=lambda: 0.0, sleep=lambda s: None)
    assert n == 8 and len(tr.written) == 8
    assert all(w is hold for w in tr.written)   # last-good cmd kept alive, no gap


def test_cmdbox_get_set_roundtrip():
    box = _CmdBox("a")
    assert box.get() == "a"
    box.set("b")
    assert box.get() == "b"


# ---- C1: control loop -------------------------------------------------------

def test_track_control_loop_updates_box_each_tick():
    cfg = _cfg()
    model, data = load_t1_model(cfg.xml); maps = build_index_maps(model)
    tr = MockTransport(model)
    ctrl, safety = _build_track_controller(cfg, model, data, maps, tr.q)
    box = _CmdBox(build_hold_cmd(model, maps, cfg, tr.q))
    res = _track_control_loop(cfg, ctrl, safety, tr, box, ticks=4, stop=lambda: False,
                              clock=lambda: 0.0, sleep=lambda s: None)
    assert res["solved"] == 4 and res["faults"] == 0
    assert box.get().q_des.shape == (model.nu,)


def test_track_control_loop_holds_last_good_on_read_fault():
    # A throwing read_lowstate (e.g. a use-after-free MemoryError) must NOT propagate out of
    # the loop — that would hit run_hw's finally -> kDamping -> drop the engaged robot.
    cfg = _cfg()
    model, data = load_t1_model(cfg.xml); maps = build_index_maps(model)
    ctrl, safety = _build_track_controller(cfg, model, data, maps, MockTransport(model).q)
    box = _CmdBox(build_hold_cmd(model, maps, cfg, MockTransport(model).q))
    initial = box.get()
    class BadRead:
        def read_lowstate(self): raise MemoryError("Could not allocate list object!")
        def write_lowcmd(self, c): pass
        def state_age(self): return 0.0
    res = _track_control_loop(cfg, ctrl, safety, BadRead(), box, ticks=3, stop=lambda: False,
                              clock=lambda: 0.0, sleep=lambda s: None)
    assert res["faults"] == 3 and res["solved"] == 0
    assert box.get() is initial   # last-good cmd retained -> publish stream never gaps


def test_track_control_loop_holds_last_good_on_solve_fault():
    # C1 fault-safety: a throwing solve must NOT propagate (which would kill the publish
    # thread / hard-damp). The box keeps its last-good cmd so the stream stays alive.
    cfg = _cfg()
    model, data = load_t1_model(cfg.xml); maps = build_index_maps(model)
    tr = MockTransport(model)
    ctrl, safety = _build_track_controller(cfg, model, data, maps, tr.q)
    box = _CmdBox(build_hold_cmd(model, maps, cfg, tr.q))
    initial = box.get()
    def boom(ls, t): raise RuntimeError("solve blew up")
    ctrl.step_track_estimated = boom
    res = _track_control_loop(cfg, ctrl, safety, tr, box, ticks=3, stop=lambda: False,
                              clock=lambda: 0.0, sleep=lambda s: None)
    assert res["faults"] == 3 and res["solved"] == 0
    assert box.get() is initial   # last-good cmd retained -> publish stream never gaps


# ---- C2: real-pose seed -----------------------------------------------------

def test_build_track_controller_seeds_from_real_pose_not_keyframe():
    cfg = _cfg()
    model, data = load_t1_model(cfg.xml); maps = build_index_maps(model)
    d = mujoco.MjData(model); mujoco.mj_resetDataKeyframe(model, d, 0)
    home = d.qpos[7:7 + model.nu].copy()
    q0 = home.copy(); q0[0] += 0.3            # measured pose differs from the keyframe
    ctrl, safety = _build_track_controller(cfg, model, data, maps, q0)
    np.testing.assert_allclose(ctrl.q_home, q0)        # posture/home seeded from real pose
    np.testing.assert_allclose(safety._hold_q, q0)     # ramp STARTS from real pose -> no lurch
    assert not np.allclose(ctrl.q_home, home)          # NOT the sim keyframe
