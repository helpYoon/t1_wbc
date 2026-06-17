"""hw-hold: pure PD-hold of a snapshotted pose (build_hold_cmd) streamed real-time (run_hold_loop)."""
import numpy as np
from t1_wbc.model import load_t1_model, build_index_maps
from t1_wbc.config import WBCConfig, servo_gains_for
from t1_wbc.run import build_hold_cmd, run_hold_loop


class _Mock:
    def __init__(self): self.written = []
    def write_lowcmd(self, cmd): self.written.append(cmd)
    def state_age(self): return 0.0


def test_build_hold_cmd_is_pure_pd_hold_with_task_info_gains():
    model, _ = load_t1_model(); maps = build_index_maps(model); cfg = WBCConfig()
    q = np.linspace(-0.3, 0.3, model.nu)
    cmd = build_hold_cmd(model, maps, cfg, q)
    np.testing.assert_allclose(cmd.q_des, q)          # holds the snapshotted pose
    np.testing.assert_allclose(cmd.qd_des, 0.0)
    np.testing.assert_allclose(cmd.tau_ff, 0.0)       # NO feedforward — pure PD
    kp, kd = servo_gains_for(maps)                    # task.info gains (20/0.5, 200/5, 50/3)
    np.testing.assert_allclose(cmd.kp, kp)
    np.testing.assert_allclose(cmd.kd, kd)


def test_run_hold_loop_streams_constant_cmd_paced():
    model, _ = load_t1_model(); maps = build_index_maps(model)
    cfg = WBCConfig(); cfg.control_period = 0.02
    hold = build_hold_cmd(model, maps, cfg, np.zeros(model.nu))
    tr = _Mock()
    clk = {"t": 0.0}; sleeps = []
    def fake_sleep(s): sleeps.append(s); clk["t"] += max(s, 0.0)
    res = run_hold_loop(cfg, hold, tr, ticks=4, clock=lambda: clk["t"], sleep=fake_sleep)
    assert res["n"] == 4 and res["overruns"] == 0
    assert len(tr.written) == 4
    assert all(w is hold for w in tr.written)          # same constant command every tick
    assert abs(sum(sleeps) - 4 * cfg.control_period) < 1e-9


def test_run_hold_loop_counts_overruns():
    model, _ = load_t1_model(); maps = build_index_maps(model)
    cfg = WBCConfig(); cfg.control_period = 0.02
    hold = build_hold_cmd(model, maps, cfg, np.zeros(model.nu))
    tr = _Mock()
    state = {"t": 0.0}
    def clock(): v = state["t"]; state["t"] += 0.05; return v
    sleeps = []
    res = run_hold_loop(cfg, hold, tr, ticks=3, clock=clock, sleep=lambda s: sleeps.append(s))
    assert res["n"] == 3 and res["overruns"] == 3 and sleeps == []
