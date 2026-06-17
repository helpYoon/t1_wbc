"""Sim + control loop and CLI for the T1 whole-body QP controller."""
import argparse
import threading
import time
import numpy as np
import mujoco
from .model import load_t1_model, build_index_maps
from .config import WBCConfig
from .controller import WBController
from .action_backend import MuJoCoBackend
from .reference import ReferenceTrajectory


def run_balance(cfg, seconds=3.0):
    """Phase-1 entry: settle then hold-home/balance headless. Returns summary dict."""
    model, data = load_t1_model(cfg.xml)
    ctrl = WBController(model, cfg); backend = MuJoCoBackend(model)
    ctrl.reset(data)
    ncon = ctrl.settle(data)
    infeas = 0; zmin = 1e9
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step1(model, data)
        cmd, diag = ctrl.step_balance(data)
        infeas += int(not diag["ok"]); zmin = min(zmin, diag["base_z"])
        backend.apply(cmd, data)
        mujoco.mj_step2(model, data)
    return dict(ncon=ncon, infeasible=infeas, min_base_z=zmin, upright=zmin > cfg.upright_z)


def run_track(cfg, seconds=None, viewer=False, log=None):
    """Settle then track the motion. If seconds is None, run the full motion. Returns summary.

    If `log` is a path, write a per-tracked-tick CSV (off by default).
    """
    model, data = load_t1_model(cfg.xml)
    ctrl = WBController(model, cfg); backend = MuJoCoBackend(model)
    ctrl.reset(data); ncon = ctrl.settle(data)
    ref = ReferenceTrajectory(model, build_index_maps(model), ctrl.q_home, cfg, 0.0, 0.0, 0.0)
    ctrl.attach_reference(ref)
    horizon = ref.duration if seconds is None else seconds
    dt = model.opt.timestep; t = 0.0; infeas = 0; zmin = 1e9; lh = []; rh = []
    handle = None
    if viewer:
        # import the submodule without rebinding the local `mujoco` name
        # (a bare `import mujoco.viewer` here would shadow the module-level
        # `mujoco` for this whole function and break mj_step1/mj_step2 calls)
        import importlib
        _viewer = importlib.import_module("mujoco.viewer")
        handle = _viewer.launch_passive(model, data)
    logger = None
    if log is not None:
        from .logging_utils import TickLogger
        logger = TickLogger(log, ["t", "base_z", "lh_err", "rh_err", "max_tau", "qp_ok"])
    n = int(horizon / dt)
    for i in range(n):
        mujoco.mj_step1(model, data)
        if i % cfg.control_decimation == 0:
            cmd, diag = ctrl.step_track(data, t)
            infeas += int(not diag["ok"]); zmin = min(zmin, diag["base_z"])
            lh.append(diag["lh_err"]); rh.append(diag["rh_err"])
            if logger is not None:
                logger.log(dict(t=t, base_z=diag["base_z"], lh_err=diag["lh_err"],
                                rh_err=diag["rh_err"], max_tau=diag["max_tau"],
                                qp_ok=int(diag["ok"])))
        backend.apply(ctrl._last, data)
        mujoco.mj_step2(model, data)
        if handle is not None:
            handle.sync()
        t += dt
    if handle is not None:
        handle.close()
    if logger is not None:
        logger.close()
    return dict(ncon=ncon, infeasible=infeas, min_base_z=zmin, upright=zmin > cfg.upright_z,
                lh_rms=float(np.mean(lh)), rh_rms=float(np.mean(rh)))


def _run_estimated(cfg, seconds, with_safety):
    """Settle, then track the motion with the WBC on ESTIMATED base state (IMU+odom+encoders
    via SimTransport + StateEstimator). With `with_safety`, the command is routed through the
    SafetyLayer (servo gains, weight-ramp, clamps, slew, infeasible->hold) — the on-robot
    command path, run in sim. Returns a summary dict."""
    from .transport import SimTransport
    from .estimator import StateEstimator
    model, data = load_t1_model(cfg.xml)
    ctrl = WBController(model, cfg); ctrl.reset(data); ncon = ctrl.settle(data)
    maps = build_index_maps(model)
    ctrl.attach_reference(ReferenceTrajectory(model, maps, ctrl.q_home, cfg, 0.0, 0.0, 0.0))
    ctrl.attach_estimator(StateEstimator(model, maps))
    tr = SimTransport(model, data)
    safety = None
    if with_safety:
        from .safety import SafetyLayer
        safety = SafetyLayer(model, cfg, maps); safety.begin(ctrl.q_home)
    horizon = ctrl.ref.duration if seconds is None else seconds
    dt = model.opt.timestep; t = 0.0; infeas = 0; zmin = 1e9; lh = []; rh = []; last = None
    for i in range(int(horizon / dt)):
        mujoco.mj_step1(model, data)
        if i % cfg.control_decimation == 0:
            cmd, diag = ctrl.step_track_estimated(tr.read_lowstate(), t)
            last = safety.wrap(cmd, ok=diag["ok"], t=t, lowstate_age=0.0) if safety else cmd
            infeas += int(not diag["ok"]); zmin = min(zmin, float(data.qpos[2]))
            lh.append(diag["lh_err"]); rh.append(diag["rh_err"])
        tr.write_lowcmd(last)
        mujoco.mj_step2(model, data)
        t += dt
    return dict(ncon=ncon, infeasible=infeas, min_base_z=zmin, upright=zmin > cfg.upright_z,
                lh_rms=float(np.mean(lh)), rh_rms=float(np.mean(rh)))


def run_track_estimated(cfg, seconds=None):
    """Track the motion on ESTIMATED base state (no safety layer)."""
    return _run_estimated(cfg, seconds, with_safety=False)


def run_track_estimated_safe(cfg, seconds=None):
    """Estimated-state track loop wrapped by the SafetyLayer (the on-robot path, in sim)."""
    return _run_estimated(cfg, seconds, with_safety=True)


def run_hw_loop(cfg, model, data, maps, transport, ticks=None,
               clock=time.monotonic, sleep=time.sleep):
    """The hardware control loop: read LowState -> estimate+WBC -> SafetyLayer -> write LowCmd.
    Transport-agnostic (real SdkTransport on robot; mock in tests).

    Paced in REAL TIME at `cfg.control_period` (default 50Hz): after each tick the loop
    sleeps until the next period boundary so the robot tracks at the intended rate rather
    than at CPU speed. Ticks whose solve+write overran the period are counted (not slept).
    `clock`/`sleep` are injectable for deterministic timing tests.

    `data` is only used to settle / seed q_home.
    Returns {"n": commands_written, "overruns": ticks_that_missed_their_deadline}.
    """
    from .estimator import StateEstimator
    from .safety import SafetyLayer
    ctrl = WBController(model, cfg); ctrl.reset(data); ctrl.settle(data)
    ctrl.attach_reference(ReferenceTrajectory(model, maps, ctrl.q_home, cfg, 0.0, 0.0, 0.0))
    ctrl.attach_estimator(StateEstimator(model, maps))
    safety = SafetyLayer(model, cfg, maps); safety.begin(ctrl.q_home)
    period = cfg.control_period; t = 0.0; n = 0; overruns = 0
    horizon_ticks = ticks if ticks is not None else int(ctrl.ref.duration / period)
    start = clock()
    for i in range(horizon_ticks):
        ls = transport.read_lowstate()
        cmd, diag = ctrl.step_track_estimated(ls, t)
        safe = safety.wrap(cmd, ok=diag["ok"], t=t, lowstate_age=transport.state_age())
        transport.write_lowcmd(safe); n += 1; t += period
        delay = (start + (i + 1) * period) - clock()   # time left until this tick's deadline
        if delay > 0:
            sleep(delay)
        else:
            overruns += 1
    return {"n": n, "overruns": overruns}


def build_hold_cmd(model, maps, cfg, q_hold):
    """A pure PD hold at q_hold: q_des=q_hold, qd_des=0, tau_ff=0, kp/kd = task.info servo
    gains. This is the SafetyLayer's hold path — the on-robot 'hold this pose' command, and
    the initial LowCmd queued before kCustom."""
    from .safety import SafetyLayer
    from .action_backend import JointCommand
    q_hold = np.asarray(q_hold, dtype=np.float64).reshape(-1)
    safety = SafetyLayer(model, cfg, maps); safety.begin(q_hold)
    z = np.zeros(model.nu)
    raw = JointCommand(q_des=q_hold.copy(), qd_des=z, kp=z, kd=z, tau_ff=z)
    return safety.wrap(raw, ok=False, t=0.0, lowstate_age=0.0)   # ok=False -> hold path


def run_hold_loop(cfg, hold_cmd, transport, ticks=None,
                  clock=time.monotonic, sleep=time.sleep):
    """Stream a constant PD-hold command at cfg.control_period (real-time paced, overrun-aware).
    ticks=None runs until KeyboardInterrupt — the operator aborts when satisfied.
    Returns {"n": commands_written, "overruns": ticks_that_missed_their_deadline}."""
    period = cfg.control_period; n = 0; overruns = 0; start = clock(); i = 0
    while ticks is None or i < ticks:
        transport.write_lowcmd(hold_cmd); n += 1
        delay = (start + (i + 1) * period) - clock()
        if delay > 0:
            sleep(delay)
        else:
            overruns += 1
        i += 1
    return {"n": n, "overruns": overruns}


def _snapshot_pose(tr, timeout=3.0):
    """Wait for a FULLY-populated LowState (the first DDS samples can be short/empty), then
    return the current measured joint positions (nu,). Snapshotting a partial state would
    hold missing joints at zero — so we require a complete one."""
    if not tr._await_valid_state(timeout, time.sleep, time.monotonic):
        raise RuntimeError(f"no complete LowState ({tr.n} joints) within {timeout}s; "
                           "robot absent or not a 7-DOF-arm unit")
    return tr.read_lowstate().joint_q.copy()


class _CmdBox:
    """Thread-shared latest safe LowCmd. A single writer (control loop) and single reader
    (publish loop) only swap an object reference, which is atomic under the GIL — no lock."""
    def __init__(self, cmd): self._cmd = cmd
    def get(self): return self._cmd
    def set(self, cmd): self._cmd = cmd


def _publish_loop(transport, box, period, stop, clock=time.monotonic, sleep=time.sleep):
    """Stream box.get() at a fixed `period` until stop() — independent of solve time, so a
    slow / blocked / throwing control loop can NEVER gap the firmware command stream (C1).
    Returns the number of commands published. `stop`/`clock`/`sleep` are injectable for tests."""
    start = clock(); i = 0
    while not stop():
        transport.write_lowcmd(box.get()); i += 1
        delay = start + i * period - clock()
        if delay > 0:
            sleep(delay)
    return i


def _track_control_loop(cfg, ctrl, safety, transport, box, ticks=None, stop=lambda: False,
                        clock=time.monotonic, sleep=time.sleep):
    """Solve the WBC at cfg.control_period and push each safe cmd into `box` for the publish
    loop to stream. A throwing solve is caught and counted; the box keeps its last-good cmd, so
    the publish stream stays alive and the robot never hard-faults from an exception (C1).
    Returns {"solved", "faults", "overruns"}."""
    period = cfg.control_period; t = 0.0; i = 0
    solved = 0; faults = 0; overruns = 0
    start = clock()
    while (ticks is None or i < ticks) and not stop():
        try:
            ls = transport.read_lowstate()    # read INSIDE the guard: a read fault must not
            cmd, diag = ctrl.step_track_estimated(ls, t)   # propagate -> finally stop() -> drop
            safe = safety.wrap(cmd, ok=diag["ok"], t=t, lowstate_age=transport.state_age())
            box.set(safe); solved += 1
        except Exception:
            faults += 1                       # keep last-good cmd; publish loop never gaps
        i += 1; t += period
        delay = start + i * period - clock()
        if delay > 0:
            sleep(delay)
        else:
            overruns += 1
    return {"solved": solved, "faults": faults, "overruns": overruns}


def _build_track_controller(cfg, model, data, maps, q0):
    """Build the WBC controller + estimator + reference + SafetyLayer, all seeded from the
    robot's MEASURED pose q0 (NOT the sim keyframe), so the ramp and posture targets start from
    where the robot actually is — no t=0 lurch (C2). Returns (ctrl, safety)."""
    from .estimator import StateEstimator
    from .safety import SafetyLayer
    q0 = np.asarray(q0, dtype=np.float64).reshape(-1).copy()
    ctrl = WBController(model, cfg); ctrl.reset(data)
    ctrl.q_home = q0                                   # posture/home target = the real pose (C2)
    ctrl.settle(data)                                  # CoM/balance target around the real pose
    ctrl.attach_reference(ReferenceTrajectory(model, maps, q0, cfg, 0.0, 0.0, 0.0))
    ctrl.attach_estimator(StateEstimator(model, maps))
    safety = SafetyLayer(model, cfg, maps); safety.begin(q0)   # ramp STARTS from the real pose (C2)
    return ctrl, safety


def _prewarm(ctrl, transport):
    """Pay the solver/JIT cold-start cost BEFORE kCustom so the first live solve isn't slow (C1)."""
    try:
        ctrl.step_track_estimated(transport.read_lowstate(), 0.0)
    except Exception:
        pass


def run_hw(cfg):
    """REAL on-robot tracking entry. Snapshots the robot pose and seeds the WBC from it (C2),
    pre-warms the QP, hands over with kCustom, then runs a DECOUPLED publish thread (streams the
    latest safe cmd at publish_period) alongside the WBC control loop (solves at control_period).
    A slow/throwing solve can't gap the stream (C1). On exit, releases to kPrepare (firmware
    re-holds the pose, robot stays standing) iff kCustom was engaged."""
    from .sdk_transport import SdkTransport
    model, data = load_t1_model(cfg.xml)
    maps = build_index_maps(model)
    tr = SdkTransport(model, maps)        # lazy-imports the SDK
    stop = threading.Event()
    pub = None
    try:
        q0 = _snapshot_pose(tr)
        ctrl, safety = _build_track_controller(cfg, model, data, maps, q0)
        _prewarm(ctrl, tr)
        hold = build_hold_cmd(model, maps, cfg, q0)
        tr.start(initial_cmd=hold)         # validate 29-DOF + kPrepare, queue hold, kCustom
        box = _CmdBox(hold)
        pub = threading.Thread(target=_publish_loop, args=(tr, box, cfg.publish_period, stop.is_set),
                               daemon=True)
        pub.start()                        # stream begins immediately after handover (no gap)
        print(f"[hw] kCustom engaged; streaming @ {1/cfg.publish_period:.0f}Hz, "
              f"solving @ {1/cfg.control_period:.0f}Hz. E-stop ready.")
        ticks = int(ctrl.ref.duration / cfg.control_period)
        res = _track_control_loop(cfg, ctrl, safety, tr, box, ticks=ticks, stop=stop.is_set)
        print(f"[hw] motion complete: {res}")
    except KeyboardInterrupt:
        print("[hw] interrupted")
    finally:
        stop.set()
        if pub is not None:
            pub.join(timeout=1.0)
        tr.stop()


def run_hw_hold(cfg, seconds=None):
    """REAL on-robot PD-hold of the current (prep) pose — the safest first hardware test.
    Operator must have set kPrepare on the remote first. Holds until `seconds` elapses (None =
    until Ctrl-C), then releases to kPrepare (firmware holds, robot stays standing). No QP,
    no estimator, no motion."""
    from .sdk_transport import SdkTransport
    model, _ = load_t1_model(cfg.xml)
    maps = build_index_maps(model)
    tr = SdkTransport(model, maps)
    try:
        hold = build_hold_cmd(model, maps, cfg, _snapshot_pose(tr))
        tr.start(initial_cmd=hold)
        ticks = None if seconds is None else int(seconds / cfg.control_period)
        print(f"[hw-hold] kCustom engaged; holding prep pose "
              f"({'until Ctrl-C' if ticks is None else f'{seconds:.1f}s'}). E-stop ready.")
        res = run_hold_loop(cfg, hold, tr, ticks=ticks)
        print(f"[hw-hold] done: {res}")
    except KeyboardInterrupt:
        print("[hw-hold] interrupted")
    finally:
        tr.stop()   # damps ONLY if kCustom was engaged; otherwise a no-op (robot stays held)


def _parse_segments(s):
    """'0,1' -> (0, 1); None -> None (full motion). Whitespace tolerated."""
    if s is None:
        return None
    return tuple(int(x) for x in s.split(",") if x.strip() != "")


def _parser():
    p = argparse.ArgumentParser(description="T1 whole-body QP controller")
    p.add_argument("--mode", choices=["balance", "track", "track-est", "track-est-safe", "hw", "hw-hold"], default="track")
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--time-scale", type=float, default=None)
    p.add_argument("--control-decimation", type=int, default=None)
    p.add_argument("--no-friction-ff", action="store_true")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--log", type=str, default=None,
                   help="optional CSV path for per-tick tracking diagnostics")
    p.add_argument("--motion", type=str, default=None,
                   help="path to a motion_plan .pkl (default: bundled motion_plan.pkl)")
    p.add_argument("--segments", type=str, default=None,
                   help="comma-separated segment indices to track, e.g. 0,1 (default: all)")
    p.add_argument("--torque-scale", type=float, default=None,
                   help="torque_limit_scale in (0,1]; reduce for conservative on-robot bring-up")
    p.add_argument("--control-period", type=float, default=None,
                   help="hardware loop cadence in seconds (default 0.02 = 50Hz)")
    return p


def _build_cfg(args):
    cfg = WBCConfig()
    if args.motion is not None:
        cfg.motion = args.motion
    if args.segments is not None:
        cfg.segments = _parse_segments(args.segments)
    if args.time_scale is not None:
        cfg.time_scale = args.time_scale
    if args.control_decimation is not None:
        cfg.control_decimation = args.control_decimation
    if args.torque_scale is not None:
        cfg.torque_limit_scale = args.torque_scale
    if args.control_period is not None:
        cfg.control_period = args.control_period
    if args.no_friction_ff:
        cfg.friction_ff = False  # disable the Coulomb friction feedforward term
    return cfg


def main():
    args = _parser().parse_args()
    cfg = _build_cfg(args)
    if args.mode == "balance":
        print(run_balance(cfg, args.seconds or 3.0))
    elif args.mode == "track-est":
        print(run_track_estimated(cfg, args.seconds))
    elif args.mode == "track-est-safe":
        print(run_track_estimated_safe(cfg, args.seconds))
    elif args.mode == "hw":
        run_hw(cfg)
    elif args.mode == "hw-hold":
        run_hw_hold(cfg, args.seconds)
    else:
        print(run_track(cfg, args.seconds, viewer=args.viewer, log=args.log))


if __name__ == "__main__":
    main()
