"""Sim + control loop and CLI for the T1 whole-body QP controller."""
import argparse
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


def run_hw_loop(cfg, model, data, maps, transport, ticks=None):
    """The hardware control loop: read LowState -> estimate+WBC -> SafetyLayer -> write LowCmd.
    Transport-agnostic (real SdkTransport on robot; mock in tests). Returns commands written.
    `data` is only used to settle / seed q_home."""
    from .estimator import StateEstimator
    from .safety import SafetyLayer
    ctrl = WBController(model, cfg); ctrl.reset(data); ctrl.settle(data)
    ctrl.attach_reference(ReferenceTrajectory(model, maps, ctrl.q_home, cfg, 0.0, 0.0, 0.0))
    ctrl.attach_estimator(StateEstimator(model, maps))
    safety = SafetyLayer(model, cfg, maps); safety.begin(ctrl.q_home)
    dt = model.opt.timestep; t = 0.0; n = 0
    horizon_ticks = ticks if ticks is not None else int(ctrl.ref.duration / dt)
    for i in range(horizon_ticks):
        ls = transport.read_lowstate()
        cmd, diag = ctrl.step_track_estimated(ls, t)
        safe = safety.wrap(cmd, ok=diag["ok"], t=t, lowstate_age=transport.state_age())
        transport.write_lowcmd(safe); n += 1; t += dt
    return n


def run_hw(cfg):
    """REAL on-robot entry. Requires the Booster SDK + a connected T1. Builds an SdkTransport,
    runs the kPrepare->kCustom handshake, then the hw loop; on exit, falls back to kDamping."""
    from .sdk_transport import SdkTransport
    import time as _t
    model, data = load_t1_model(cfg.xml)
    maps = build_index_maps(model)
    tr = SdkTransport(model, maps)        # lazy-imports the SDK
    _t.sleep(0.2)                          # let the first LowState callback arrive
    tr.start()
    try:
        run_hw_loop(cfg, model, data, maps, tr)
    finally:
        tr.stop()


def main():
    p = argparse.ArgumentParser(description="T1 whole-body QP controller")
    p.add_argument("--mode", choices=["balance", "track", "track-est", "track-est-safe", "hw"], default="track")
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--time-scale", type=float, default=None)
    p.add_argument("--control-decimation", type=int, default=None)
    p.add_argument("--no-friction-ff", action="store_true")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--log", type=str, default=None,
                   help="optional CSV path for per-tick tracking diagnostics")
    args = p.parse_args()
    cfg = WBCConfig()
    if args.time_scale is not None:
        cfg.time_scale = args.time_scale
    if args.control_decimation is not None:
        cfg.control_decimation = args.control_decimation
    if args.no_friction_ff:
        cfg.friction_ff = False  # disable the Coulomb friction feedforward term
    if args.mode == "balance":
        print(run_balance(cfg, args.seconds or 3.0))
    elif args.mode == "track-est":
        print(run_track_estimated(cfg, args.seconds))
    elif args.mode == "track-est-safe":
        print(run_track_estimated_safe(cfg, args.seconds))
    elif args.mode == "hw":
        run_hw(cfg)
    else:
        print(run_track(cfg, args.seconds, viewer=args.viewer, log=args.log))


if __name__ == "__main__":
    main()
