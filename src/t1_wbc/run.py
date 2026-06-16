"""Sim + control loop and CLI for the T1 whole-body QP controller."""
import argparse
import time
import numpy as np
import mujoco
from .model import load_t1_model, build_index_maps
from .config import WBCConfig
from .controller import WBController, BatchedWarpController, MjlabBatchedWarpController
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


def run_batched_balance(cfg, num_envs, ticks=200, perturb=True, seed=0):
    """Phase-2 GPU entry: B-batched settle then hold-home/balance for `ticks` steps.

    Builds a ``BatchedWarpController`` (mjwarp + batched-torch WBC), settles every
    env to full double stance, then runs the batched balance loop. Returns a summary
    dict with per-env feasibility / force-balance / upright over the whole run.
    """
    ctrl = BatchedWarpController(cfg, num_envs=num_envs)
    ctrl.reset(perturb=perturb, seed=seed)
    ncon = ctrl.settle()
    mg = ctrl.total_mass * (-float(ctrl.model.opt.gravity[2]))
    min_base_z = ctrl.base_z().copy()
    all_ok = True
    worst_fz_err = 0.0
    last = None
    for _ in range(ticks):
        diag = ctrl.step_balance()
        last = diag
        min_base_z = np.minimum(min_base_z, diag["base_z"])
        all_ok = all_ok and bool(diag["ok"].all())
        worst_fz_err = max(worst_fz_err, float(np.abs(diag["sum_fz"] - mg).max()))
    return dict(num_envs=num_envs, ticks=ticks,
                ncon_all8=bool((ncon == 8).all()),
                min_base_z=float(min_base_z.min()), upright=bool(min_base_z.min() > cfg.upright_z),
                all_feasible=bool(all_ok), worst_fz_err=float(worst_fz_err),
                mg=float(mg), final_min_fz=float(last["sum_fz"].min()))


def run_mjlab_balance(cfg, num_envs, ticks=200, perturb=True, seed=0):
    """Phase-3 entry: batched WBC balance driven INSIDE an mjlab Simulation.

    Same batched WBC pipeline as ``run_batched_balance`` but mjlab owns the sim
    (``MjlabBatchedWarpController``): per spec §7, each tick reads dynamics from the
    current forwarded ``sim.wp_data``, writes ``sim.data.ctrl`` in place, then
    ``sim.step()``. Returns the same per-run summary dict.
    """
    ctrl = MjlabBatchedWarpController(cfg, num_envs=num_envs)
    ctrl.reset(perturb=perturb, seed=seed)
    ncon = ctrl.settle()
    mg = ctrl.total_mass * (-float(ctrl.model.opt.gravity[2]))
    min_base_z = ctrl.base_z().copy()
    all_ok = True
    worst_fz_err = 0.0
    last = None
    for _ in range(ticks):
        diag = ctrl.step_balance()
        last = diag
        min_base_z = np.minimum(min_base_z, diag["base_z"])
        all_ok = all_ok and bool(diag["ok"].all())
        worst_fz_err = max(worst_fz_err, float(np.abs(diag["sum_fz"] - mg).max()))
    return dict(engine="mjlab", num_envs=num_envs, ticks=ticks,
                ncon_all8=bool((ncon == 8).all()),
                min_base_z=float(min_base_z.min()), upright=bool(min_base_z.min() > cfg.upright_z),
                all_feasible=bool(all_ok), worst_fz_err=float(worst_fz_err),
                mg=float(mg), final_min_fz=float(last["sum_fz"].min()))


def benchmark_throughput(cfg, batch_sizes=(64, 256, 1024), ticks=100, warmup=10, seed=0):
    """Measure batched-WBC throughput (steps/s) across batch sizes on the GPU.

    steps/s = (num_envs * timed_ticks) / wall_time over the full per-tick pipeline
    (warp dynamics extract -> batched ADMM solve -> recover_tau -> mjw.step). Prints
    a row per batch size and returns the list of per-B result dicts (the speedup curve).
    """
    import torch
    rows = []
    print(f"{'B':>6}  {'ticks':>6}  {'wall_s':>9}  {'ms/tick':>9}  {'steps/s':>12}  {'env_steps/s':>13}")
    for B in batch_sizes:
        bcfg = WBCConfig(**{**cfg.__dict__, "num_envs": B})
        ctrl = BatchedWarpController(bcfg, num_envs=B)
        ctrl.reset(perturb=True, seed=seed)
        ctrl.settle()
        for _ in range(warmup):                       # warm up kernels/caches
            ctrl.step_balance()
        if ctrl.tdev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ticks):
            ctrl.step_balance()
        if ctrl.tdev == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        ms_tick = wall / ticks * 1e3
        steps_s = ticks / wall                         # solver ticks/s (batched)
        env_steps_s = B * ticks / wall                 # env-steps/s (throughput)
        rows.append(dict(num_envs=B, ticks=ticks, wall_s=wall, ms_per_tick=ms_tick,
                         steps_per_s=steps_s, env_steps_per_s=env_steps_s))
        print(f"{B:>6}  {ticks:>6}  {wall:>9.3f}  {ms_tick:>9.2f}  {steps_s:>12.2f}  {env_steps_s:>13.1f}")
        del ctrl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    base = rows[0]["env_steps_per_s"]
    print("\nspeedup vs B={} (env-steps/s): ".format(batch_sizes[0])
          + "  ".join(f"B={r['num_envs']}:{r['env_steps_per_s']/base:.2f}x" for r in rows))
    return rows


def main():
    p = argparse.ArgumentParser(description="T1 whole-body QP controller")
    p.add_argument("--mode",
                   choices=["balance", "track", "batched-balance", "mjlab-balance", "benchmark"],
                   default="track")
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--time-scale", type=float, default=None)
    p.add_argument("--control-decimation", type=int, default=None)
    p.add_argument("--no-friction-ff", action="store_true")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--log", type=str, default=None,
                   help="optional CSV path for per-tick tracking diagnostics")
    # batched GPU (Phase 2) options
    p.add_argument("--num-envs", type=int, default=64, help="batch size for batched-balance")
    p.add_argument("--ticks", type=int, default=None,
                   help="ticks: balance default 200, benchmark default 30 (timed)")
    p.add_argument("--device", type=str, default=None, help="override solve device (cpu|cuda)")
    p.add_argument("--admm-max-iter", type=int, default=None,
                   help="ADMM iteration cap (B=64 GPU balance needs ~20000+; 50000 verified)")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 256, 1024],
                   help="batch sizes for the throughput benchmark")
    args = p.parse_args()
    cfg = WBCConfig()
    if args.time_scale is not None:
        cfg.time_scale = args.time_scale
    if args.control_decimation is not None:
        cfg.control_decimation = args.control_decimation
    if args.no_friction_ff:
        cfg.friction_ff = False  # disable the Coulomb friction feedforward term
    if args.device is not None:
        cfg.device = args.device
    if args.admm_max_iter is not None:
        cfg.admm_max_iter = args.admm_max_iter
    if args.mode == "balance":
        print(run_balance(cfg, args.seconds or 3.0))
    elif args.mode == "batched-balance":
        # default the GPU balance path to a feasible iteration cap (verified e2e: 50000)
        if args.device is None:
            cfg.device = "cuda"
        if args.admm_max_iter is None:
            cfg.admm_max_iter = 50000
        cfg.num_envs = args.num_envs
        print(run_batched_balance(cfg, args.num_envs, ticks=(args.ticks or 200)))
    elif args.mode == "mjlab-balance":
        # mjlab end-to-end (spec §7 / Phase 3): mjlab owns the sim; same WBC pipeline.
        if args.device is None:
            cfg.device = "cuda"
        if args.admm_max_iter is None:
            cfg.admm_max_iter = 50000
        cfg.num_envs = args.num_envs
        print(run_mjlab_balance(cfg, args.num_envs, ticks=(args.ticks or 200)))
    elif args.mode == "benchmark":
        # throughput sweep: the speedup curve is iteration-independent (every env does
        # identical work), so use a moderate iter cap to keep large-B runs tractable.
        if args.device is None:
            cfg.device = "cuda"
        if args.admm_max_iter is None:
            cfg.admm_max_iter = 4000
        benchmark_throughput(cfg, batch_sizes=tuple(args.batch_sizes),
                             ticks=(args.ticks or 30))
    else:
        print(run_track(cfg, args.seconds, viewer=args.viewer, log=args.log))


if __name__ == "__main__":
    main()
