"""Capture a deterministic QP from the CURRENT (torch) t1_wbc on a fixed tick.
Oracle for the numpy de-torch. Run in the mpc-rl env (torch present)."""
import numpy as np, mujoco, torch
from t1_wbc.model import load_t1_model, build_index_maps
from t1_wbc.config import WBCConfig
from t1_wbc.controller import WBController
from t1_wbc.reference import ReferenceTrajectory
from t1_wbc.wbc_qp import assemble_wbc_qp, recover_tau
from t1_wbc.action_backend import MuJoCoBackend
from t1_wbc.targets import tracking_targets_from_refsample

cfg = WBCConfig()
model, data = load_t1_model(cfg.xml)
ctrl = WBController(model, cfg)
ctrl.reset(data); ctrl.settle(data)
ref = ReferenceTrajectory(model, build_index_maps(model), ctrl.q_home, cfg, 0.0, 0.0, 0.0)
ctrl.attach_reference(ref)
backend = MuJoCoBackend(model)
dt = model.opt.timestep
for i in range(int(1.0 / dt)):                 # advance to t=1.0 s (arms/CoM off-home)
    mujoco.mj_step1(model, data)
    ctrl.step_track(data, i * dt)
    backend.apply(ctrl._last, data)
    mujoco.mj_step2(model, data)

d = ctrl.dyn.extract(data)
rs = ctrl.ref.sample(1.0)
q_act, qd_act = ctrl._act_state(data)
tg = tracking_targets_from_refsample(rs, q_act, qd_act, ctrl.q_home)
qp = assemble_wbc_qp(d, tg, cfg, ctrl.ctrlrange, ctrl.nv, ctrl.nu)
z, ok = ctrl.solver.solve(qp)
tau = recover_tau(z, d, cfg, ctrl.nv)

t0 = lambda x: (x.detach().cpu().numpy()[0] if torch.is_tensor(x) else np.asarray(x))
out = dict(H=t0(qp.H), g=t0(qp.g), A_eq=t0(qp.A_eq), b_eq=t0(qp.b_eq),
           G=t0(qp.G), b=t0(qp.b), z=t0(z), tau=t0(tau))
for k, v in d.items():
    if torch.is_tensor(v) and k != "actuated_dof":
        out["dyn__" + k] = t0(v)
out["x_half"] = np.float64(d["x_half"]); out["y_half"] = np.float64(d["y_half"])
out["adof"] = d["actuated_dof"].detach().cpu().numpy()
out["ctrlrange"] = ctrl.ctrlrange.detach().cpu().numpy()
for nm, field in [("q_ref", tg.q_ref), ("qd_ref", tg.qd_ref), ("tracked", tg.tracked),
                  ("q_act", tg.q_act), ("qd_act", tg.qd_act), ("com_ref", tg.com_ref),
                  ("q_home", tg.q_home),
                  ("lh_pos", tg.lh_pos), ("lh_quat", tg.lh_quat_xyzw), ("lh_vel", tg.lh_vel),
                  ("rh_pos", tg.rh_pos), ("rh_quat", tg.rh_quat_xyzw), ("rh_vel", tg.rh_vel),
                  ("base_quat", tg.base_quat_xyzw), ("base_omega", tg.base_omega_world)]:
    out[nm] = t0(field)
np.savez("/tmp/golden_qp.npz", **out)
print("OK  nz", out["H"].shape[0], " infeas", float(np.max(np.maximum(out["G"] @ out["z"] - out["b"], 0.0))),
      " eqres", float(np.max(np.abs(out["A_eq"] @ out["z"] - out["b_eq"]))), " ok", bool(ok.all()))
