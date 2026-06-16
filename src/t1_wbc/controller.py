"""Per-tick whole-body controller: settle, then balance/track -> JointCommand.
Backend-agnostic: produces JointCommands; run.py applies them via an ActionBackend.
B=1 CPU path (CpuDynamics + chosen solver backend); settle stays single-MjData."""
import numpy as np, mujoco
from .model import build_index_maps, load_t1_model
from .dynamics import CpuDynamics
from .wbc_qp import assemble_wbc_qp, recover_tau
from .solver import make_solver
from .action_backend import JointCommand
from .targets import balance_targets, tracking_targets_from_refsample


class WBController:
    def __init__(self, model, cfg):
        self.model = model; self.cfg = cfg; self.nu = model.nu; self.nv = model.nv; self.dt = model.opt.timestep
        self.maps = build_index_maps(model)
        self.dyn = CpuDynamics(model, self.maps)
        self.solver = make_solver(cfg)
        self.ctrlrange = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
        self.q_home = None; self.com_target = None; self._last = None; self.ref = None

    def attach_reference(self, ref):
        """Provide a ReferenceTrajectory for step_track."""
        self.ref = ref

    def reset(self, data):
        mujoco.mj_resetDataKeyframe(self.model, data, 0)
        self.q_home = data.qpos[7:7+self.nu].copy()

    def settle(self, data):
        nu = self.nu; qh = self.q_home
        for _ in range(int(self.cfg.settle_seconds / self.dt)):
            mujoco.mj_step1(self.model, data)
            tau = self.cfg.settle_kp*(qh - data.qpos[7:7+nu]) + self.cfg.settle_kd*(-data.qvel[6:6+nu])
            data.ctrl[:] = np.clip(tau, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
            mujoco.mj_step2(self.model, data)
        mujoco.mj_forward(self.model, data); d = self.dyn.extract(data)
        sup = 0.5*(d["foot_L_world"][:2] + d["foot_R_world"][:2])
        self.com_target = np.array([sup[0], sup[1], d["com"][2]])
        return data.ncon

    def _act_state(self, data):
        return data.qpos[7:7+self.nu].copy(), data.qvel[6:6+self.nu].copy()

    def _solve_to_cmd(self, d, tg, q_act, qd_act):
        qp = assemble_wbc_qp(d, tg, self.cfg, self.ctrlrange, self.nv, self.nu)
        z, ok = self.solver.solve(qp)
        tau_ff = recover_tau(z, d, self.cfg, self.nv)
        vdot_a = z[d["actuated_dof"]]
        qd_des = qd_act + vdot_a*self.dt
        q_des = q_act + qd_act*self.dt + 0.5*vdot_a*self.dt**2
        servo = lambda v: np.full(self.nu, v)
        cmd = JointCommand(q_des=q_des, qd_des=qd_des,
                           kp=servo(self.cfg.servo_kp), kd=servo(self.cfg.servo_kd), tau_ff=tau_ff)
        self._last = cmd
        return cmd, z, ok, tau_ff

    def step_balance(self, data):
        d = self.dyn.extract(data)
        q_act, qd_act = self._act_state(data)
        tg = balance_targets(self.q_home, q_act, qd_act, self.com_target)
        cmd, z, ok, tau_ff = self._solve_to_cmd(d, tg, q_act, qd_act)
        return cmd, dict(ok=bool(ok), base_z=float(data.qpos[2]),
                         min_fz=float(min(z[self.nv+2], z[self.nv+8])),
                         max_tau=float(np.abs(tau_ff).max()))

    def step_track(self, data, t):
        assert self.ref is not None, "call attach_reference(ref) before step_track"
        d = self.dyn.extract(data)
        q_act, qd_act = self._act_state(data)
        rs = self.ref.sample(t)
        tg = tracking_targets_from_refsample(rs, q_act, qd_act, self.q_home)
        cmd, z, ok, tau_ff = self._solve_to_cmd(d, tg, q_act, qd_act)
        lh_err = float(np.linalg.norm(d["hand_L_world"] - tg.lh_pos))
        rh_err = float(np.linalg.norm(d["hand_R_world"] - tg.rh_pos))
        return cmd, dict(ok=bool(ok), base_z=float(data.qpos[2]),
                         min_fz=float(min(z[self.nv+2], z[self.nv+8])),
                         max_tau=float(np.abs(tau_ff).max()), lh_err=lh_err, rh_err=rh_err)
