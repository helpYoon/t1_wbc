"""Per-tick whole-body controller: settle, then balance/track -> JointCommand.
Backend-agnostic: produces JointCommands; run.py applies them via MuJoCoBackend/transport.
B=1 CPU path (CpuDynamics + chosen solver backend); settle stays single-MjData."""
import numpy as np, mujoco
from .model import build_index_maps, load_t1_model
from .dynamics import CpuDynamics
from .wbc_qp import assemble_wbc_qp, recover_tau
from .solver import make_solver
from .config import effective_ctrlrange
from .action_backend import JointCommand
from .targets import balance_targets, tracking_targets_from_refsample


class WBController:
    def __init__(self, model, cfg):
        self.model = model; self.cfg = cfg; self.nu = model.nu; self.nv = model.nv; self.dt = model.opt.timestep
        self.maps = build_index_maps(model)
        self.dyn = CpuDynamics(model, self.maps)
        self.est = None
        self._est_data = mujoco.MjData(model)
        self.solver = make_solver(cfg)
        self.ctrlrange = effective_ctrlrange(model, cfg, reserve_margin=True)
        self._t_est = 0.0
        self._qd_filt = None                              # EMA state for joint-velocity low-pass
        self._base_twist_filt = None                      # EMA state for base-twist low-pass
        self._servo_kp = np.full(self.nu, cfg.servo_kp)   # constant in the JointCommand (0 in sim;
        self._servo_kd = np.full(self.nu, cfg.servo_kd)   # SafetyLayer overrides on hardware)
        self.q_home = None; self.com_target = None; self._last = None; self.ref = None

    def attach_reference(self, ref):
        """Provide a ReferenceTrajectory for step_track."""
        self.ref = ref

    def attach_estimator(self, est):
        self.est = est

    def _filter_joint_vel(self, qd_meas):
        """First-order EMA low-pass on the measured joint velocity (recovers the temporal
        smoothing the MPC's horizon gave; the per-tick QP otherwise maps q̇ noise into tau_ff
        chatter). cfg.vel_filter_alpha=1.0 disables it."""
        a = float(self.cfg.vel_filter_alpha)
        if a >= 1.0:
            return qd_meas
        if self._qd_filt is None:
            self._qd_filt = np.asarray(qd_meas, dtype=np.float64).copy()
        else:
            self._qd_filt = a * np.asarray(qd_meas, dtype=np.float64) + (1.0 - a) * self._qd_filt
        return self._qd_filt

    def _filter_base_twist(self, twist):
        """EMA low-pass on the estimator base twist (lin+ang vel) BEFORE it enters the QP. The
        base twist drives the highest-weight tasks (CoM kd_com·Jcom·q̇ at w_com=300, base-ori
        kd_base_ori·ω), and unlike the joint velocity it was previously fed RAW — so estimator
        lin-vel / gyro noise mapped straight into leg tau_ff. cfg.base_vel_filter_alpha=1.0 off."""
        a = float(self.cfg.base_vel_filter_alpha)
        if a >= 1.0:
            return twist
        if self._base_twist_filt is None:
            self._base_twist_filt = np.asarray(twist, dtype=np.float64).copy()
        else:
            self._base_twist_filt = a * np.asarray(twist, dtype=np.float64) + (1.0 - a) * self._base_twist_filt
        return self._base_twist_filt

    def _assemble_est_dynamics(self, lowstate, base_twist=None, joint_dq=None):
        """Estimator + measured joints -> a forwarded MjData -> extract.
        `joint_dq` overrides lowstate.joint_dq (e.g. the velocity-filtered value)."""
        ls = lowstate
        t = self._t_est
        self.est.update_imu(ls.imu_rpy, ls.imu_gyro, ls.imu_acc, t)
        self.est.update_odometer(ls.odom_xytheta[0], ls.odom_xytheta[1], ls.odom_xytheta[2], t)
        self.est.update_base_pose_and_contacts(ls.joint_q)
        d = self._est_data
        q = self.est.quat_xyzw()
        d.qpos[0:3] = self.est.position()
        d.qpos[3:7] = [q[3], q[0], q[1], q[2]]                  # xyzw -> wxyz
        d.qpos[7:7 + self.nu] = ls.joint_q
        if base_twist is not None:
            d.qvel[0:6] = base_twist
        else:
            bt = np.empty(6, dtype=np.float64)
            bt[0:3] = self.est.lin_vel(); bt[3:6] = self.est.ang_vel()
            d.qvel[0:6] = self._filter_base_twist(bt)
        d.qvel[6:6 + self.nu] = ls.joint_dq if joint_dq is None else joint_dq
        mujoco.mj_forward(self.model, d)
        return self.dyn.extract(d)

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
        # Command the REFERENCE pose/velocity (computed-torque + PD form) — NOT measured-state +
        # integrated accel. The old form set qd_des≈qd_act, so on hardware the firmware kd term
        # (kd·(qd_des−q̇)) cancels to ~0 (no damping) and kp·(q_des−q)≈kp·dt·qd_act acts as velocity
        # ANTI-damping → whole-body buzz. (Harmless in sim: there servo kp=0, or zero-latency 500Hz
        # makes the 1-step prediction exact.) Tracking the reference makes the PD a proper
        # stabilizer; tau_ff carries the WBC dynamics/balance. Matches t1_controller's command.
        q_des = tg.q_ref if tg.q_ref is not None else tg.q_home
        qd_des = tg.qd_ref if tg.qd_ref is not None else np.zeros_like(q_act)
        cmd = JointCommand(q_des=np.asarray(q_des, dtype=np.float64).copy(),
                           qd_des=np.asarray(qd_des, dtype=np.float64).copy(),
                           kp=self._servo_kp, kd=self._servo_kd, tau_ff=tau_ff)
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

    def step_track_estimated(self, lowstate, t):
        assert self.ref is not None and self.est is not None
        self._t_est = t
        q_act = np.asarray(lowstate.joint_q, dtype=np.float64)
        qd_act = self._filter_joint_vel(np.asarray(lowstate.joint_dq, dtype=np.float64))
        d = self._assemble_est_dynamics(lowstate, joint_dq=qd_act)
        rs = self.ref.sample(t)
        tg = tracking_targets_from_refsample(rs, q_act, qd_act, self.q_home)
        cmd, z, ok, tau_ff = self._solve_to_cmd(d, tg, q_act, qd_act)
        lh_err = float(np.linalg.norm(d["hand_L_world"] - tg.lh_pos))
        rh_err = float(np.linalg.norm(d["hand_R_world"] - tg.rh_pos))
        return cmd, dict(ok=bool(ok), base_z=float(self.est.position()[2]),
                         max_tau=float(np.abs(tau_ff).max()), lh_err=lh_err, rh_err=rh_err)
