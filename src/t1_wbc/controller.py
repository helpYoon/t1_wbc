"""Per-tick whole-body controller: settle, then balance/track -> JointCommand.
Backend-agnostic: produces batched JointCommands; run.py applies them via an ActionBackend.
B=1 CPU path (CpuDynamics + chosen solver backend); settle stays single-MjData.

``BatchedWarpController`` is the B>1 GPU path: a mjwarp-batched settle + WBC balance
loop. Dynamics come from ``WarpDynamics`` (f32 on the warp device), are cast to f64
for the ADMM solve, and torques are written back to ``wd.ctrl`` as f32. Per-env
contact count uses ``bincount`` on ``wd.contact.worldid`` (there is no ``wd.ncon``)."""
import numpy as np, torch, mujoco
from .model import (build_index_maps, load_t1_warp_model, build_warp_handles,
                    T1_NCONMAX, T1_NJMAX)
from .dynamics import CpuDynamics
from .dynamics_warp import WarpDynamics
from .wbc_qp import assemble_wbc_qp, recover_tau
from .solver import make_solver, AdmmBackend
from .action_backend import JointCommand
from .targets import balance_targets, tracking_targets_from_refsample


# Lazy warp accessors so this module imports on CPU-only envs (the B=1 path needs
# no warp). The BatchedWarpController route imports mujoco_warp/warp on first use.
def _warp():
    import mujoco_warp as mjw, warp as wp
    return mjw, wp


def wp_to_torch(arr):
    _, wp = _warp()
    return wp.to_torch(arr)


def wp_from_torch_f32(tensor):
    _, wp = _warp()
    return wp.from_torch(tensor, dtype=wp.float32)


def mjw_forward(wm, wd):
    mjw, _ = _warp()
    mjw.forward(wm, wd)


def mjw_step(wm, wd):
    mjw, _ = _warp()
    mjw.step(wm, wd)


class WBController:
    def __init__(self, model, cfg):
        self.model = model; self.cfg = cfg; self.nu = model.nu; self.nv = model.nv; self.dt = model.opt.timestep
        self.maps = build_index_maps(model)
        self.dyn = CpuDynamics(model, self.maps, dtype=torch.float64)
        self.solver = make_solver(cfg)
        self.ctrlrange = torch.as_tensor(model.actuator_ctrlrange, dtype=torch.float64)
        self.q_home = None; self.com_target = None; self._last = None; self.ref = None

    def attach_reference(self, ref):
        """Provide a ReferenceTrajectory for step_track."""
        self.ref = ref

    def reset(self, data):
        mujoco.mj_resetDataKeyframe(self.model, data, 0)
        self.q_home = torch.as_tensor(data.qpos[7:7+self.nu].copy(), dtype=torch.float64).unsqueeze(0)

    def settle(self, data):
        nu = self.nu; qh = self.q_home[0].numpy()
        for _ in range(int(self.cfg.settle_seconds / self.dt)):
            mujoco.mj_step1(self.model, data)
            tau = self.cfg.settle_kp*(qh - data.qpos[7:7+nu]) + self.cfg.settle_kd*(-data.qvel[6:6+nu])
            data.ctrl[:] = np.clip(tau, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
            mujoco.mj_step2(self.model, data)
        mujoco.mj_forward(self.model, data); d = self.dyn.extract(data)
        sup = 0.5*(d["foot_L_world"][0, :2] + d["foot_R_world"][0, :2])
        self.com_target = torch.tensor([[sup[0], sup[1], d["com"][0, 2]]], dtype=torch.float64)
        return data.ncon

    def _act_state(self, data):
        q_act = torch.as_tensor(data.qpos[7:7+self.nu].copy(), dtype=torch.float64).unsqueeze(0)
        qd_act = torch.as_tensor(data.qvel[6:6+self.nu].copy(), dtype=torch.float64).unsqueeze(0)
        return q_act, qd_act

    def _solve_to_cmd(self, d, tg, q_act, qd_act):
        qp = assemble_wbc_qp(d, tg, self.cfg, self.ctrlrange, self.nv, self.nu)
        z, ok = self.solver.solve(qp)
        tau_ff = recover_tau(z, d, self.cfg, self.nv)
        vdot_a = z[:, d["actuated_dof"]]
        qd_des = qd_act + vdot_a*self.dt; q_des = q_act + qd_act*self.dt + 0.5*vdot_a*self.dt**2
        servo = lambda v: torch.full((1, self.nu), v, dtype=torch.float64)
        cmd = JointCommand(q_des=q_des, qd_des=qd_des, kp=servo(self.cfg.servo_kp),
                           kd=servo(self.cfg.servo_kd), tau_ff=tau_ff)
        self._last = cmd
        return cmd, z, ok, tau_ff

    def step_balance(self, data):
        d = self.dyn.extract(data)
        q_act, qd_act = self._act_state(data)
        tg = balance_targets(self.q_home, q_act, qd_act, self.com_target)
        cmd, z, ok, tau_ff = self._solve_to_cmd(d, tg, q_act, qd_act)
        return cmd, dict(ok=bool(ok.all()), base_z=float(data.qpos[2]),
                         min_fz=float(min(z[0, self.nv+2], z[0, self.nv+8])), max_tau=float(tau_ff.abs().max()))

    def step_track(self, data, t):
        assert self.ref is not None, "call attach_reference(ref) before step_track"
        d = self.dyn.extract(data)
        q_act, qd_act = self._act_state(data)
        rs = self.ref.sample(t)
        tg = tracking_targets_from_refsample(rs, q_act, qd_act, self.q_home, dtype=torch.float64)
        cmd, z, ok, tau_ff = self._solve_to_cmd(d, tg, q_act, qd_act)
        lh_err = float(torch.linalg.norm(d["hand_L_world"][0] - tg.lh_pos[0]))
        rh_err = float(torch.linalg.norm(d["hand_R_world"][0] - tg.rh_pos[0]))
        return cmd, dict(ok=bool(ok.all()), base_z=float(data.qpos[2]),
                         min_fz=float(min(z[0, self.nv+2], z[0, self.nv+8])),
                         max_tau=float(tau_ff.abs().max()), lh_err=lh_err, rh_err=rh_err)


class BatchedWarpController:
    """B>1 GPU whole-body balance controller (mjwarp + batched-torch WBC).

    Pipeline: ``load_t1_warp_model`` -> ``WarpDynamics`` -> PD-hold settle -> lock a
    per-env CoM target at the support center -> per-tick batched WBC solve. The QP is
    solved in f64 (verified robust, ``rho_eq=100``); torques are cast to f32 for
    ``wd.ctrl``. Per-env feasibility uses the external residual, never ``sol.converged``.
    """

    def __init__(self, cfg, num_envs=None, xml=None):
        self.cfg = cfg
        self.B = int(num_envs if num_envs is not None else cfg.num_envs)
        self.DT = torch.float64 if cfg.dtype == "float64" else torch.float32
        m, wm, wd, handles = load_t1_warp_model(self.B, xml=(xml or cfg.xml))
        self.wm = wm; self.wd = wd
        self._init_common(m, wm, handles)

    def _init_common(self, m, wm, handles):
        """Shared state setup for any batched-warp engine (standalone or mjlab).

        Subclasses set the engine-specific sim handles before calling this; everything
        here is model-/handle-derived and identical across engines."""
        self.model = m; self.handles = handles
        self.nv = m.nv; self.nu = m.nu; self.dt = float(m.opt.timestep)
        self.dev = handles["wp_device"]
        self.tdev = "cuda" if self.dev.startswith("cuda") else "cpu"
        self.total_mass = float(handles["total_mass"])
        self.dyn = WarpDynamics(wm, handles)
        self.solver = AdmmBackend(self.cfg)
        self.ctrlrange = torch.as_tensor(handles["ctrlrange"], dtype=self.DT, device=self.tdev)
        # home keyframe (actuated qpos slice)
        key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.qpos0 = m.key_qpos[key].copy()
        self.q_home_np = self.qpos0[7:7 + self.nu].copy()
        self.q_home = torch.as_tensor(self.q_home_np, dtype=torch.float32, device=self.tdev)
        self.com_target = None

    # ---- engine hooks (overridden by the mjlab subclass) --------------------
    # The standalone-Warp engine owns the sim: forward/step on (wm, wd), read/write
    # qpos/qvel/ctrl as wp arrays. The mjlab subclass overrides these to drive
    # ``sim.forward()``/``sim.step()`` and read/write the zero-copy torch views.
    def _engine_forward(self):
        mjw_forward(self.wm, self.wd)

    def _engine_step(self):
        mjw_step(self.wm, self.wd)

    def _read_qpos(self):
        return wp_to_torch(self.wd.qpos)

    def _read_qvel(self):
        return wp_to_torch(self.wd.qvel)

    def _write_ctrl(self, tau_f32):
        """Write torques (already cast to f32) into the engine's batched ctrl."""
        self.wd.ctrl = wp_from_torch_f32(tau_f32.contiguous())

    def _set_state(self, qpos_B_np, qvel_B_np):
        """Seed the engine's batched qpos/qvel from numpy (B, nq)/(B, nv)."""
        self.wd.qpos = wp_from_torch_f32(torch.as_tensor(qpos_B_np.astype(np.float32), device=self.tdev))
        self.wd.qvel = wp_from_torch_f32(torch.as_tensor(qvel_B_np.astype(np.float32), device=self.tdev))

    def _extract_dyn(self):
        return self.dyn.extract(self.wd)

    # ---- warp <-> torch helpers ---------------------------------------------
    def _wd_qpos(self):
        return self._read_qpos()

    def _wd_qvel(self):
        return self._read_qvel()

    def base_z(self):
        """Per-env base height (B,) numpy from the current sim state."""
        return self._wd_qpos()[:, 2].detach().cpu().numpy()

    # ---- reset / settle ------------------------------------------------------
    def reset(self, perturb=False, seed=0):
        """Seed every env from the home keyframe (optionally per-env perturbed)."""
        B, nv, nu = self.B, self.nv, self.nu
        qpos_B = np.tile(self.qpos0, (B, 1)).astype(np.float64)
        qvel_B = np.zeros((B, nv), dtype=np.float64)
        if perturb:
            rng = np.random.default_rng(seed)
            qpos_B[1:, 0:3] += rng.normal(0, 0.01, size=(B - 1, 3))    # base pos jitter
            qpos_B[1:, 7:] += rng.normal(0, 0.03, size=(B - 1, nu))    # actuated-joint jitter
            qvel_B[1:, :] += rng.normal(0, 0.03, size=(B - 1, nv))     # generalized-vel jitter
        self._set_state(qpos_B, qvel_B)
        self.com_target = None

    def _contact_worldid(self):
        """Return (nacon, worldid[:nacon]) from the flat contact buffer (no wd.ncon)."""
        nacon = int(wp_to_torch(self.wd.nacon).detach().cpu().numpy().ravel()[0])
        wid = wp_to_torch(self.wd.contact.worldid)[:nacon].long()
        return nacon, wid

    def _per_env_ncon(self):
        """Per-env contact count (B,) from the flat contact buffer (no wd.ncon)."""
        _, wid = self._contact_worldid()
        return torch.bincount(wid, minlength=self.B)

    def settle(self, kp=None, kd=None):
        """PD-hold settle for ``settle_seconds``; lock per-env CoM target at support
        center. Returns per-env contact count (B,) numpy."""
        B, nu = self.B, self.nu
        kp = self.cfg.settle_kp if kp is None else kp
        kd = self.cfg.settle_kd if kd is None else kd
        kp_t = torch.full((nu,), float(kp), device=self.tdev)
        kd_t = torch.full((nu,), float(kd), device=self.tdev)
        n_settle = int(self.cfg.settle_seconds / self.dt)
        for _ in range(n_settle):
            self._engine_forward()
            qpos = self._wd_qpos(); qvel = self._wd_qvel()
            q_act = qpos[:, 7:7 + nu]; qd_act = qvel[:, 6:6 + nu]
            tau = kp_t * (self.q_home - q_act) + kd_t * (-qd_act)
            self._write_ctrl(tau.to(torch.float32))
            self._engine_step()
        self._engine_forward()
        ncon = self._per_env_ncon()
        # lock per-env CoM target at the post-settle support center
        dyn0 = self._extract_dyn()
        sup = 0.5 * (dyn0["foot_L_world"][:, :2] + dyn0["foot_R_world"][:, :2])   # (B,2)
        self.com_target = torch.cat([sup, dyn0["com"][:, 2:3]], dim=1).to(self.DT)  # (B,3)
        return ncon.detach().cpu().numpy()

    # ---- balance tick --------------------------------------------------------
    def step_balance(self):
        """One batched WBC balance tick: extract -> solve -> recover_tau -> step.
        Returns a per-env diagnostic dict (ok/eq/viol/sum_fz/base_z/max_tau)."""
        assert self.com_target is not None, "call settle() before step_balance()"
        B, nv, nu = self.B, self.nv, self.nu
        # §7 ordering: forward at the CURRENT state -> extract -> assemble -> solve ->
        # recover_tau -> write ctrl -> step. The mjlab subclass replaces the forward/step
        # hooks with sim.forward()/sim.step() but keeps this exact contract (no staleness).
        self._engine_forward()
        dyn_f32 = self._extract_dyn()                            # GPU batched dynamics (f32)
        dyn = {k: (v.to(self.DT) if torch.is_tensor(v) and v.dtype.is_floating_point else v)
               for k, v in dyn_f32.items()}                      # f64 for the ADMM solve
        qpos = self._wd_qpos(); qvel = self._wd_qvel()
        q_act = qpos[:, 7:7 + nu].to(self.DT)
        qd_act = qvel[:, 6:6 + nu].to(self.DT)
        q_home = self.q_home.to(self.DT).unsqueeze(0).expand(B, -1)
        tg = balance_targets(q_home, q_act, qd_act, self.com_target)
        qp = assemble_wbc_qp(dyn, tg, self.cfg, self.ctrlrange, nv, nu)
        z, ok = self.solver.solve(qp)                            # trust external residual
        tau = recover_tau(z, dyn, self.cfg, nv)                  # (B,nu) f64
        self._write_ctrl(tau.to(torch.float32))                 # f32 write-back
        self._engine_step()
        sum_fz = (z[:, nv + 2] + z[:, nv + 8]).detach().cpu().numpy()
        # Per-env external residual (never trust sol.converged). The batch-max scalars
        # are just amax() over these, so there's no separate external_residual() call.
        eq_pe = (torch.bmm(qp.A_eq, z.unsqueeze(-1)).squeeze(-1) - qp.b_eq).abs().amax(dim=1)
        viol_pe = torch.clamp(torch.bmm(qp.G, z.unsqueeze(-1)).squeeze(-1) - qp.b, min=0.0).amax(dim=1)
        return dict(ok=ok.detach().cpu().numpy(),
                    eq=eq_pe.detach().cpu().numpy(), viol=viol_pe.detach().cpu().numpy(),
                    eq_max=float(eq_pe.max()), viol_max=float(viol_pe.max()),
                    sum_fz=sum_fz, base_z=self.base_z(),
                    max_tau=tau.abs().amax(dim=-1).detach().cpu().numpy())


# ============================================================================
# mjlab integration (spec §7, Phase 3) — drive the batched WBC inside an mjlab
# Simulation instead of a t1_wbc-owned standalone Warp sim.
# ============================================================================
def build_mjlab_sim(cfg, num_envs, device=None, xml=None):
    """Build a minimal batched mjlab T1 scene + dynamics handles.

    mjlab IS mujoco_warp: ``mjlab.sim.Simulation`` takes a ``mujoco.MjModel`` directly
    and exposes ``sim.wp_model``/``sim.wp_data`` (raw mujoco_warp Model/Data — what
    ``WarpDynamics.extract`` consumes) plus ``sim.model``/``sim.data`` (zero-copy torch
    views). The "scene" here is the bare T1 MjModel loaded into a batched Simulation;
    no terrain/asset_zoo entity is needed for a balance gate.

    The DENSE jacobian (``mjJAC_DENSE`` on ``m.opt`` before construction AND the cfg's
    ``jacobian='dense'``) is what lets ``WarpDynamics`` slice the padded ``qM`` to ``nv``;
    we assert ``sim.wp_model.is_sparse is False``. Returns ``(sim, handles)`` where
    ``handles`` matches ``load_t1_warp_model`` (built via the shared ``build_warp_handles``
    so dynamics extraction is engine-identical), with ``wp_device``/``nworld`` taken from
    the mjlab sim.
    """
    import dataclasses
    from mjlab.sim.sim import Simulation, SimulationCfg, MujocoCfg

    B = int(num_envs)
    device = device or ("cuda" if (cfg.device or "").startswith("cuda") else "cpu")
    m = mujoco.MjModel.from_xml_path(xml or cfg.xml)
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE        # dense qM for the WBC slice
    mcfg = dataclasses.replace(MujocoCfg(), timestep=float(m.opt.timestep),
                               jacobian="dense", cone="pyramidal")
    # Size the warp contact/constraint buffers above the T1 double-stance nefc so
    # large-B perturbed runs don't overflow and drop contacts (see load_t1_warp_model).
    scfg = SimulationCfg(mujoco=mcfg, nconmax=T1_NCONMAX, njmax=T1_NJMAX)
    sim = Simulation(num_envs=B, cfg=scfg, model=m, device=device)
    assert sim.wp_model.is_sparse is False, "mjlab T1 scene must use a dense jacobian"
    handles = build_warp_handles(m, B, str(sim.wp_data.qpos.device))
    return sim, handles


class MjlabBatchedWarpController(BatchedWarpController):
    """B>1 GPU balance controller driving the WBC inside an mjlab ``Simulation``.

    Identical batched WBC pipeline to ``BatchedWarpController`` (WarpDynamics -> f64
    ADMM solve -> f32 recover_tau), but the sim is owned by mjlab. Per spec §7 (mjlab
    path), each tick: read dynamics from the CURRENT forwarded ``sim.wp_data`` ->
    compute ctrl -> write ``sim.data.ctrl`` in place -> ``sim.step()``. Dynamics
    extraction uses ``sim.wp_model``/``sim.wp_data`` (the raw mujoco_warp structs);
    state and ctrl use ``sim.data`` (zero-copy torch views, mutated in place).
    """

    def __init__(self, cfg, num_envs=None, xml=None, device=None):
        self.cfg = cfg
        self.B = int(num_envs if num_envs is not None else cfg.num_envs)
        self.DT = torch.float64 if cfg.dtype == "float64" else torch.float32
        sim, handles = build_mjlab_sim(cfg, self.B, device=device, xml=(xml or cfg.xml))
        self.sim = sim
        # WarpDynamics + the engine hooks operate on the raw mujoco_warp structs.
        self.wm = sim.wp_model
        self.wd = sim.wp_data
        self._init_common(sim.mj_model, sim.wp_model, handles)

    # ---- engine hooks: mjlab owns the sim -----------------------------------
    def _engine_forward(self):
        self.sim.forward()

    def _engine_step(self):
        self.sim.step()

    def _read_qpos(self):
        # zero-copy torch view of the batched qpos (B, nq)
        return self.sim.data.qpos

    def _read_qvel(self):
        return self.sim.data.qvel

    def _write_ctrl(self, tau_f32):
        """Write torques (f32) into the sim's batched ctrl IN PLACE (zero-copy view)."""
        ctrl = self.sim.data.ctrl
        ctrl[:] = tau_f32.to(dtype=ctrl.dtype, device=ctrl.device)

    def _set_state(self, qpos_B_np, qvel_B_np):
        """Seed the mjlab sim's batched qpos/qvel in place (zero-copy torch views)."""
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        qpos[:] = torch.as_tensor(qpos_B_np, dtype=qpos.dtype, device=qpos.device)
        qvel[:] = torch.as_tensor(qvel_B_np, dtype=qvel.dtype, device=qvel.device)

    def _extract_dyn(self):
        # WarpDynamics.extract consumes the raw mujoco_warp Data (sim.wp_data).
        return self.dyn.extract(self.sim.wp_data)
