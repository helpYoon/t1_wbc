"""Batched-torch WBC QP assembly (finite-only inequality rows) + torque recovery.
Verified to match the legacy numpy formulation to machine precision and to solve == proxsuite."""
from dataclasses import dataclass
import torch

@dataclass
class BatchedQP:
    H: torch.Tensor; g: torch.Tensor          # (B,nz,nz),(B,nz)
    A_eq: torch.Tensor; b_eq: torch.Tensor    # (B,neq,nz),(B,neq)
    G: torch.Tensor; b: torch.Tensor          # (B,nineq,nz),(B,nineq)  (G z <= b, finite-only)

def ori_error_world(q_des_xyzw, q_cur_xyzw):
    """Batched world-frame orientation error 2·log(q_des ⊗ q_cur⁻¹) -> (B,3) rotation vector.
    Inputs (B,4) xyzw. The returned vector rotates the current frame onto the desired frame."""
    qd = q_des_xyzw; qc = q_cur_xyzw
    # q_cur^{-1} (unit quaternion -> conjugate): negate vector part
    cx, cy, cz, cw = -qc[:, 0], -qc[:, 1], -qc[:, 2], qc[:, 3]
    dx, dy, dz, dw = qd[:, 0], qd[:, 1], qd[:, 2], qd[:, 3]
    # e = q_des ⊗ q_cur^{-1}  (Hamilton product, xyzw)
    ew = dw * cw - dx * cx - dy * cy - dz * cz
    ex = dw * cx + dx * cw + dy * cz - dz * cy
    ey = dw * cy - dx * cz + dy * cw + dz * cx
    ez = dw * cz + dx * cy - dy * cx + dz * cw
    # canonicalize to the short way (w >= 0)
    sgn = torch.where(ew < 0, -torch.ones_like(ew), torch.ones_like(ew))
    ew, ex, ey, ez = ew * sgn, ex * sgn, ey * sgn, ez * sgn
    vnorm = torch.sqrt(ex * ex + ey * ey + ez * ez)
    angle = 2.0 * torch.atan2(vnorm, ew)
    scale = torch.where(vnorm > 1e-12, angle / vnorm.clamp_min(1e-12), torch.zeros_like(vnorm))
    return torch.stack([ex * scale, ey * scale, ez * scale], dim=-1)


def _contact_transpose(dyn, nv):
    B = dyn["Jfoot_L"].shape[0]
    JcT = torch.zeros(B, nv, 12, dtype=dyn["Jfoot_L"].dtype, device=dyn["Jfoot_L"].device)
    JcT[:, :, 0:6] = dyn["Jfoot_L"].transpose(1, 2)
    JcT[:, :, 6:12] = dyn["Jfoot_R"].transpose(1, 2)
    return JcT

def _finite_contact_rows(B, nz, nv, mu, x_half, y_half, fz_min, dtype, device):
    specs = []  # (list[(col,val)], lc): sum val*z[col] >= lc
    for off in (0, 6):
        c = nv + off; rz = mu * (x_half + y_half)
        specs += [([(c+2, 1.)], fz_min),
                  ([(c+0, 1.), (c+2, mu)], 0.), ([(c+0, -1.), (c+2, mu)], 0.),
                  ([(c+1, 1.), (c+2, mu)], 0.), ([(c+1, -1.), (c+2, mu)], 0.),
                  ([(c+4, 1.), (c+2, x_half)], 0.), ([(c+4, -1.), (c+2, x_half)], 0.),
                  ([(c+3, 1.), (c+2, y_half)], 0.), ([(c+3, -1.), (c+2, y_half)], 0.),
                  ([(c+5, 1.), (c+2, rz)], 0.), ([(c+5, -1.), (c+2, rz)], 0.)]
    m = len(specs)
    G = torch.zeros(m, nz, dtype=dtype, device=device); bvec = torch.zeros(m, dtype=dtype, device=device)
    for r, (idxvals, lc) in enumerate(specs):
        for col, val in idxvals: G[r, col] = -val      # -C z <= -lc
        bvec[r] = -lc
    return G.unsqueeze(0).expand(B, -1, -1).contiguous(), bvec.unsqueeze(0).expand(B, -1).contiguous()

def _torque_rows(dyn, ctrlrange, nz, nv, JcT, tau_fric):
    B = dyn["M"].shape[0]; adof = dyn["actuated_dof"]; nu = adof.shape[0]
    dtype = dyn["M"].dtype; device = dyn["M"].device
    Tsel = torch.zeros(B, nu, nz, dtype=dtype, device=device)
    Tsel[:, :, :nv] = dyn["M"][:, adof, :]; Tsel[:, :, nv:] = -JcT[:, adof, :]
    tau_const = dyn["h"][:, adof] + tau_fric[:, adof]
    lo = ctrlrange[:, 0].unsqueeze(0) - tau_const; hi = ctrlrange[:, 1].unsqueeze(0) - tau_const
    return torch.cat([Tsel, -Tsel], dim=1), torch.cat([hi, -lo], dim=1)

def assemble_wbc_qp(dyn, targets, cfg, ctrlrange, nv, nu):
    B = dyn["M"].shape[0]; nz = nv + 12
    dtype = dyn["M"].dtype; device = dyn["M"].device; adof = dyn["actuated_dof"]
    JcT = _contact_transpose(dyn, nv)
    A_base = torch.zeros(B, 6, nz, dtype=dtype, device=device)
    A_base[:, :, :nv] = dyn["M"][:, :6, :]; A_base[:, :, nv:] = -JcT[:, :6, :]
    A_feet = torch.zeros(B, 12, nz, dtype=dtype, device=device)
    A_feet[:, 0:6, :nv] = dyn["Jfoot_L"]; A_feet[:, 6:12, :nv] = dyn["Jfoot_R"]
    A_eq = torch.cat([A_base, A_feet], dim=1)
    b_eq = torch.cat([-dyn["h"][:, :6], torch.zeros(B, 12, dtype=dtype, device=device)], dim=1)
    tau_fric = (dyn["tau_fric_coeff"] * torch.tanh(dyn["qvel"] / cfg.fric_eps)
                if cfg.friction_ff else torch.zeros_like(dyn["qvel"]))
    Gc, bc = _finite_contact_rows(B, nz, nv, cfg.mu, dyn["x_half"], dyn["y_half"], cfg.fz_min, dtype, device)
    Gt, bt = _torque_rows(dyn, ctrlrange, nz, nv, JcT, tau_fric)
    G = torch.cat([Gc, Gt], dim=1); b = torch.cat([bc, bt], dim=1)
    H = torch.zeros(B, nz, nz, dtype=dtype, device=device); g = torch.zeros(B, nz, dtype=dtype, device=device)

    def _add_task(J, a_des, w):
        # weighted operational-space task: min ||J vdot - a_des||^2_w  ->  H += Jᵀ W J ; g += -Jᵀ W a_des
        JWl = J.transpose(1, 2) * w
        H[:, :nv, :nv] += torch.bmm(JWl, J)
        g[:, :nv] += -torch.bmm(JWl, a_des.unsqueeze(-1)).squeeze(-1)

    # posture / joint task (tracked -> q_ref weight w_track_joint, else home weight w_post)
    q_ref = targets.q_ref if targets.q_ref is not None else targets.q_home
    qd_ref = targets.qd_ref if targets.qd_ref is not None else torch.zeros_like(targets.q_home)
    vdot_j = cfg.kp_post * (q_ref - targets.q_act) + cfg.kd_post * (qd_ref - targets.qd_act)
    w_j = torch.where(targets.tracked, torch.as_tensor(cfg.w_track_joint, dtype=dtype, device=device),
                      torch.as_tensor(cfg.w_post, dtype=dtype, device=device))
    # scatter the per-DOF posture weights onto H's diagonal / g (adof entries are unique)
    H[:, adof, adof] += w_j
    g[:, adof] += -(w_j * vdot_j)
    # CoM task
    Jcom = dyn["Jcom"]; Jcomv = torch.bmm(Jcom, dyn["qvel"].unsqueeze(-1)).squeeze(-1)
    a_com = cfg.kp_com * (targets.com_ref - dyn["com"]) + cfg.kd_com * (-Jcomv)
    w_com = torch.as_tensor(cfg.w_com, dtype=dtype, device=device)
    _add_task(Jcom, a_com, w_com)

    # hand position (and optional orientation) tracking tasks
    w_hand = torch.as_tensor(cfg.w_hand, dtype=dtype, device=device)
    for pos_key, quat_t, vel_t, Jkey, world_key, cur_quat_key in (
        (targets.lh_pos, targets.lh_quat_xyzw, targets.lh_vel, "Jhand_L", "hand_L_world", "hand_L_quat_xyzw"),
        (targets.rh_pos, targets.rh_quat_xyzw, targets.rh_vel, "Jhand_R", "hand_R_world", "hand_R_quat_xyzw"),
    ):
        if pos_key is None:
            continue
        Jh = dyn[Jkey]                                   # (B,6,nv) [jacp; jacr]
        Jhv = torch.bmm(Jh, dyn["qvel"].unsqueeze(-1)).squeeze(-1)   # (B,6)
        Jp = Jh[:, :3, :]
        vel_des = vel_t if vel_t is not None else torch.zeros(B, 3, dtype=dtype, device=device)
        a_pos = cfg.kp_hand * (pos_key - dyn[world_key]) + cfg.kd_hand * (vel_des - Jhv[:, :3])
        _add_task(Jp, a_pos, w_hand)
        if quat_t is not None and cur_quat_key in dyn:    # optional hand orientation task
            Jr = Jh[:, 3:6, :]
            a_ori_h = cfg.kp_hand * ori_error_world(quat_t, dyn[cur_quat_key]) \
                + cfg.kd_hand * (-Jhv[:, 3:6])
            _add_task(Jr, a_ori_h, w_hand)
    # base orientation tracking task (constant world-frame angular selector = rows 3:6)
    if targets.base_quat_xyzw is not None:
        J_base_ang = torch.zeros(B, 3, nv, dtype=dtype, device=device)
        J_base_ang[:, :, 3:6] = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
        omega_cur = dyn["qvel"][:, 3:6]
        omega_des = (targets.base_omega_world if targets.base_omega_world is not None
                     else torch.zeros(B, 3, dtype=dtype, device=device))
        ori_err = ori_error_world(targets.base_quat_xyzw, dyn["base_quat_xyzw"])
        a_ori = cfg.kp_base_ori * ori_err + cfg.kd_base_ori * (omega_des - omega_cur)
        w_base = torch.as_tensor(cfg.w_base_ori, dtype=dtype, device=device)
        _add_task(J_base_ang, a_ori, w_base)
    eye = lambda k: torch.eye(k, dtype=dtype, device=device).unsqueeze(0)
    H[:, :nv, :nv] += cfg.reg_vdot * eye(nv); H[:, nv:, nv:] += cfg.reg_W * eye(12)
    H = 0.5 * (H + H.transpose(1, 2)) + cfg.reg_psd * eye(nz)
    return BatchedQP(H=H, g=g, A_eq=A_eq, b_eq=b_eq, G=G, b=b)

def recover_tau(z, dyn, cfg, nv):
    adof = dyn["actuated_dof"]; JcT = _contact_transpose(dyn, nv)
    tau_fric = (dyn["tau_fric_coeff"] * torch.tanh(dyn["qvel"] / cfg.fric_eps)
                if cfg.friction_ff else torch.zeros_like(dyn["qvel"]))
    Mv = torch.bmm(dyn["M"], z[:, :nv].unsqueeze(-1)).squeeze(-1)
    JcTW = torch.bmm(JcT, z[:, nv:].unsqueeze(-1)).squeeze(-1)
    return (Mv + dyn["h"] + tau_fric - JcTW)[:, adof]
