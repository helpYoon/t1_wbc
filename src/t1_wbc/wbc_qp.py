"""Numpy WBC QP assembly (finite-only inequality rows) + torque recovery.
Verified to match the legacy torch formulation to machine precision and to solve == proxsuite."""
import numpy as np

from .solver import BatchedQP

def ori_error_world(q_des_xyzw, q_cur_xyzw):
    qd = np.asarray(q_des_xyzw, dtype=np.float64); qc = np.asarray(q_cur_xyzw, dtype=np.float64)
    cx, cy, cz, cw = -qc[0], -qc[1], -qc[2], qc[3]
    dx, dy, dz, dw = qd[0], qd[1], qd[2], qd[3]
    ew = dw*cw - dx*cx - dy*cy - dz*cz
    ex = dw*cx + dx*cw + dy*cz - dz*cy
    ey = dw*cy - dx*cz + dy*cw + dz*cx
    ez = dw*cz + dx*cy - dy*cx + dz*cw
    sgn = -1.0 if ew < 0 else 1.0
    ew, ex, ey, ez = ew*sgn, ex*sgn, ey*sgn, ez*sgn
    vnorm = np.sqrt(ex*ex + ey*ey + ez*ez)
    angle = 2.0 * np.arctan2(vnorm, ew)
    scale = angle / vnorm if vnorm > 1e-12 else 0.0
    return np.array([ex*scale, ey*scale, ez*scale])


def _contact_transpose(dyn, nv):
    JcT = np.zeros((nv, 12), dtype=np.float64)
    JcT[:, 0:6] = dyn["Jfoot_L"].T
    JcT[:, 6:12] = dyn["Jfoot_R"].T
    return JcT

def _finite_contact_rows(nz, nv, mu, x_half, y_half, fz_min):
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
    G = np.zeros((m, nz), dtype=np.float64); bvec = np.zeros(m, dtype=np.float64)
    for r, (idxvals, lc) in enumerate(specs):
        for col, val in idxvals: G[r, col] = -val      # -C z <= -lc
        bvec[r] = -lc
    return G, bvec

def _torque_rows(dyn, ctrlrange, nz, nv, JcT, tau_fric):
    adof = np.asarray(dyn["actuated_dof"]); nu = adof.shape[0]
    Tsel = np.zeros((nu, nz), dtype=np.float64)
    Tsel[:, :nv] = dyn["M"][adof, :]; Tsel[:, nv:] = -JcT[adof, :]
    tau_const = dyn["h"][adof] + tau_fric[adof]
    lo = ctrlrange[:, 0] - tau_const; hi = ctrlrange[:, 1] - tau_const
    return np.concatenate([Tsel, -Tsel], axis=0), np.concatenate([hi, -lo], axis=0)

def assemble_wbc_qp(dyn, targets, cfg, ctrlrange, nv, nu):
    nz = nv + 12
    adof = np.asarray(dyn["actuated_dof"])
    JcT = _contact_transpose(dyn, nv)
    A_base = np.zeros((6, nz), dtype=np.float64)
    A_base[:, :nv] = dyn["M"][:6, :]; A_base[:, nv:] = -JcT[:6, :]
    A_feet = np.zeros((12, nz), dtype=np.float64)
    A_feet[0:6, :nv] = dyn["Jfoot_L"]; A_feet[6:12, :nv] = dyn["Jfoot_R"]
    A_eq = np.concatenate([A_base, A_feet], axis=0)
    b_eq = np.concatenate([-dyn["h"][:6], np.zeros(12, dtype=np.float64)], axis=0)
    tau_fric = (dyn["tau_fric_coeff"] * np.tanh(dyn["qvel"] / cfg.fric_eps)
                if cfg.friction_ff else np.zeros_like(dyn["qvel"]))
    Gc, bc = _finite_contact_rows(nz, nv, cfg.mu, dyn["x_half"], dyn["y_half"], cfg.fz_min)
    Gt, bt = _torque_rows(dyn, ctrlrange, nz, nv, JcT, tau_fric)
    G = np.concatenate([Gc, Gt], axis=0); b = np.concatenate([bc, bt], axis=0)
    H = np.zeros((nz, nz), dtype=np.float64); g = np.zeros(nz, dtype=np.float64)

    def _add_task(J, a_des, w):
        # weighted operational-space task: min ||J vdot - a_des||^2_w  ->  H += Jᵀ W J ; g += -Jᵀ W a_des
        JWl = J.T * w
        H[:nv, :nv] += JWl @ J
        g[:nv] += -(JWl @ a_des)

    # posture / joint task (tracked -> q_ref weight w_track_joint, else home weight w_post)
    q_ref = targets.q_ref if targets.q_ref is not None else targets.q_home
    qd_ref = targets.qd_ref if targets.qd_ref is not None else np.zeros_like(targets.q_home)
    vdot_j = cfg.kp_post * (q_ref - targets.q_act) + cfg.kd_post * (qd_ref - targets.qd_act)
    w_j = np.where(targets.tracked, np.float64(cfg.w_track_joint), np.float64(cfg.w_post))
    # scatter the per-DOF posture weights onto H's diagonal / g (adof entries are unique)
    H[adof, adof] += w_j
    g[adof] += -(w_j * vdot_j)
    # CoM task
    Jcom = dyn["Jcom"]; Jcomv = Jcom @ dyn["qvel"]
    a_com = cfg.kp_com * (targets.com_ref - dyn["com"]) + cfg.kd_com * (-Jcomv)
    w_com = np.asarray(cfg.w_com, dtype=np.float64)
    _add_task(Jcom, a_com, w_com)

    # hand position (and optional orientation) tracking tasks
    w_hand = np.asarray(cfg.w_hand, dtype=np.float64)
    for pos_key, quat_t, vel_t, Jkey, world_key, cur_quat_key in (
        (targets.lh_pos, targets.lh_quat_xyzw, targets.lh_vel, "Jhand_L", "hand_L_world", "hand_L_quat_xyzw"),
        (targets.rh_pos, targets.rh_quat_xyzw, targets.rh_vel, "Jhand_R", "hand_R_world", "hand_R_quat_xyzw"),
    ):
        if pos_key is None:
            continue
        Jh = dyn[Jkey]                                   # (6,nv) [jacp; jacr]
        Jhv = Jh @ dyn["qvel"]                           # (6,)
        Jp = Jh[:3, :]
        vel_des = vel_t if vel_t is not None else np.zeros(3, dtype=np.float64)
        a_pos = cfg.kp_hand * (pos_key - dyn[world_key]) + cfg.kd_hand * (vel_des - Jhv[:3])
        _add_task(Jp, a_pos, w_hand)
        if quat_t is not None and cur_quat_key in dyn:    # optional hand orientation task
            Jr = Jh[3:6, :]
            a_ori_h = cfg.kp_hand * ori_error_world(quat_t, dyn[cur_quat_key]) \
                + cfg.kd_hand * (-Jhv[3:6])
            _add_task(Jr, a_ori_h, w_hand)
    # base orientation tracking task (constant world-frame angular selector = rows 3:6)
    if targets.base_quat_xyzw is not None:
        J_base_ang = np.zeros((3, nv), dtype=np.float64)
        J_base_ang[:, 3:6] = np.eye(3, dtype=np.float64)
        omega_cur = dyn["qvel"][3:6]
        omega_des = (targets.base_omega_world if targets.base_omega_world is not None
                     else np.zeros(3, dtype=np.float64))
        ori_err = ori_error_world(targets.base_quat_xyzw, dyn["base_quat_xyzw"])
        a_ori = cfg.kp_base_ori * ori_err + cfg.kd_base_ori * (omega_des - omega_cur)
        w_base = np.asarray(cfg.w_base_ori, dtype=np.float64)
        _add_task(J_base_ang, a_ori, w_base)
    H[:nv, :nv] += cfg.reg_vdot * np.eye(nv); H[nv:, nv:] += cfg.reg_W * np.eye(12)
    H = 0.5 * (H + H.T) + cfg.reg_psd * np.eye(nz)
    return BatchedQP(H=H, g=g, A_eq=A_eq, b_eq=b_eq, G=G, b=b)

def recover_tau(z, dyn, cfg, nv):
    adof = np.asarray(dyn["actuated_dof"]); JcT = _contact_transpose(dyn, nv)
    tau_fric = (dyn["tau_fric_coeff"] * np.tanh(dyn["qvel"] / cfg.fric_eps)
                if cfg.friction_ff else np.zeros_like(dyn["qvel"]))
    return (dyn["M"] @ z[:nv] + dyn["h"] + tau_fric - JcT @ z[nv:])[adof]
