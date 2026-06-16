"""WBC dynamics extraction from MuJoCo (no Pinocchio). Verified vs mujoco 3.6.0.

Caller MUST run mj_step1 (or mj_forward) on `data` before extract_dynamics, so
data.qM / qfrc_bias / qfrc_passive / xpos / xmat / subtree_com are current.
"""
import numpy as np
import mujoco
import torch


def make_handles(model, index_maps):
    """Lightweight per-run handles for extraction (body ids + sole point + actuated dofs)."""
    feet = index_maps["feet"]
    actuated_dof = np.array([j["dofadr"] for j in index_maps["actuated_joints"]], dtype=int)
    return {
        "base_body": index_maps["base_body_id"],
        "foot_L": feet["left"]["body_id"], "foot_R": feet["right"]["body_id"],
        "hand_L": index_maps["hands"]["left"]["body_id"],
        "hand_R": index_maps["hands"]["right"]["body_id"],
        "sole_local": feet["left"]["sole_local"],   # identical L/R in foot frame
        "x_half": feet["left"]["x_half"], "y_half": feet["left"]["y_half"],
        "actuated_dof": actuated_dof,
    }


def _body_jac6(model, data, bid, point_local=None):
    """6xnv [jacp; jacr] (linear top, angular bottom) at a body-local point, + world point."""
    nv = model.nv
    jacp = np.zeros((3, nv)); jacr = np.zeros((3, nv))
    if point_local is None:
        pw = data.xpos[bid].copy()
    else:
        pw = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ np.asarray(point_local, float)
    mujoco.mj_jac(model, data, jacp, jacr, pw, bid)
    return np.vstack([jacp, jacr]), pw


def extract_dynamics(model, data, handles):
    nv = model.nv
    M = np.zeros((nv, nv)); mujoco.mj_fullM(model, M, data.qM)
    h = data.qfrc_bias.copy() - data.qfrc_passive.copy()
    Jfoot_L, pL = _body_jac6(model, data, handles["foot_L"], handles["sole_local"])
    Jfoot_R, pR = _body_jac6(model, data, handles["foot_R"], handles["sole_local"])
    Jcom = np.zeros((3, nv)); mujoco.mj_jacSubtreeCom(model, data, Jcom, handles["base_body"])
    Jhand_L, hL = _body_jac6(model, data, handles["hand_L"])
    Jhand_R, hR = _body_jac6(model, data, handles["hand_R"])
    bq_wxyz = data.qpos[3:7].copy()                 # base orientation (wxyz)
    base_quat_xyzw = np.array([bq_wxyz[1], bq_wxyz[2], bq_wxyz[3], bq_wxyz[0]])
    qhL = np.zeros(4); mujoco.mju_mat2Quat(qhL, data.xmat[handles["hand_L"]])  # wxyz
    qhR = np.zeros(4); mujoco.mju_mat2Quat(qhR, data.xmat[handles["hand_R"]])
    to_xyzw = lambda q: np.array([q[1], q[2], q[3], q[0]])
    return {
        "M": M, "h": h, "tau_fric_coeff": model.dof_frictionloss.copy(),
        "Jfoot_L": Jfoot_L, "Jfoot_R": Jfoot_R,
        "foot_L_world": pL, "foot_R_world": pR,
        "Jcom": Jcom, "com": data.subtree_com[handles["base_body"]].copy(),
        "Jhand_L": Jhand_L, "Jhand_R": Jhand_R,
        "hand_L_world": hL, "hand_R_world": hR,
        "hand_L_quat_xyzw": to_xyzw(qhL), "hand_R_quat_xyzw": to_xyzw(qhR),
        "base_quat_xyzw": base_quat_xyzw,
        "actuated_dof": handles["actuated_dof"],
    }


class CpuDynamics:
    """B=1 dynamics: the validated plain-MuJoCo extraction wrapped as a (1,…) torch dict (f64)."""
    def __init__(self, model, index_maps, dtype=torch.float64):
        self.model = model; self.handles = make_handles(model, index_maps); self.dtype = dtype
        self._adof = torch.as_tensor(self.handles["actuated_dof"], dtype=torch.long)

    def extract(self, data):
        raw = extract_dynamics(self.model, data, self.handles)  # existing numpy dict
        t = lambda a: torch.as_tensor(a, dtype=self.dtype).unsqueeze(0)
        out = {k: t(raw[k]) for k in ("M", "h", "Jfoot_L", "Jfoot_R", "foot_L_world",
               "foot_R_world", "Jcom", "com", "Jhand_L", "Jhand_R", "hand_L_world",
               "hand_R_world", "hand_L_quat_xyzw", "hand_R_quat_xyzw", "base_quat_xyzw")}
        out["qvel"] = torch.as_tensor(data.qvel.copy(), dtype=self.dtype).unsqueeze(0)
        out["tau_fric_coeff"] = torch.as_tensor(raw["tau_fric_coeff"], dtype=self.dtype).unsqueeze(0)
        out["actuated_dof"] = self._adof
        out["x_half"] = self.handles["x_half"]; out["y_half"] = self.handles["y_half"]
        return out
