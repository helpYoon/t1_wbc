"""Batched WarpDynamics extraction (verified vs CPU MuJoCo, T1).

B>1 dynamics from a mjwarp Model/Data -> (B,...) torch dict (f32). Every
``wp.array``/``wp.from_torch`` routes through ``handles["wp_device"]``; ``qM`` is
padded (B,48,48) and sliced to ``nv``; ``mjw.jac`` writes its outputs in place.

Drive between extracts: write ``wd.qpos``/``wd.qvel`` via ``wp.from_torch`` (tensor
on ``wp_device``), ``mjw.forward(wm, wd)``, then ``extract(wd)``.
"""
import numpy as np
import mujoco_warp as mjw
import warp as wp
import torch


def _foot_world_point(wd, bid, p_local):
    xpos = wp.to_torch(wd.xpos); xmat = wp.to_torch(wd.xmat)        # (B,nbody,3),(B,nbody,3,3)
    p = torch.as_tensor(p_local, dtype=torch.float32, device=xpos.device)
    return xpos[:, bid] + torch.einsum("bij,j->bi", xmat[:, bid], p)


def _pt_body(world_pts, bid, B, dev):
    pt = wp.from_torch(world_pts.contiguous(), dtype=wp.vec3f)      # tensor MUST be on warp device
    body = wp.array(np.full(B, bid, np.int32), dtype=wp.int32, device=dev)
    return pt, body


def _jac6(wm, wd, bid, p_local, h):
    nv, B, dev = h["nv"], h["nworld"], h["wp_device"]
    pw = (wp.to_torch(wd.xpos)[:, bid].contiguous() if p_local is None
          else _foot_world_point(wd, bid, p_local).contiguous())
    jacp = wp.zeros((B, 3, nv), dtype=wp.float32, device=dev)
    jacr = wp.zeros((B, 3, nv), dtype=wp.float32, device=dev)
    pt, body = _pt_body(pw, bid, B, dev)
    mjw.jac(wm, wd, jacp, jacr, pt, body)                          # IN-PLACE, returns None
    return torch.cat([wp.to_torch(jacp), wp.to_torch(jacr)], dim=1), pw


class WarpDynamics:
    """B>1 dynamics from a mjwarp Model/Data -> (B,...) torch dict (f32)."""
    def __init__(self, wm, handles):
        self.wm = wm; self.h = handles
        self._adof = torch.as_tensor(handles["actuated_dof"], dtype=torch.long, device=handles["wp_device"])
        self._fric = torch.as_tensor(handles["dof_frictionloss"], dtype=torch.float32,
                                     device=handles["wp_device"])

    def extract(self, wd):
        h = self.h; nv, nbody, B, dev = h["nv"], h["nbody"], h["nworld"], h["wp_device"]
        M = wp.to_torch(wd.qM)[:, :nv, :nv].clone()                 # padded (B,48,48) -> slice nv
        bias = (wp.to_torch(wd.qfrc_bias) - wp.to_torch(wd.qfrc_passive)).clone()
        Jfoot_L, pL = _jac6(self.wm, wd, h["foot_L"], h["sole_local"], h)
        Jfoot_R, pR = _jac6(self.wm, wd, h["foot_R"], h["sole_local"], h)
        Jhand_L, hL = _jac6(self.wm, wd, h["hand_L"], None, h)
        Jhand_R, hR = _jac6(self.wm, wd, h["hand_R"], None, h)
        xipos = wp.to_torch(wd.xipos)
        mass = torch.as_tensor(h["body_mass"], dtype=torch.float32, device=M.device)
        Jcom = torch.zeros((B, 3, nv), dtype=torch.float32, device=M.device)
        jacp = wp.zeros((B, 3, nv), dtype=wp.float32, device=dev)
        for bdy in range(nbody):
            if mass[bdy].item() == 0.0: continue
            pt, body = _pt_body(xipos[:, bdy].contiguous(), bdy, B, dev)
            mjw.jac(self.wm, wd, jacp, None, pt, body)
            Jcom += mass[bdy] * wp.to_torch(jacp)
        Jcom /= float(h["total_mass"])
        com = wp.to_torch(wd.subtree_com)[:, h["base_body"]].clone()
        return dict(M=M, h=bias, Jfoot_L=Jfoot_L, Jfoot_R=Jfoot_R, foot_L_world=pL, foot_R_world=pR,
                    Jhand_L=Jhand_L, Jhand_R=Jhand_R, hand_L_world=hL, hand_R_world=hR,
                    Jcom=Jcom, com=com, qvel=wp.to_torch(wd.qvel).clone(),
                    tau_fric_coeff=self._fric.unsqueeze(0).expand(B, -1),
                    actuated_dof=self._adof, x_half=h["x_half"], y_half=h["y_half"])
