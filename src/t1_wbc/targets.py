"""Batched WBC task targets (leading dim B). None disables that task."""
from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class Targets:
    q_home: torch.Tensor          # (B,nu)
    q_act: torch.Tensor           # (B,nu)
    qd_act: torch.Tensor          # (B,nu)
    com_ref: torch.Tensor         # (B,3)
    tracked: torch.Tensor         # (B,nu) bool
    q_ref: torch.Tensor | None = None        # (B,nu) tracked-joint reference
    qd_ref: torch.Tensor | None = None
    base_quat_xyzw: torch.Tensor | None = None   # (B,4) or None
    base_omega_world: torch.Tensor | None = None
    lh_pos: torch.Tensor | None = None
    lh_quat_xyzw: torch.Tensor | None = None     # None disables L-hand ori task
    lh_vel: torch.Tensor | None = None
    rh_pos: torch.Tensor | None = None
    rh_quat_xyzw: torch.Tensor | None = None
    rh_vel: torch.Tensor | None = None

def balance_targets(q_home, q_act, qd_act, com_ref):
    """Posture-hold-to-home + CoM-over-support; no base/hand tasks."""
    B, nu = q_home.shape
    return Targets(q_home=q_home, q_act=q_act, qd_act=qd_act, com_ref=com_ref,
                   tracked=torch.zeros(B, nu, dtype=torch.bool, device=q_home.device),
                   q_ref=q_home, qd_ref=torch.zeros_like(q_home))


def tracking_targets_from_refsample(refsample, q_act, qd_act, q_home, dtype=torch.float64):
    """Broadcast a single-robot numpy RefSample into a (1,…) torch Targets (B=1).

    Joint refs/tracked, CoM, base orientation, and hand pos/quat/vel are pulled from the
    RefSample; q_act/qd_act/q_home are the (1,nu) torch measured/home states. A hand-ori or
    base-ori target whose source quat is None disables that orientation task."""
    device = q_act.device
    v3 = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=dtype,
                                   device=device).reshape(1, 3)
    v4 = lambda a: (None if a is None else
                    torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=dtype,
                                    device=device).reshape(1, 4))
    vnu = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=dtype,
                                    device=device).reshape(1, -1)
    rs = refsample
    tracked = torch.as_tensor(np.asarray(rs.tracked, dtype=bool),
                              dtype=torch.bool, device=device).reshape(1, -1)
    return Targets(
        q_home=q_home, q_act=q_act, qd_act=qd_act,
        com_ref=v3(rs.com_ref), tracked=tracked,
        q_ref=vnu(rs.q_ref), qd_ref=vnu(rs.qd_ref),
        base_quat_xyzw=v4(rs.base_quat_xyzw), base_omega_world=v3(rs.base_omega_world),
        lh_pos=v3(rs.left_hand_pos), lh_quat_xyzw=v4(rs.left_hand_quat_xyzw),
        lh_vel=v3(rs.left_hand_vel),
        rh_pos=v3(rs.right_hand_pos), rh_quat_xyzw=v4(rs.right_hand_quat_xyzw),
        rh_vel=v3(rs.right_hand_vel),
    )
