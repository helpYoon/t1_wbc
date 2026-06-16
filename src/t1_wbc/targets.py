"""Numpy WBC task targets (single robot, no batch dim). None disables that task."""
from dataclasses import dataclass
import numpy as np

@dataclass
class Targets:
    q_home: np.ndarray            # (nu,)
    q_act: np.ndarray             # (nu,)
    qd_act: np.ndarray            # (nu,)
    com_ref: np.ndarray           # (3,)
    tracked: np.ndarray           # (nu,) bool
    q_ref: np.ndarray | None = None          # (nu,) tracked-joint reference
    qd_ref: np.ndarray | None = None
    base_quat_xyzw: np.ndarray | None = None     # (4,) or None
    base_omega_world: np.ndarray | None = None
    lh_pos: np.ndarray | None = None
    lh_quat_xyzw: np.ndarray | None = None       # None disables L-hand ori task
    lh_vel: np.ndarray | None = None
    rh_pos: np.ndarray | None = None
    rh_quat_xyzw: np.ndarray | None = None
    rh_vel: np.ndarray | None = None

def balance_targets(q_home, q_act, qd_act, com_ref):
    """Posture-hold-to-home + CoM-over-support; no base/hand tasks."""
    return Targets(q_home=q_home, q_act=q_act, qd_act=qd_act, com_ref=com_ref,
                   tracked=np.zeros(len(q_home), dtype=bool),
                   q_ref=q_home, qd_ref=np.zeros_like(q_home))


def tracking_targets_from_refsample(refsample, q_act, qd_act, q_home):
    """Build a single-robot numpy Targets from a numpy RefSample.

    Joint refs/tracked, CoM, base orientation, and hand pos/quat/vel are pulled from the
    RefSample; q_act/qd_act/q_home are the (nu,) measured/home states. A hand-ori or
    base-ori target whose source quat is None disables that orientation task."""
    v3 = lambda a: np.asarray(a, dtype=np.float64).reshape(3)
    v4 = lambda a: (None if a is None else np.asarray(a, dtype=np.float64).reshape(4))
    vnu = lambda a: np.asarray(a, dtype=np.float64).reshape(-1)
    rs = refsample
    tracked = np.asarray(rs.tracked, dtype=bool).reshape(-1)
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
