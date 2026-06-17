"""Tunable configuration for the T1 whole-body QP controller."""
from dataclasses import dataclass, field
import numpy as np

from ._assets import asset

T1_XML = asset("robot", "t1.xml")
PKL = asset("motion_plan.pkl")


@dataclass
class WBCConfig:
    # paths
    xml: str = T1_XML
    motion: str = PKL
    segments: tuple | None = None   # None = full motion; else iterable of segment indices to track
    # contact / friction
    mu: float = 0.6
    fz_min: float = 0.0
    fric_eps: float = 0.05          # tanh smoothing for Coulomb friction feedforward
    friction_ff: bool = True        # enable the Coulomb friction feedforward term
    # CoM / balance task (safety-critical, highest weight)
    kp_com: float = 80.0
    kd_com: float = 18.0
    w_com: np.ndarray = field(default_factory=lambda: np.array([300.0, 300.0, 50.0]))
    # joint task (posture = untracked-at-home; tracking weight bumps tracked joints)
    kp_post: float = 120.0
    kd_post: float = 12.0
    w_post: float = 20.0            # hold weight for untracked joints (incl. the lateral
                                    # balance joints hip_roll/ankle_roll); was 1.0 -> they
                                    # gave up and the robot rolled. Now held as firmly as the
                                    # tracked joints (mirrors MPC holding Hip_Roll >= Hip_Pitch).
    w_track_joint: float = 20.0     # weight for tracked joints (Phase 3)
    # hand task (Phase 3)
    kp_hand: float = 100.0
    kd_hand: float = 20.0
    w_hand: float = 10.0
    # base orientation task — PRIMARY balance term, mirroring t1_controller MPC's trunk
    # pitch/roll cost (Q=20, equal pitch+roll). Was 5 (far too weak -> robot rolled over).
    kp_base_ori: float = 60.0
    kd_base_ori: float = 12.0
    w_base_ori: float = 50.0
    # regularization
    reg_vdot: float = 1e-3
    reg_W: float = 1e-5
    reg_psd: float = 1e-8           # symmetrizing jitter to keep the QP Hessian PD
    # joint-servo gains in the JointCommand (0 in sim = pure ID; nonzero on hardware)
    servo_kp: float = 0.0
    servo_kd: float = 0.0
    torque_limit_scale: float = 1.0   # (0,1]; reduce for conservative on-robot bring-up
    tau_pd_margin: float = 0.0        # Nm headroom reserved for the firmware PD on top of tau_ff
    tau_slew_max: float = 80.0        # max |delta tau| per control tick (Nm)
    ramp_seconds: float = 2.0         # weight-ramp blend hold-pose -> WBC
    watchdog_timeout_s: float = 0.05  # LowState staleness -> safe hold
    # timing
    time_scale: float = 5.0
    control_decimation: int = 1     # solve QP every k physics steps
    control_period: float = 0.02    # WBC solve cadence (s); 0.02 = 50Hz, matches SDK example
    publish_period: float = 0.005   # LowCmd stream cadence (s); 0.005 = 200Hz, decoupled from solve
    settle_seconds: float = 0.5
    settle_kp: float = 60.0         # PD-hold settle gains (shared by B=1 and batched paths)
    settle_kd: float = 6.0
    upright_z: float = 0.5          # base-height threshold for the "upright" summary flag


def effective_ctrlrange(model, cfg, reserve_margin=False):
    """Per-actuator torque [lo, hi] after the conservative `torque_limit_scale`.
    With `reserve_margin`, also pull each bound in by `tau_pd_margin` to leave headroom
    for the firmware PD (the in-QP torque bound uses this; the SafetyLayer clamps to the
    unmargined scaled range)."""
    cr = np.asarray(model.actuator_ctrlrange, dtype=np.float64) * cfg.torque_limit_scale
    if reserve_margin:
        cr[:, 0] = cr[:, 0] + cfg.tau_pd_margin
        cr[:, 1] = cr[:, 1] - cfg.tau_pd_margin
    return cr


_SERVO_GROUPS = {  # (kp, kd) by joint group, from the T1 task.info joint_pd_gains schedule
    "head_arm": (20.0, 0.5), "waist_hip_knee": (200.0, 5.0), "ankle": (50.0, 3.0),
}
def servo_gains_for(index_maps):
    """Per-MuJoCo-joint (kp, kd) from the T1 task.info joint_pd_gains schedule."""
    n2i = index_maps["name_to_act_index"]
    nu = len(n2i)
    kp = np.zeros(nu); kd = np.zeros(nu)
    for name, i in n2i.items():
        if "Ankle" in name:
            g = _SERVO_GROUPS["ankle"]
        elif ("Hip" in name) or ("Knee" in name) or (name == "Waist"):
            g = _SERVO_GROUPS["waist_hip_knee"]
        else:                                  # head + all arm joints
            g = _SERVO_GROUPS["head_arm"]
        kp[i], kd[i] = g
    return kp, kd
