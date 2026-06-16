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
    w_post: float = 1.0             # weight for untracked (held) joints
    w_track_joint: float = 20.0     # weight for tracked joints (Phase 3)
    # hand task (Phase 3)
    kp_hand: float = 100.0
    kd_hand: float = 20.0
    w_hand: float = 10.0
    # base orientation task (Phase 3)
    kp_base_ori: float = 60.0
    kd_base_ori: float = 12.0
    w_base_ori: float = 5.0
    # regularization
    reg_vdot: float = 1e-3
    reg_W: float = 1e-5
    reg_psd: float = 1e-8           # symmetrizing jitter to keep the QP Hessian PD
    # joint-servo gains in the JointCommand (0 in sim = pure ID; nonzero on hardware)
    servo_kp: float = 0.0
    servo_kd: float = 0.0
    # timing
    time_scale: float = 5.0
    control_decimation: int = 1     # solve QP every k physics steps
    settle_seconds: float = 0.5
    settle_kp: float = 60.0         # PD-hold settle gains (shared by B=1 and batched paths)
    settle_kd: float = 6.0
    upright_z: float = 0.5          # base-height threshold for the "upright" summary flag
