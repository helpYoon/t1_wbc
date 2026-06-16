"""Backend-agnostic joint command + the MuJoCo apply backend."""
from dataclasses import dataclass
import numpy as np


@dataclass
class JointCommand:
    """Per-actuated-joint command, numpy (nu,) arrays in MuJoCo actuator order."""
    q_des: np.ndarray; qd_des: np.ndarray
    kp: np.ndarray; kd: np.ndarray
    tau_ff: np.ndarray

    def torque(self, q, qd):
        """The effective actuator law (both sim and Booster kCustom), unclipped:
        tau = kp(q_des-q) + kd(qd_des-qd) + tau_ff. Callers clip to their own bounds."""
        return self.kp * (self.q_des - q) + self.kd * (self.qd_des - qd) + self.tau_ff


class MuJoCoBackend:
    """CPU single-sim apply (B=1): runs the kCustom law and writes clipped torque to data.ctrl."""
    def __init__(self, model):
        self.model = model
        self.lo = model.actuator_ctrlrange[:, 0].copy(); self.hi = model.actuator_ctrlrange[:, 1].copy()

    def apply(self, cmd, data):
        nu = self.model.nu
        q = data.qpos[7:7+nu]; qd = data.qvel[6:6+nu]
        data.ctrl[:] = np.clip(cmd.torque(q, qd), self.lo, self.hi)
