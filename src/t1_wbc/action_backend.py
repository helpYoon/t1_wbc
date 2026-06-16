"""Backend-agnostic joint command + action backends (MuJoCo now; Booster SDK later)."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class JointCommand:
    """Per-actuated-joint command, numpy (nu,) arrays in MuJoCo actuator order.
    Effective law (both sim and Booster kCustom): tau = kp(q_des-q)+kd(qd_des-qd)+tau_ff."""
    q_des: np.ndarray; qd_des: np.ndarray
    kp: np.ndarray; kd: np.ndarray
    tau_ff: np.ndarray


class ActionBackend(ABC):
    @abstractmethod
    def apply(self, cmd, data):  # cmd: a single batched JointCommand
        ...


class MuJoCoBackend(ActionBackend):
    """CPU single-sim apply (B=1). The Warp batched apply lives in controller.py (Phase 2)."""
    def __init__(self, model):
        self.model = model
        self.lo = model.actuator_ctrlrange[:, 0].copy(); self.hi = model.actuator_ctrlrange[:, 1].copy()

    def apply(self, cmd, data):
        nu = self.model.nu
        q = data.qpos[7:7+nu]; qd = data.qvel[6:6+nu]
        tau = cmd.kp*(cmd.q_des - q) + cmd.kd*(cmd.qd_des - qd) + cmd.tau_ff
        data.ctrl[:] = np.clip(tau, self.lo, self.hi)


class BoosterSdkBackend(ActionBackend):
    """FUTURE (§15 of spec): map JointCommand 1:1 onto MotorCmd and Write(LowCmd).
    Requires: kPrepare->kCustom handshake, SERIAL cmd_type, SDK joint-index remap,
    weight ramp, 500 Hz stream, watchdog. Not implemented for the sim milestone."""
    def apply(self, cmd, data):
        raise NotImplementedError("BoosterSdkBackend is a future hardware path (spec §15).")
