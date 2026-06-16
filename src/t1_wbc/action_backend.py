"""Backend-agnostic joint command + action backends (MuJoCo now; Booster SDK later)."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np, torch


@dataclass
class JointCommand:
    """Per-actuated-joint command, batched (B,nu) tensors in MuJoCo actuator order.
    Effective law (both sim and Booster kCustom): tau = kp(q_des-q)+kd(qd_des-qd)+tau_ff."""
    q_des: torch.Tensor; qd_des: torch.Tensor   # (B,nu)
    kp: torch.Tensor; kd: torch.Tensor
    tau_ff: torch.Tensor


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
        q = torch.as_tensor(data.qpos[7:7+nu], dtype=cmd.q_des.dtype)
        qd = torch.as_tensor(data.qvel[6:6+nu], dtype=cmd.q_des.dtype)
        tau = (cmd.kp*(cmd.q_des - q) + cmd.kd*(cmd.qd_des - qd) + cmd.tau_ff)[0].cpu().numpy()
        data.ctrl[:] = np.clip(tau, self.lo, self.hi)


class BoosterSdkBackend(ActionBackend):
    """FUTURE (§15 of spec): map JointCommand 1:1 onto MotorCmd and Write(LowCmd).
    Requires: kPrepare->kCustom handshake, SERIAL cmd_type, SDK joint-index remap,
    weight ramp, 500 Hz stream, watchdog. Not implemented for the sim milestone."""
    def apply(self, cmd, data):
        raise NotImplementedError("BoosterSdkBackend is a future hardware path (spec §15).")
