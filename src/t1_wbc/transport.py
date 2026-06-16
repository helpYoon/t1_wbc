"""Transport abstraction: LowState in / LowCmd out. SimTransport synthesizes SDK-shaped
sensor data from a MuJoCo sim and applies commands; SdkTransport (Stage 2b) wraps the SDK."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R


@dataclass
class LowState:
    imu_rpy: np.ndarray      # (3,) roll,pitch,yaw
    imu_gyro: np.ndarray     # (3,) body angular velocity
    imu_acc: np.ndarray      # (3,) body specific force (INCLUDES gravity reaction)
    joint_q: np.ndarray      # (nu,)
    joint_dq: np.ndarray     # (nu,)
    odom_xytheta: np.ndarray # (3,) planar x,y,theta


# A LowCmd is structurally identical to the WBC's JointCommand (q_des, qd_des, kp, kd,
# tau_ff in MuJoCo actuator order) — alias it so the controller output flows straight to
# the transport with no rebuild.
from .action_backend import JointCommand as LowCmd


class Transport(ABC):
    @abstractmethod
    def read_lowstate(self) -> LowState: ...
    @abstractmethod
    def write_lowcmd(self, cmd: LowCmd) -> None: ...
    def state_age(self) -> float:
        """Seconds since the last LowState was received (0.0 when synthesized in sim)."""
        return 0.0


class SimTransport(Transport):
    """SDK-shaped sensors from a MuJoCo sim; commands applied via the kCustom law.
    Joints stay in MuJoCo actuator order (no SDK remap in sim)."""
    def __init__(self, model, data, noise=None):
        self.model = model; self.data = data; self.noise = noise
        self.nu = model.nu
        self.lo = model.actuator_ctrlrange[:, 0].copy()
        self.hi = model.actuator_ctrlrange[:, 1].copy()
        self.g = np.array([0.0, 0.0, -9.81])

    def read_lowstate(self) -> LowState:
        d = self.data; nu = self.nu
        bq = d.qpos[3:7]                                   # wxyz
        Rwb = R.from_quat([bq[1], bq[2], bq[3], bq[0]]).as_matrix()  # body->world
        yaw, pitch, roll = R.from_matrix(Rwb).as_euler("ZYX")
        rpy = np.array([roll, pitch, yaw])
        gyro = d.qvel[3:6].copy()                          # base angular vel (body frame)
        acc = Rwb.T @ (-self.g)                            # gravity reaction only (quasi-static)
        q = d.qpos[7:7 + nu].copy(); dq = d.qvel[6:6 + nu].copy()
        odom = np.array([d.qpos[0], d.qpos[1], yaw])
        ls = LowState(imu_rpy=rpy, imu_gyro=gyro, imu_acc=acc,
                      joint_q=q, joint_dq=dq, odom_xytheta=odom)
        if self.noise is not None:
            ls.imu_gyro = ls.imu_gyro + self.noise * np.random.randn(3)
            ls.odom_xytheta = ls.odom_xytheta + self.noise * np.random.randn(3)
        return ls

    def write_lowcmd(self, cmd: LowCmd) -> None:
        d = self.data; nu = self.nu
        q = d.qpos[7:7 + nu]; dq = d.qvel[6:6 + nu]
        d.ctrl[:] = np.clip(cmd.torque(q, dq), self.lo, self.hi)
