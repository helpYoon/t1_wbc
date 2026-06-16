"""Floating-base state estimator for the Booster T1 — MuJoCo-FK port of the C++
StateEstimator. Reconstructs base pose/velocity/contacts from IMU (rpy/gyro/acc),
planar odometry (x,y,theta), and joint encoders. No SDK/torch — pure numpy + MuJoCo FK."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R


class StateEstimator:
    def __init__(self, model, index_maps, contact_threshold_m=0.01, comp_tau_s=0.05):
        self.model = model
        self.maps = index_maps
        self.nu = model.nu
        self.contact_threshold_m = float(contact_threshold_m)
        self.comp_tau_s = float(comp_tau_s)
        self._scratch = mujoco.MjData(model)
        self.base_body = index_maps["base_body_id"]
        self.foot_L = index_maps["feet"]["left"]["body_id"]
        self.foot_R = index_maps["feet"]["right"]["body_id"]
        self.sole_local = np.asarray(index_maps["feet"]["left"]["sole_local"], dtype=np.float64)
        # state
        self._yaw0 = None
        self._quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
        self._gyro = np.zeros(3)
        self._acc = np.array([0.0, 0.0, 9.81])
        self._t_imu = None
        self._xy0 = None
        self._pos = np.zeros(3)
        self._contacts = np.array([True, True])
        self._lin_vel = np.zeros(3)
        self._t_odom = None
        self._xy_world_prev = None

    def update_imu(self, rpy, gyro, acc, t):
        roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
        if self._yaw0 is None:
            self._yaw0 = yaw
        self._quat_xyzw = R.from_euler("ZYX", [yaw - self._yaw0, pitch, roll]).as_quat()
        self._gyro = np.asarray(gyro, dtype=np.float64)
        self._acc = np.asarray(acc, dtype=np.float64)
        self._t_imu = float(t)

    def quat_xyzw(self):
        return self._quat_xyzw.copy()

    def ang_vel(self):
        return self._gyro.copy()
