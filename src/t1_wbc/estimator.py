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
        # sole point in the foot frame. index_maps stores the foot collision-box
        # *center* (geom_pos); the actual sole is the box bottom face, so drop it by
        # the box half-height (geom_size[2]) to pin the true contact surface to ground.
        _foot_box = self._foot_contact_box_id(self.foot_L)
        self.sole_local = np.asarray(index_maps["feet"]["left"]["sole_local"], dtype=np.float64).copy()
        self.sole_local[2] -= float(model.geom_size[_foot_box, 2])
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

    def _foot_contact_box_id(self, foot_body_id):
        m = self.model
        g0, gn = m.body_geomadr[foot_body_id], m.body_geomnum[foot_body_id]
        return next(g for g in range(g0, g0 + gn)
                    if m.geom_contype[g] != 0
                    and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX)

    def _foot_world_z(self, base_xyz, quat_xyzw, joint_q):
        d = self._scratch
        d.qpos[0:3] = base_xyz
        d.qpos[3:7] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]  # xyzw->wxyz
        d.qpos[7:7 + self.nu] = joint_q
        mujoco.mj_kinematics(self.model, d)
        zl = (d.xpos[self.foot_L] + d.xmat[self.foot_L].reshape(3, 3) @ self.sole_local)[2]
        zr = (d.xpos[self.foot_R] + d.xmat[self.foot_R].reshape(3, 3) @ self.sole_local)[2]
        return float(zl), float(zr)

    def update_base_pose_and_contacts(self, joint_q):
        joint_q = np.asarray(joint_q, dtype=np.float64)
        # base z: FK with base at origin, then drop so the lower foot sits at world z=0
        zl0, zr0 = self._foot_world_z([0.0, 0.0, 0.0], self._quat_xyzw, joint_q)
        base_z = -min(zl0, zr0)
        self._pos[2] = base_z
        # contact flags: foot world-z (with base at the recovered height) near ground
        zl = zl0 + base_z; zr = zr0 + base_z
        self._contacts = np.array([zl < self.contact_threshold_m,
                                   zr < self.contact_threshold_m])

    def update_odometer(self, x, y, theta, t):
        t = float(t)
        if self._xy0 is None:
            self._xy0 = np.array([float(x), float(y)])
        # rotate (odom - xy0) into the yaw-zeroed frame (use IMU yaw0; 0 if no IMU yet)
        yaw0 = self._yaw0 if self._yaw0 is not None else 0.0
        c, s = np.cos(-yaw0), np.sin(-yaw0)
        dx, dy = float(x) - self._xy0[0], float(y) - self._xy0[1]
        xy_world = np.array([c * dx - s * dy, s * dx + c * dy])
        self._pos[0], self._pos[1] = xy_world[0], xy_world[1]
        # complementary filter: high-pass IMU integration + low-pass odometer finite-diff
        g = np.array([0.0, 0.0, -9.81])
        Rwb = R.from_quat(self._quat_xyzw).as_matrix()          # body->world
        a_world = Rwb @ self._acc + g                           # acc includes gravity reaction
        v_odom = np.zeros(3)
        if self._t_odom is not None and t > self._t_odom:
            dt = t - self._t_odom
            v_odom[:2] = (xy_world - self._xy_world_prev) / dt
            alpha = self.comp_tau_s / (self.comp_tau_s + dt)
            v_world = alpha * (self._lin_vel_world() + a_world * dt) + (1.0 - alpha) * v_odom
            self._lin_vel = Rwb.T @ v_world                     # store body-frame (getLinearVelocityLocal)
        self._t_odom = t
        self._xy_world_prev = xy_world.copy()

    def _lin_vel_world(self):
        return R.from_quat(self._quat_xyzw).as_matrix() @ self._lin_vel

    def lin_vel(self):
        return self._lin_vel.copy()

    def position(self):
        return self._pos.copy()

    def contact_flags(self):
        return self._contacts.copy()
