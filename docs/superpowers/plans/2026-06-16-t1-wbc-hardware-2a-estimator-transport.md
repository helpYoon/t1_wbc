# t1_wbc Hardware Stage 2a — Estimator + Transport + Estimator-in-the-Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the WBC run on an estimated floating-base state (IMU + planar odometry + joint encoders) instead of MuJoCo ground-truth, and verify in sim that it still tracks the motion upright.

**Architecture:** A pure `StateEstimator` (MuJoCo-FK port of the C++ one) reconstructs base pose/velocity/contacts from sensor inputs; a state-assembly step writes (estimated base + measured joints) into a scratch `MjData`, `mj_forward`s it, and reuses the existing numpy `CpuDynamics.extract` unchanged; a `Transport` abstraction (`SimTransport` now, `SdkTransport` in 2b) supplies sensor data and applies commands so one control loop runs in sim and (later) on hardware.

**Tech Stack:** Python, numpy, mujoco==3.6.0, scipy (`Rotation`), proxsuite, pytest. Repo: `/home/yoonwoo/humanoid_mpc_ws/src/t1_wbc` (torch-free, numpy/proxsuite foundation from Plan 1).

**Spec:** `docs/superpowers/specs/2026-06-16-t1-wbc-hardware-2a-estimator-transport-design.md`.

**Branch:** Execute on a feature branch off `main` (e.g. `hw-2a-estimator`). The repo venv `.venv/bin/python` is torch-free. **Run pytest with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`** (stray ROS `launch_testing` plugin else crashes collection).

**Commit with:** `git -c user.name='t1-wbc' -c user.email='t1-wbc@local' commit ...`.

---

## File Structure

```
src/t1_wbc/
  estimator.py        # NEW — StateEstimator (orientation/ang-vel, FK base-z + contacts, odom xy + lin-vel)
  transport.py        # NEW — LowState, LowCmd, Transport (ABC), SimTransport
  controller.py       # MODIFY — attach_estimator(), step_track_estimated()
  run.py              # MODIFY — run_track_estimated(), --mode track-est
tests/t1_wbc/
  test_estimator.py        # NEW — Tasks 1-3
  test_transport.py        # NEW — Task 4
  test_state_assembly.py   # NEW — Task 5
  test_track_estimated.py  # NEW — Tasks 6-7
```

**Existing interfaces this plan uses (already in the repo):**
- `model.build_index_maps(model)` → dict with `base_body_id`, `feet["left"|"right"]["body_id"|"sole_local"|"x_half"|"y_half"]`, `name_to_act_index`, etc.
- `model.load_t1_model(xml=None)` → `(model, data)`.
- `dynamics.CpuDynamics(model, index_maps).extract(data)` → numpy dict (`M, h, Jcom, com, Jfoot_L/R, hand_*_world, base_quat_xyzw, qvel, actuated_dof, x_half, y_half, …`).
- `controller.WBController`: `reset(data)`, `settle(data)`, `attach_reference(ref)`, `_act_state(data)`, `_solve_to_cmd(d, tg, q_act, qd_act) -> (cmd, z, ok, tau_ff)`, fields `model, cfg, nu, nv, dt, q_home, dyn, ref`.
- `reference.ReferenceTrajectory(model, index_maps, q_home, cfg, x0, y0, yaw0)` with `.sample(t) -> RefSample` and `.duration`.
- `targets.tracking_targets_from_refsample(rs, q_act, qd_act, q_home) -> Targets`.
- `action_backend.JointCommand(q_des, qd_des, kp, kd, tau_ff)` (numpy `(nu,)` fields).

---

## Task 1: Estimator — orientation (yaw-zeroed) + angular velocity

**Files:** Create `src/t1_wbc/estimator.py`; Test `tests/t1_wbc/test_estimator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/t1_wbc/test_estimator.py
import numpy as np
from scipy.spatial.transform import Rotation as R
from t1_wbc.model import load_t1_model, build_index_maps
from t1_wbc.estimator import StateEstimator

def _est():
    model, _ = load_t1_model()
    return StateEstimator(model, build_index_maps(model))

def test_initial_yaw_is_zeroed():
    est = _est()
    est.update_imu(rpy=[0.0, 0.1, 1.3], gyro=[0, 0, 0], acc=[0, 0, 9.81], t=0.0)
    q = est.quat_xyzw()                       # xyzw
    yaw, pitch, roll = R.from_quat(q).as_euler("ZYX")
    assert abs(yaw) < 1e-9                     # startup yaw subtracted
    assert abs(pitch - 0.1) < 1e-9 and abs(roll - 0.0) < 1e-9

def test_yaw_is_relative_to_first_sample():
    est = _est()
    est.update_imu([0, 0, 1.3], [0, 0, 0], [0, 0, 9.81], 0.0)
    est.update_imu([0, 0, 1.3 + 0.4], [0, 0, 0], [0, 0, 9.81], 0.002)
    yaw = R.from_quat(est.quat_xyzw()).as_euler("ZYX")[0]
    assert abs(yaw - 0.4) < 1e-9

def test_ang_vel_passthrough():
    est = _est()
    est.update_imu([0, 0, 0], gyro=[0.1, -0.2, 0.3], acc=[0, 0, 9.81], t=0.0)
    np.testing.assert_allclose(est.ang_vel(), [0.1, -0.2, 0.3])
```

- [ ] **Step 2: Run, expect FAIL** (module missing):
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/t1_wbc/test_estimator.py -q`

- [ ] **Step 3: Create `estimator.py` with the orientation/ang-vel core**

```python
# src/t1_wbc/estimator.py
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
```

- [ ] **Step 4: Run, expect PASS** (3 tests):
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/t1_wbc/test_estimator.py -q`

- [ ] **Step 5: Commit**
```bash
git add src/t1_wbc/estimator.py tests/t1_wbc/test_estimator.py
git -c user.name='t1-wbc' -c user.email='t1-wbc@local' commit -m "feat(estimator): orientation (yaw-zeroed) + angular velocity"
```

---

## Task 2: Estimator — base z via FK foot-pinning + contact flags

**Files:** Modify `src/t1_wbc/estimator.py`; Test `tests/t1_wbc/test_estimator.py`

- [ ] **Step 1: Write the failing test** (append)

```python
# append to tests/t1_wbc/test_estimator.py
import mujoco
def test_fk_base_z_pins_lower_foot_to_ground():
    model, data = load_t1_model()
    mujoco.mj_resetDataKeyframe(model, data, 0); mujoco.mj_forward(model, data)
    true_base_z = float(data.qpos[2])
    bq = data.qpos[3:7]                                 # wxyz
    rpy = R.from_quat([bq[1], bq[2], bq[3], bq[0]]).as_euler("ZYX")[::-1]  # roll,pitch,yaw
    jq = data.qpos[7:7 + model.nu].copy()
    est = StateEstimator(model, build_index_maps(model))
    est.update_imu(rpy, [0, 0, 0], [0, 0, 9.81], 0.0)
    est.update_base_pose_and_contacts(jq)
    # home keyframe has both feet flat on the ground -> recovered base z ~= true base z
    assert abs(est.position()[2] - true_base_z) < 5e-3
    assert est.contact_flags().tolist() == [True, True]

def test_contact_flag_clears_when_foot_lifted():
    model, data = load_t1_model()
    mujoco.mj_resetDataKeyframe(model, data, 0); mujoco.mj_forward(model, data)
    jq = data.qpos[7:7 + model.nu].copy()
    maps = build_index_maps(model)
    li = maps["name_to_act_index"]["Left_Knee_Pitch"]
    jq[li] += 0.6                                        # bend left knee -> lift left foot
    est = StateEstimator(model, maps)
    est.update_imu([0, 0, 0], [0, 0, 0], [0, 0, 9.81], 0.0)
    est.update_base_pose_and_contacts(jq)
    assert est.contact_flags().tolist() == [False, True] or est.contact_flags().tolist() == [True, False]
```

- [ ] **Step 2: Run, expect FAIL** (`update_base_pose_and_contacts`/`position` missing).

- [ ] **Step 3: Add FK base-z + contacts to `estimator.py`**

```python
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

    def position(self):
        return self._pos.copy()

    def contact_flags(self):
        return self._contacts.copy()
```

- [ ] **Step 4: Run, expect PASS** (5 tests total).

- [ ] **Step 5: Commit**
```bash
git add src/t1_wbc/estimator.py tests/t1_wbc/test_estimator.py
git -c user.name='t1-wbc' -c user.email='t1-wbc@local' commit -m "feat(estimator): FK base-z foot-pinning + contact flags"
```

---

## Task 3: Estimator — odometer x,y (yaw-zeroed) + complementary-filter linear velocity

**Files:** Modify `src/t1_wbc/estimator.py`; Test `tests/t1_wbc/test_estimator.py`

- [ ] **Step 1: Write the failing test** (append)

```python
# append to tests/t1_wbc/test_estimator.py
def test_odometer_xy_centered_and_yaw_zeroed():
    est = _est()
    est.update_imu([0, 0, 0.5], [0, 0, 0], [0, 0, 9.81], 0.0)   # yaw0 = 0.5
    est.update_odometer(2.0, 0.0, 0.5, 0.0)                     # first odom -> origin
    np.testing.assert_allclose(est.position()[:2], [0.0, 0.0], atol=1e-9)
    # move +1 m along world-x; in the yaw-zeroed (-0.5 rad) frame it rotates
    est.update_odometer(2.0 + np.cos(0.0), 0.0 + np.sin(0.0), 0.5, 0.01)
    exp = R.from_euler("z", -0.5).apply([1.0, 0.0, 0.0])[:2]
    np.testing.assert_allclose(est.position()[:2], exp, atol=1e-9)

def test_lin_vel_converges_to_constant_odometer_velocity():
    est = _est()
    est.update_imu([0, 0, 0], [0, 0, 0], [0, 0, 9.81], 0.0)     # acc = gravity reaction only
    est.update_odometer(0.0, 0.0, 0.0, 0.0)
    dt, v = 0.002, 0.3
    for k in range(1, 400):
        t = k * dt
        est.update_imu([0, 0, 0], [0, 0, 0], [0, 0, 9.81], t)
        est.update_odometer(v * t, 0.0, 0.0, t)                 # constant 0.3 m/s in world x
    np.testing.assert_allclose(est.lin_vel()[:2], [v, 0.0], atol=2e-2)
```

- [ ] **Step 2: Run, expect FAIL** (`update_odometer`/`lin_vel` missing).

- [ ] **Step 3: Add odometer xy + complementary-filter lin-vel to `estimator.py`**

```python
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
```

- [ ] **Step 4: Run, expect PASS** (7 tests). If `test_lin_vel...` is just outside `atol=2e-2`, the filter is converging correctly but slowly — confirm it's monotonically approaching `v`; do NOT loosen below 3e-2 (a larger miss means the `a_world`/blend sign is wrong, a real bug).

- [ ] **Step 5: Commit**
```bash
git add src/t1_wbc/estimator.py tests/t1_wbc/test_estimator.py
git -c user.name='t1-wbc' -c user.email='t1-wbc@local' commit -m "feat(estimator): odometer xy (yaw-zeroed) + complementary-filter linear velocity"
```

---

## Task 4: Transport — LowState/LowCmd + SimTransport

**Files:** Create `src/t1_wbc/transport.py`; Test `tests/t1_wbc/test_transport.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/t1_wbc/test_transport.py
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
from t1_wbc.model import load_t1_model
from t1_wbc.transport import SimTransport, LowState, LowCmd

def _sim():
    model, data = load_t1_model()
    mujoco.mj_resetDataKeyframe(model, data, 0); mujoco.mj_forward(model, data)
    return model, data

def test_read_lowstate_shapes_and_joint_values():
    model, data = _sim()
    ls = SimTransport(model, data).read_lowstate()
    assert isinstance(ls, LowState)
    assert ls.imu_rpy.shape == (3,) and ls.imu_gyro.shape == (3,) and ls.imu_acc.shape == (3,)
    assert ls.joint_q.shape == (model.nu,) and ls.joint_dq.shape == (model.nu,)
    assert ls.odom_xytheta.shape == (3,)
    np.testing.assert_allclose(ls.joint_q, data.qpos[7:7 + model.nu])
    # at the home keyframe (upright, ~level) the IMU acc is ~gravity-reaction up in body frame
    np.testing.assert_allclose(ls.imu_acc, [0, 0, 9.81], atol=0.2)

def test_read_lowstate_imu_matches_base_orientation():
    model, data = _sim()
    ls = SimTransport(model, data).read_lowstate()
    bq = data.qpos[3:7]
    exp = R.from_quat([bq[1], bq[2], bq[3], bq[0]]).as_euler("ZYX")  # yaw,pitch,roll
    np.testing.assert_allclose(ls.imu_rpy, exp[::-1], atol=1e-9)     # rpy = roll,pitch,yaw

def test_write_lowcmd_applies_kcustom_law():
    model, data = _sim()
    nu = model.nu
    tr = SimTransport(model, data)
    cmd = LowCmd(q_des=data.qpos[7:7+nu].copy(), qd_des=np.zeros(nu),
                 kp=np.zeros(nu), kd=np.zeros(nu), tau_ff=np.full(nu, 1.5))
    tr.write_lowcmd(cmd)
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    np.testing.assert_allclose(data.ctrl, np.clip(np.full(nu, 1.5), lo, hi))
```

- [ ] **Step 2: Run, expect FAIL** (module missing).

- [ ] **Step 3: Create `transport.py`**

```python
# src/t1_wbc/transport.py
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


@dataclass
class LowCmd:
    q_des: np.ndarray; qd_des: np.ndarray; kp: np.ndarray; kd: np.ndarray; tau_ff: np.ndarray  # (nu,)


class Transport(ABC):
    @abstractmethod
    def read_lowstate(self) -> LowState: ...
    @abstractmethod
    def write_lowcmd(self, cmd: LowCmd) -> None: ...


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
            ls.odom_xytheta = ls.odom_xytheta + self.noise * np.array([1, 1, 1]) * np.random.randn(3)
        return ls

    def write_lowcmd(self, cmd: LowCmd) -> None:
        d = self.data; nu = self.nu
        q = d.qpos[7:7 + nu]; dq = d.qvel[6:6 + nu]
        tau = cmd.kp * (cmd.q_des - q) + cmd.kd * (cmd.qd_des - dq) + cmd.tau_ff
        d.ctrl[:] = np.clip(tau, self.lo, self.hi)
```

- [ ] **Step 4: Run, expect PASS** (3 tests).

- [ ] **Step 5: Commit**
```bash
git add src/t1_wbc/transport.py tests/t1_wbc/test_transport.py
git -c user.name='t1-wbc' -c user.email='t1-wbc@local' commit -m "feat(transport): LowState/LowCmd + SimTransport (SDK-shaped sensors, kCustom apply)"
```

---

## Task 5: State assembly + round-trip equivalence

**Files:** Modify `src/t1_wbc/controller.py`; Test `tests/t1_wbc/test_state_assembly.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/t1_wbc/test_state_assembly.py
import numpy as np, mujoco
from t1_wbc.model import load_t1_model, build_index_maps
from t1_wbc.config import WBCConfig
from t1_wbc.controller import WBController
from t1_wbc.dynamics import CpuDynamics
from t1_wbc.estimator import StateEstimator
from t1_wbc.transport import SimTransport

def test_assembled_state_reproduces_dynamics():
    cfg = WBCConfig(); model, data = load_t1_model(cfg.xml)
    mujoco.mj_resetDataKeyframe(model, data, 0); mujoco.mj_forward(model, data)
    maps = build_index_maps(model)
    direct = CpuDynamics(model, maps).extract(data)             # ground-truth dynamics

    ctrl = WBController(model, cfg)
    est = StateEstimator(model, maps); ctrl.attach_estimator(est)
    ls = SimTransport(model, data).read_lowstate()
    # drive the estimator, then assemble + extract, copying TRUE base twist to isolate the
    # pose path (lin-vel estimation quality is covered by the in-the-loop test, not here):
    d_est = ctrl._assemble_est_dynamics(ls, base_twist=data.qvel[0:6].copy())
    for k in ("M", "com", "Jfoot_L", "Jfoot_R", "Jcom"):
        np.testing.assert_allclose(d_est[k], direct[k], atol=1e-3, err_msg=k)
    np.testing.assert_allclose(d_est["h"], direct["h"], atol=1e-2)
```

- [ ] **Step 2: Run, expect FAIL** (`attach_estimator`/`_assemble_est_dynamics` missing).

- [ ] **Step 3: Add estimator wiring + state assembly to `WBController`**

In `controller.py`, add to `WBController.__init__` (after `self.dyn = ...`):
```python
        self.est = None
        self._est_data = mujoco.MjData(model)
```
and add these methods to `WBController`:
```python
    def attach_estimator(self, est):
        self.est = est

    def _assemble_est_dynamics(self, lowstate, base_twist=None):
        """Estimator + measured joints -> a forwarded MjData -> extract (unchanged)."""
        ls = lowstate
        self.est.update_imu(ls.imu_rpy, ls.imu_gyro, ls.imu_acc, getattr(self, "_t_est", 0.0))
        self.est.update_odometer(ls.odom_xytheta[0], ls.odom_xytheta[1], ls.odom_xytheta[2],
                                 getattr(self, "_t_est", 0.0))
        self.est.update_base_pose_and_contacts(ls.joint_q)
        d = self._est_data
        q = self.est.quat_xyzw()
        d.qpos[0:3] = self.est.position()
        d.qpos[3:7] = [q[3], q[0], q[1], q[2]]                  # xyzw -> wxyz
        d.qpos[7:7 + self.nu] = ls.joint_q
        if base_twist is not None:
            d.qvel[0:6] = base_twist
        else:
            d.qvel[0:3] = self.est.lin_vel(); d.qvel[3:6] = self.est.ang_vel()
        d.qvel[6:6 + self.nu] = ls.joint_dq
        mujoco.mj_forward(self.model, d)
        return self.dyn.extract(d)
```

- [ ] **Step 4: Run, expect PASS.** If `M`/`com`/`Jcom` match but `h` is just over `1e-2`, that's Coriolis from a tiny base-twist mismatch — confirm it's small and keep `1e-2`; a large `M`/`com`/`Jfoot` mismatch means the base pose or wxyz packing is wrong (real bug).

- [ ] **Step 5: Commit**
```bash
git add src/t1_wbc/controller.py tests/t1_wbc/test_state_assembly.py
git -c user.name='t1-wbc' -c user.email='t1-wbc@local' commit -m "feat(controller): estimator wiring + state assembly (estimated MjData -> extract)"
```

---

## Task 6: Transport-driven loop + `--mode track-est` (one tick)

**Files:** Modify `src/t1_wbc/controller.py`, `src/t1_wbc/run.py`; Test `tests/t1_wbc/test_track_estimated.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/t1_wbc/test_track_estimated.py
import numpy as np, mujoco
from t1_wbc.model import load_t1_model, build_index_maps
from t1_wbc.config import WBCConfig
from t1_wbc.controller import WBController
from t1_wbc.reference import ReferenceTrajectory
from t1_wbc.estimator import StateEstimator
from t1_wbc.transport import SimTransport

def test_one_estimated_track_tick():
    cfg = WBCConfig(); model, data = load_t1_model(cfg.xml)
    ctrl = WBController(model, cfg); ctrl.reset(data); ctrl.settle(data)
    maps = build_index_maps(model)
    ctrl.attach_reference(ReferenceTrajectory(model, maps, ctrl.q_home, cfg, 0.0, 0.0, 0.0))
    ctrl.attach_estimator(StateEstimator(model, maps))
    tr = SimTransport(model, data)
    cmd, diag = ctrl.step_track_estimated(tr.read_lowstate(), 0.0)
    assert isinstance(cmd.tau_ff, np.ndarray) and cmd.tau_ff.shape == (model.nu,)
    assert diag["ok"] is True
```

- [ ] **Step 2: Run, expect FAIL** (`step_track_estimated` missing).

- [ ] **Step 3: Add `step_track_estimated` to `WBController`**

```python
    def step_track_estimated(self, lowstate, t):
        assert self.ref is not None and self.est is not None
        self._t_est = t
        d = self._assemble_est_dynamics(lowstate)
        q_act = np.asarray(lowstate.joint_q, dtype=np.float64)
        qd_act = np.asarray(lowstate.joint_dq, dtype=np.float64)
        rs = self.ref.sample(t)
        tg = tracking_targets_from_refsample(rs, q_act, qd_act, self.q_home)   # already imported at top of controller.py
        cmd, z, ok, tau_ff = self._solve_to_cmd(d, tg, q_act, qd_act)
        lh_err = float(np.linalg.norm(d["hand_L_world"] - tg.lh_pos))
        rh_err = float(np.linalg.norm(d["hand_R_world"] - tg.rh_pos))
        return cmd, dict(ok=bool(ok), base_z=float(self.est.position()[2]),
                         max_tau=float(np.abs(tau_ff).max()), lh_err=lh_err, rh_err=rh_err)
```

- [ ] **Step 4: Add `run_track_estimated` + the CLI mode to `run.py`**

Add the function (mirrors `run_track` but reads/writes through `SimTransport` and drives the estimator):
```python
def run_track_estimated(cfg, seconds=None, log=None):
    """Settle, then track the motion with the WBC running on ESTIMATED base state
    (IMU+odom+encoders via SimTransport + StateEstimator). Returns a summary dict."""
    from .transport import SimTransport, LowCmd
    from .estimator import StateEstimator
    model, data = load_t1_model(cfg.xml)
    ctrl = WBController(model, cfg); ctrl.reset(data); ncon = ctrl.settle(data)
    maps = build_index_maps(model)
    ctrl.attach_reference(ReferenceTrajectory(model, maps, ctrl.q_home, cfg, 0.0, 0.0, 0.0))
    ctrl.attach_estimator(StateEstimator(model, maps))
    tr = SimTransport(model, data)
    horizon = ctrl.ref.duration if seconds is None else seconds
    dt = model.opt.timestep; t = 0.0; infeas = 0; zmin = 1e9; lh = []; rh = []
    last = None
    for i in range(int(horizon / dt)):
        mujoco.mj_step1(model, data)
        if i % cfg.control_decimation == 0:
            cmd, diag = ctrl.step_track_estimated(tr.read_lowstate(), t)
            last = cmd
            infeas += int(not diag["ok"]); zmin = min(zmin, float(data.qpos[2]))
            lh.append(diag["lh_err"]); rh.append(diag["rh_err"])
        tr.write_lowcmd(LowCmd(q_des=last.q_des, qd_des=last.qd_des, kp=last.kp, kd=last.kd, tau_ff=last.tau_ff))
        mujoco.mj_step2(model, data)
        t += dt
    return dict(ncon=ncon, infeasible=infeas, min_base_z=zmin, upright=zmin > cfg.upright_z,
                lh_rms=float(np.mean(lh)), rh_rms=float(np.mean(rh)))
```
In `main()`, add `"track-est"` to the `--mode` choices and an `elif args.mode == "track-est": print(run_track_estimated(cfg, args.seconds, log=args.log))`.

- [ ] **Step 5: Run, expect PASS:**
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/t1_wbc/test_track_estimated.py::test_one_estimated_track_tick -q`

- [ ] **Step 6: Commit**
```bash
git add src/t1_wbc/controller.py src/t1_wbc/run.py tests/t1_wbc/test_track_estimated.py
git -c user.name='t1-wbc' -c user.email='t1-wbc@local' commit -m "feat: transport-driven estimated control loop + --mode track-est"
```

---

## Task 7: Estimator-in-the-loop regression (the gate)

**Files:** Modify `tests/t1_wbc/test_track_estimated.py`

- [ ] **Step 1: Write the regression test** (append)

```python
# append to tests/t1_wbc/test_track_estimated.py
from t1_wbc.run import run_track_estimated, run_track

def test_estimated_track_stays_upright():
    cfg = WBCConfig()
    out = run_track_estimated(cfg, seconds=5.0)
    assert out["upright"] is True
    assert out["infeasible"] == 0
    assert out["min_base_z"] > 0.55
    # hand tracking on estimated state within 1.5x of the ground-truth baseline (~1.2/0.86 cm @5s)
    base = run_track(cfg, seconds=5.0)
    assert out["lh_rms"] < 1.5 * base["lh_rms"] + 0.005
    assert out["rh_rms"] < 1.5 * base["rh_rms"] + 0.005
```

- [ ] **Step 2: Run it** (steps two 5 s sims; allow a couple minutes):
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/t1_wbc/test_track_estimated.py::test_estimated_track_stays_upright -q -s`
Expected: PASS — the robot tracks the motion upright on estimated state, hand RMS close to the ground-truth baseline. If it does NOT pass, do NOT relax the asserts — report the two `out`/`base` dicts and STOP (a real divergence means the estimator/assembly is degrading the loop and must be debugged, not papered over).

- [ ] **Step 3: Manual full-motion confirmation:**
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -c "from t1_wbc.run import run_track_estimated; from t1_wbc.config import WBCConfig; print(run_track_estimated(WBCConfig()))"`
Record the dict. Confirm `upright: True`, `infeasible: 0`, hand RMS comparable to the ground-truth full-motion baseline (≈1.7/1.3 cm).

- [ ] **Step 4: Whole suite green:**
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/t1_wbc -q`

- [ ] **Step 5: Commit**
```bash
git add tests/t1_wbc/test_track_estimated.py
git -c user.name='t1-wbc' -c user.email='t1-wbc@local' commit -m "test: estimator-in-the-loop track regression (upright on estimated state)"
```

---

## Definition of Done (Stage 2a)

- `pytest tests/t1_wbc` green (existing 13 + estimator + transport + state-assembly + estimated-track).
- `t1-wbc --mode track-est` keeps the robot upright over the full motion on **estimated** state, hand RMS within tolerance of the ground-truth baseline.
- `estimator.py` is a pure module (no SDK, no torch); the loop is transport-agnostic — swapping `SimTransport`→`SdkTransport` is the only change for Stage 2b.

**Next:** Stage 2b — real `SdkTransport` (`B1LowStateSubscriber`/`B1LowCmdPublisher`/`B1LocoClient`, `kPrepare→kCustom`, 29-DOF raw-index joint map), nonzero servo gains, and the safety + torque-safety layer (umbrella spec §8/§8.1), executed on-robot.
