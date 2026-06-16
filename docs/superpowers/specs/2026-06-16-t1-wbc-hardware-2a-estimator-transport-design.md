# t1_wbc Hardware — Stage 2a: State Estimator + Transport + Estimator-in-the-Loop Sim

**Date:** 2026-06-16
**Status:** Design — approved in brainstorming, pending spec review
**Umbrella spec:** `2026-06-16-t1-wbc-hardware-deployment-design.md` (§7 estimator, §9 verification). This doc concretizes the **sim-testable half** (estimator + transport + verification). The on-robot SDK glue + safety is **Stage 2b** (separate spec).
**Builds on:** the merged numpy/proxsuite foundation (Plan 1).

## 1. Summary

Make the WBC run on an **estimated** floating-base state — reconstructed from IMU + planar odometry + joint encoders — instead of MuJoCo ground-truth, and verify in simulation that the robot still tracks the motion upright. Everything here is buildable and fully testable in this environment; no robot is required. Three pieces:

1. **`estimator.py`** — a Python state estimator (faithful port of the C++ `StateEstimator`, using MuJoCo FK) that produces base pose/velocity/contacts from IMU + odometry + encoders.
2. **State assembly** — reconstruct a MuJoCo `MjData` from (estimated base + measured joints), `mj_forward`, and reuse the existing `CpuDynamics.extract` unchanged, so the WBC's dynamics model stays MuJoCo and only the *source* of base state changes.
3. **`transport.py`** — a `LowState`/`LowCmd` abstraction with a `SimTransport` that synthesizes SDK-shaped sensor data from a MuJoCo sim and applies commands back, so the same control loop runs in sim now and on the SDK later.

## 2. Goals / Non-goals

**Goals**
- A pure, unit-tested `estimator.py` matching the documented C++ behavior, using MuJoCo FK.
- A transport-agnostic control loop where the only sim/hardware difference is the `Transport`.
- An **estimator-in-the-loop sim run**: the full track motion with the WBC on estimated state, staying upright with hand RMS within tolerance of the ground-truth baseline.

**Non-goals (→ Stage 2b)**
- Real `SdkTransport` (`B1LowStateSubscriber`/`B1LowCmdPublisher`/`B1LocoClient`), `kPrepare→kCustom` handshake, the 29-DOF joint mapping, the safety/torque-safety layer, on-robot bring-up.
- Any change to the verified WBC math or the QP.

## 3. SDK API reference (for the transport data shapes; full use is 2b)

Confirmed from `booster_robotics_sdk/python` + examples (informs the `LowState`/`LowCmd` dataclasses so 2b is a thin swap):
- **State (subscribe):** `B1LowStateSubscriber(cb)` → `imu_state.{rpy[3], gyro[3], acc[3]}`, `motor_state_serial[i].{q, dq, ddq, tau_est}`. `B1OdometerStateSubscriber(cb)` → `{x, y, theta}`.
- **Command (publish):** `B1LowCmdPublisher.Write(LowCmd)`, `LowCmd.cmd_type = LowCmdType.SERIAL`, `motor_cmd[i].{q, dq, tau, kp, kd}`, ~500 Hz.
- **Mode:** `B1LocoClient.ChangeMode(RobotMode.kPrepare→kCustom)`; `kDamping` = safe abort.
- **Joint-count risk (2b, recorded here):** SDK defines `JointIndex` (23-DOF / 4-DOF arm, `kJointCnt=23`) **and** `JointIndexWith7DofArm` (29-DOF, `kJointCnt7DofArm=29`). Our T1 is the **29-DOF** robot (`T1JointGains.h`, indices 0–28, ankles serial). The vendored Python binding exposed only the 23-DOF enum/`B1JointCnt=23`, so 2b will use **raw integer SDK indices 0–28** from a ported name→index table, and "does the binding command 29 joints" is a must-verify-on-robot item. Stage 2a does not touch this — `SimTransport` uses the MuJoCo actuator order directly.

## 4. State estimator (`estimator.py`)

Faithful port of `t1_controller/.../booster_t1_hw_interface/StateEstimator`, FK done with a scratch `MjData` (as `reference._fk_com` already does). The SDK gives IMU `(rpy, gyro, acc)` and planar odometry `(x, y, θ)` but **no quaternion, base z, base linear velocity, or contact flags** — the estimator fills exactly those.

**Interface**
```
class StateEstimator:
    def __init__(self, model, index_maps, contact_threshold_m=0.01, comp_tau_s=0.05): ...
    def update_imu(self, rpy, gyro, acc, t):                    # body-frame; acc INCLUDES gravity reaction
    def update_odometer(self, x, y, theta, t):
    def update_base_pose_and_contacts(self, joint_q):           # FK; call after update_imu for this tick
    # outputs (all world/body per the WBC's convention):
    def quat_xyzw(self): ...        # base orientation, initial-yaw subtracted
    def ang_vel(self): ...          # body angular velocity (= gyro)
    def position(self): ...         # world x,y (odometer, yaw-zeroed) + z (FK)
    def lin_vel(self): ...          # body linear velocity (complementary filter)
    def contact_flags(self): ...    # [left, right] bool
```

**Algorithm (ported verbatim in behavior)**
- **Orientation:** IMU `rpy` → quaternion; at first sample capture `yaw0`; thereafter subtract `yaw0` so startup yaw = 0 (matches the reference anchored at yaw0=0).
- **Angular velocity:** `= gyro` (already body frame).
- **Position x,y:** odometer `(x,y)` centered at the first sample and rotated into the yaw-zeroed frame.
- **Position z:** scratch `MjData` ← estimated base quat + `joint_q`; `mj_forward`; read both foot sole-frame world-z; **pin the lower foot to z=0** ⇒ base z. (Foot frames = the same sole points `build_index_maps` already provides.)
- **Contact flags:** after FK, foot in contact iff its world-z is within `contact_threshold_m` (≈10 mm) of the ground.
- **Linear velocity:** complementary blend (`comp_tau_s≈0.05`) of gravity-removed IMU integration and the finite-difference of odometer x,y.

## 5. State assembly → WBC

Per control tick, reconstruct a persistent `MjData` (`self._est_data`) and run the existing extraction:
1. `qpos[0:3]` ← estimator world position; `qpos[3:7]` ← estimator quaternion (xyzw→wxyz); `qpos[7:7+nu]` ← measured `joint_q`.
2. `qvel[0:6]` ← estimator base twist (MuJoCo free-joint convention); `qvel[6:6+nu]` ← measured `joint_dq`.
3. `mujoco.mj_forward(model, est_data)`.
4. `CpuDynamics.extract(est_data)` — **unchanged** → `M, h, J, com, …` for the WBC.

This keeps MuJoCo as the dynamics model; the only change vs the sim path is that base state comes from the estimator, not ground-truth. The free-joint twist convention is pinned by a **round-trip test** (§7), not asserted in prose.

## 6. Transport abstraction (`transport.py`)

```
@dataclass
class LowState:   imu_rpy; imu_gyro; imu_acc; joint_q; joint_dq; odom_xytheta   # numpy
@dataclass
class LowCmd:     q_des; qd_des; kp; kd; tau_ff                                  # numpy (nu,)
class Transport(ABC):
    def read_lowstate(self) -> LowState: ...
    def write_lowcmd(self, cmd: LowCmd) -> None: ...
```
- **`SimTransport(model, data, noise=None)`** — `read_lowstate` synthesizes from the MuJoCo sim: `imu_rpy` from base quat→ZYX, `imu_gyro` = body angular velocity, `imu_acc` = body-frame specific force *including gravity reaction* (`R_bodyᵀ(a_world − g)`), `joint_q/dq` from `data`, `odom_xytheta` from base x/y/yaw; optional injected IMU/odometer noise. `write_lowcmd` applies the `kCustom` law to `data.ctrl` (reuses `MuJoCoBackend`). Joints are in MuJoCo actuator order (no SDK remap in sim).
- **`SdkTransport`** — interface only here; implemented in 2b.

## 7. Control loop + run mode

A transport-driven loop, identical in sim and (2b) on hardware:
```
ls = transport.read_lowstate()
est.update_imu(ls.imu_rpy, ls.imu_gyro, ls.imu_acc, t)
est.update_odometer(*ls.odom_xytheta, t)
est.update_base_pose_and_contacts(ls.joint_q)
# assemble est MjData (§5) -> extract -> tracking targets -> assemble QP -> solve -> recover tau
cmd = LowCmd(q_des, qd_des, kp=servo_kp, kd=servo_kd, tau_ff=tau)
transport.write_lowcmd(cmd)
```
New run mode `--mode track-est` drives this loop with `SimTransport` over the same sim setup as `--mode track`. (`--mode track` keeps the ground-truth path as the regression baseline.)

## 8. Verification (all in sim, no robot)

- **Estimator unit tests:** FK base-z vs a known pose (place the robot at a known height, confirm recovered z); contact-flag thresholding (lift a foot, flag clears); yaw-zeroing (first sample → yaw 0); complementary-filter convergence (constant odometer velocity → lin_vel converges to it).
- **State-assembly round-trip:** from a known sim `data`, derive a `LowState`, run the estimator + assembly, `extract`, and confirm `M, com, Jfoot` match a direct `extract(data)` within tolerance (pins the free-joint twist convention and the whole estimate→WBC path).
- **Estimator-in-the-loop sim (the real gate):** full track motion via `--mode track-est` → `upright=True`, `infeasible=0`, hand RMS within a tolerance (e.g. ≤ 1.5×) of the ground-truth baseline (≈1.7/1.3 cm). Plus an optional injected-noise run to characterize robustness.

## 9. File structure (additions to the repo)

```
src/t1_wbc/
  estimator.py        # NEW — StateEstimator (MuJoCo FK)
  transport.py        # NEW — LowState/LowCmd, Transport, SimTransport (+ SdkTransport stub)
  controller.py       # ADD — a transport-driven loop / step_track_estimated path reusing _solve_to_cmd
  run.py              # ADD — --mode track-est
tests/t1_wbc/
  test_estimator.py       # NEW
  test_state_assembly.py  # NEW
  test_track_estimated.py # NEW (estimator-in-the-loop regression)
```

## 10. Success criteria

- `pytest tests/t1_wbc` green (existing 13 + estimator/assembly/estimated-track).
- `t1-wbc --mode track-est` keeps the robot upright over the full motion on **estimated** state, hand RMS within tolerance of the ground-truth baseline.
- `estimator.py` is a pure module (no SDK / no torch); the loop is transport-agnostic (swapping `SimTransport`→`SdkTransport` is the only change for 2b).

## 11. Risks & open questions

- **Estimator fidelity gap.** MuJoCo-FK port must match the C++ Pinocchio behavior closely enough that the closed loop stays upright; the estimator-in-the-loop gate is the backstop. Foot sole frames + joint order must align with `build_index_maps`.
- **Free-joint velocity convention.** MuJoCo free-joint `qvel` (translational vs rotational frame) must be written correctly in state assembly — pinned by the round-trip test, not assumed.
- **Sim IMU realism.** `SimTransport`'s synthesized `acc`/`gyro`/odometry are idealized (optional noise only); true sensor bias/latency is a Stage-2b/on-robot concern.

## 12. Out of scope → Stage 2b

Real `SdkTransport` (SDK pub/sub + `B1LocoClient`), the `kPrepare→kCustom` handshake, the 29-DOF raw-index joint mapping (`T1JointGains.h` table), nonzero servo gains from `task.info`, and the safety + torque-safety layer (the five-layer torque bounding, weight ramp, watchdog, clamps) + on-robot bring-up.
