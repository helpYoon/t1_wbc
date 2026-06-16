# t1_wbc — Hardware Deployment (de-batched, numpy + proxsuite, Booster SDK)

**Date:** 2026-06-16
**Status:** Design — approved in brainstorming, pending spec review
**Supersedes scope of:** the batched-GPU path (`2026-06-14-t1-wbc-batched-gpu-design.md`), which is removed by this work.

## 1. Summary

Turn `t1_wbc` from a torch-based, GPU-batchable research controller into a **self-contained, single-robot, real-time whole-body controller deployable on the physical Booster T1 via the Booster Robotics SDK**.

Three coupled changes:

1. **Delete all batching** (mujoco_warp / GPU / mjlab) and the torch dependency.
2. **De-torch the B=1 hot path to pure numpy + proxsuite** for deterministic, warmup-free, sub-millisecond real-time solves.
3. **Add a hardware path**: a Python floating-base state estimator, a real `BoosterSdkBackend`, and the `kPrepare→kCustom` handshake + safety layer — structured so sim and hardware share one identical control core and differ only in transport.

The result is a standalone repository that runs in MuJoCo with **no torch / no themis_mpc / no t1_controller dependency**, and that drives the real robot when the (optional) Booster SDK Python package is installed.

## 2. Goals / Non-goals

**Goals**
- A self-contained, `pip install`-able repository (own `pyproject.toml`, vendored robot model + sample motion, no cross-repo imports).
- Single control core (numpy + proxsuite) shared verbatim between sim and hardware.
- A floating-base state estimator (IMU + planar odometry + FK) replacing the sim's ground-truth base pose.
- A real `BoosterSdkBackend` with handshake, joint remap, nonzero servo gains, and a safety layer (weight ramp, torque/position clamps, watchdog, abort).
- Deterministic real-time: target a full **500 Hz** loop; decimation to 250 Hz is the documented fallback if the solve budget is exceeded.
- Sim-verified to the current quality bar (full motion upright, ≈1.5 cm hand RMS, 0 infeasible) using **estimated** (not ground-truth) base state.

**Non-goals (this spec)**
- On-robot execution/tuning (no robot or SDK runtime in this environment). We deliver code-complete + sim-verified, plus a written bring-up procedure.
- Re-adding any batched/GPU/RL-throughput path. If that is wanted later it is a separate, re-added path and is explicitly out of scope here.
- Locomotion / stepping / a centroidal MPC layer. The controller remains the fixed-double-stance per-tick WBC.
- Network/DDS configuration of the SDK transport beyond what an example client needs.

## 3. The self-contained-repository requirement

`t1_wbc` currently lives as a sub-package of the `themis_training` repo (`mpc-rl/pyproject.toml`), pulling `torch`, `mujoco-warp`, `jax`, and `themis_mpc.admm_qp`, and reading the robot model + motion via **absolute paths into `t1_controller/`** (`config.T1_XML`, `config.PKL`). After this work it must stand alone.

**Dependency severance**
- Remove `torch`, `mujoco-warp`, `jax`, and `themis_mpc` usage entirely.
- Replace the `themis_mpc.admm_qp` solver with proxsuite (already a dependency).
- Replace the C++/Pinocchio estimator dependency with an in-repo Python port (MuJoCo FK).

**Vendored assets** (so sim runs out of the box)
- The T1 MuJoCo model: `urdf/t1.xml` + the referenced `meshes/` (61 mesh refs) from `t1_controller/robot_models/booster_t1/t1_description`, copied under `assets/robot/`. Mesh path references in the XML are rewritten to the vendored location.
- A sample motion: `t1_controller/data/motion_plan.pkl` copied to `assets/motion_plan.pkl`, used by `run_track` and tests. Overridable via config/CLI.

**Packaging**
- Own `pyproject.toml`, package name `t1_wbc`. Runtime deps: `numpy`, `mujoco`, `scipy` (for `scipy.spatial.transform.Rotation` in the reference/estimator), `proxsuite`. Optional extra `[hardware]`: the Booster SDK Python package (declared `booster_robotic_sdk` locally / published `booster_robotics_sdk_python` — **resolve the exact installable name at impl time**).
- Console script `t1-wbc = t1_wbc.run:main`.
- Assets shipped as package data, resolved via `importlib.resources` (no absolute paths).

**Physical extraction.** The package is restructured to `src/t1_wbc/` with assets + `pyproject.toml` + `tests/` at the repo root, such that the directory can be `git init`'d / moved out of `mpc-rl` and installed standalone. Whether the extraction happens in-place or to a new path is an implementation-plan detail; the design requirement is that **no symbol resolves outside the repo**.

## 4. Architecture — one core, two transports

```
            ┌─────────────────────── control core (numpy, identical sim & hw) ──────────────────────┐
LowState ──▶│ estimator ─▶ dynamics.extract ─▶ assemble_wbc_qp ─▶ proxsuite.solve ─▶ recover_tau ─▶ │──▶ LowCmd
(IMU/q/qd/  │   (base pose/vel/contacts)        (M,h,J from MuJoCo)   (exact QP)       (τ_ff)        │  (q_des,qd_des,
 odometer)  └──────────────────────────────────────────────────────────────────────────────────────┘   kp,kd,τ_ff)
                 ▲                                                                                  │
                 │  Transport.read_lowstate()                              Transport.write_lowcmd() │
        ┌────────┴─────────┐                                              ┌────────────────────────┴┐
        │  SimTransport     │  LowState synthesized from MuJoCo data       │  SdkTransport            │
        │  (MuJoCo)         │  (+ optional IMU/odometer noise);            │  (Booster SDK pub/sub)   │
        │                   │  LowCmd applied to data.ctrl                 │  500 Hz LowState/LowCmd  │
        └───────────────────┘                                              └──────────────────────────┘
```

**`Transport` interface** (the only sim/hardware difference):
- `read_lowstate() -> LowState` — IMU `(rpy, gyro, acc)`, per-joint `(q, qd)` in SDK order, odometer `(x, y, theta)`.
- `write_lowcmd(LowCmd)` — per-joint `(q_des, qd_des, kp, kd, tau_ff)` in SDK order.
- `SimTransport` derives an SDK-shaped `LowState` from MuJoCo (with optional injected IMU noise / odometer drift to stress the estimator) and applies `LowCmd` to `data.ctrl` via the existing `kCustom` law. `SdkTransport` wraps the SDK subscriber/publisher.

Because the estimator, WBC, and command synthesis sit **above** the transport, sim verification exercises the real hardware code path; only the bytes' source/sink change.

## 5. Deletions (batching + torch)

| File | Remove | Keep |
|---|---|---|
| `dynamics_warp.py` | entire file | — |
| `controller.py` | `BatchedWarpController`, `MjlabBatchedWarpController`, `_warp()`/`wp_to_torch`/`wp_from_torch_f32`/`mjw_*` | `WBController` (de-torched) |
| `model.py` | `load_t1_warp_model`, `build_warp_handles`, `T1_NCONMAX`, `T1_NJMAX` | `load_t1_model`, `build_index_maps` |
| `run.py` | `run_batched_balance`, `run_mjlab_balance`, `benchmark_throughput`; `--num-envs/--device/--admm-max-iter/--batch-sizes`; `batched-balance/mjlab-balance/benchmark` modes | `run_balance`, `run_track` |
| `config.py` | `backend`, `device`, `dtype`, `num_envs`, `admm_*` | task gains/weights, friction, regs, timing, settle, the new fields below |
| `solver.py` | `AdmmBackend`, `external_residual`/`_per_env_residual`, `themis_mpc.admm_qp` import | `ProxsuiteBackend` (now numpy-native, the only backend) |

Net effect: `torch`, `mujoco-warp`, `jax`, and `themis_mpc` are no longer imported anywhere in `t1_wbc`.

## 6. De-torch the hot path to numpy + proxsuite

The QP math is **unchanged** (Section "fundamental math" of the controller). Only the array library and the batch dimension change.

- `wbc_qp.assemble_wbc_qp` / `recover_tau`: numpy, no leading `B` dim. `H: (nz,nz)`, `g: (nz,)`, `A_eq: (neq,nz)`, `G: (nineq,nz)`. The `ori_error_world` quaternion-log becomes numpy. Tasks (`_add_task`, CoM, hands, base-ori, posture-diagonal-scatter) stay structurally identical.
- `dynamics.CpuDynamics.extract`: return a numpy dict (it is already numpy internally; drop the `torch.as_tensor(...).unsqueeze(0)` wrapping).
- `targets.py`, `reference.ReferenceTrajectory.sample`: return numpy.
- `solver.ProxsuiteBackend`: consumes the numpy QP directly (dropping the prior torch→numpy conversion); **persists the proxsuite QP object and warm-starts across ticks** (new — the current backend re-inits each call); one-sided `G z ≤ b` via `lb=-inf`. Becomes the sole `make_solver` result.
- `controller.WBController`: numpy throughout. **Precompute loop-invariant constants once at init** — friction-cone `Gc/bc` rows, the base-orientation selector, identity/regularizer blocks, and the config weight vectors (these were rebuilt every tick; now the only path, so the win is unconditional).

**Regression gate:** `run_track` (ground-truth state, no estimator) must reproduce the current result — upright, ≈1.5 cm hand RMS, 0 infeasible over the full motion — now via exact proxsuite instead of torch ADMM. Expect equal-or-better tracking (exact solve vs iterative).

**Real-time budget (500 Hz ⇒ 2 ms):** estimator + `extract_dynamics` (MuJoCo `mj_fullM` + 4× `mj_jac` + `mj_jacSubtreeCom`) + numpy assembly + warm-started proxsuite + `recover_tau`. Proxsuite on this size (nz≈47, ~18 eq, ~80 ineq) is expected sub-millisecond; MuJoCo extract ≈0.1–0.3 ms. The plan includes a per-phase timing harness to confirm the budget; **if 500 Hz is not met, decimate the QP to 250 Hz and hold the last `LowCmd`** (firmware `kCustom` PD stabilizes between updates) — a config knob, not a redesign.

## 7. State estimator (`estimator.py`)

Faithful Python port of `t1_controller/robot_runtime/booster_t1_hw_interface/StateEstimator`, using `t1_wbc`'s MuJoCo model for FK instead of Pinocchio. The SDK publishes IMU `(rpy, gyro, acc)` and planar odometry `(x, y, theta)` but **no quaternion, no base z, no base linear velocity, no contact flags** — the estimator fills exactly the base fields `extract_dynamics` reads from `data` today.

- **orientation** ← IMU rpy → quaternion, with **initial yaw subtracted** (robot sees yaw=0 at startup, matching the reference frame).
- **angular velocity** ← IMU gyro (already body-frame).
- **position x,y** ← odometer, centered and rotated into the yaw-zeroed frame.
- **position z** ← MuJoCo FK from current joints + estimated quaternion, pinning the **lower** of the two foot frames to world z=0.
- **contact flags** ← after FK, a foot is in contact if its world-z is within a threshold (`≈10 mm`) of the ground.
- **linear velocity** ← complementary filter (`τ≈0.05 s`) blending gravity-removed IMU integration with the finite-difference of odometer xy.

Updates: `update_imu(rpy, gyro, acc, t)` at LowState rate; `update_odometer(x, y, theta, t)` on odometer callbacks; `update_base_pose_and_contacts(joint_positions)` (FK) each tick after IMU. Outputs the quaternion, body angular velocity, world position, body linear velocity, and `[left, right]` contact flags consumed by the WBC.

FK uses a scratch `MjData` exactly as `reference._fk_com` already does. The estimator is a pure module (no SDK types), so it is unit-testable against synthetic IMU/odometer/joint inputs.

## 8. Booster SDK backend + operational/safety layer

`action_backend.BoosterSdkBackend` (replaces the `NotImplementedError` stub) plus a ported `JointMapping` and gains.

- **Transport**: subscribe to `LowState` (IMU + per-joint q/qd) and the odometer topic; publish `LowCmd`.
- **Handshake**: `kPrepare → kCustom`, SERIAL `cmd_type`, per the existing `BoosterT1HwInterface.cpp` sequence.
- **Joint remap**: MuJoCo actuator order ↔ SDK joint index, ported from `JointMapping.h` (a single source of truth, round-trip tested).
- **Command law**: `τ = kp(q_des−q) + kd(qd_des−qd) + τ_ff`, computed by the firmware in `kCustom`, with **nonzero per-joint servo `kp/kd`** (not the sim's 0/0) so the servo absorbs model error.
  - **Initial gains (before on-robot tuning) are taken verbatim from the T1 `model_settings.joint_pd_gains`** — the same single source the existing C++ hardware loader (`T1HoldGainsLoader`) projects into `motorCmd.kp/kd`:

    | Joint group | kp | kd |
    |---|---|---|
    | Head (2) + all 14 arm joints | 20.0 | 0.5 |
    | Waist + all Hip (pitch/roll/yaw) + Knee | 200.0 | 5.0 |
    | Ankle pitch + Ankle roll (both feet) | 50.0 | 3.0 |

    (29 joints total; stiff hips/knees/waist, compliant ankles for contact/CoP, soft arms for manipulation.) These per-joint pairs are **vendored into `t1_wbc`'s config** (keyed by joint name, projected to MuJoCo actuator order — no runtime read of `t1_controller`) as the `servo_kp/servo_kd` defaults, and the same gains drive the weight-ramp hold-pose (matching the C++ deployment). They are the *starting point* for on-robot tuning, not final values.
- **Safety layer (required for a real humanoid):**
  - **Weight ramp**: blend from a PD hold-pose into the WBC output over a configurable duration at startup (no torque step into the QP solution).
  - **Clamps**: per-joint torque and position limits applied to every `LowCmd`.
  - **Watchdog**: abort to a safe hold if `LowState` goes stale beyond a timeout.
  - **Abort/E-stop**: clean transition out of `kCustom`.
- **Rate**: stream at 500 Hz (or the decimation fallback from §6).

### 8.1 Torque safety — bounding the commanded torque (defense in depth)

The single most important hardware-safety property: the commanded motor torque is bounded at **five independent layers**, so no single failure makes it spike.

1. **In-solver hard torque limits (primary).** `assemble_wbc_qp` adds, per actuated joint, the inequality `τ_min ≤ (M·v̈ + h + τ_fric − Jcᵀ·W) ≤ τ_max` (`_torque_rows`). proxsuite enforces it exactly — the QP **cannot** return a `τ_ff` outside the limits and instead sacrifices the *soft* CoM/hand/posture tasks; `reg_vdot` keeps accelerations bounded so the objective never drives torque toward infinity. Contact forces are separately hard-bounded (friction cone + `fz_max` + CoP), bounding the `Jcᵀ·W` term.

2. **`τ_max` = verified real limits, with conservative bring-up caps.** The per-joint `τ_max` is the robot's actuator torque limit. We **verify the vendored XML `actuator_ctrlrange` against the real T1 datasheet** and add a config `torque_limit_scale ∈ (0,1]` to run the first on-robot sessions at a reduced fraction (e.g. 0.3–0.5), then relax. The QP, `recover_tau`, and the clamp all use the same scaled limit.

3. **The firmware PD adds on top of `τ_ff` — bounded explicitly.** On hardware `τ_motor = kp(q_des−q) + kd(q̇_des−q̇) + τ_ff`; the QP bounds only `τ_ff`. We bound the total by (a) the moderate servo gains of §8 (not high-gain position control); (b) a **`τ_ff` headroom budget** — the QP `τ_max` is set to `scale·τ_limit − τ_pd_margin` so `τ_ff` + the expected PD term stays inside the joint limit; (c) the backend final clamp on the *full* command (layer 4); and (d) the firmware's own joint-torque saturation. `q_des/q̇_des` are the consistent one-tick setpoint, so in normal operation the PD term is small.

4. **Backend final clamp + slew limit (last line).** Every `LowCmd` is per-joint torque-clamped to a hard safe range and **rate-limited** (max |Δτ| per tick), so a discontinuity — reference jump, contact-flag flip, estimator glitch — cannot produce a spike. Position setpoints are clamped to joint limits too.

5. **Infeasible / unsafe solve → safe hold, never garbage.** If proxsuite reports infeasible (or the external residual exceeds tolerance), the backend does **not** apply the raw solution — it holds the last safe command / ramps to the PD hold-pose and flags it. The startup **weight ramp** (no torque step into the QP solution) and the **watchdog** (stale-`LowState` → safe hold) cover the remaining discontinuity/loss-of-signal cases.

New config knobs: `torque_limit_scale`, `tau_pd_margin`, `tau_slew_max` (per-tick torque rate cap), plus per-joint `torque_limit`/`pos_limit` (defaults from the vendored XML, overridable). First on-robot runs use reduced `torque_limit_scale` + a slow weight ramp + a human on the E-stop.

## 9. Verification (sim only)

1. **Regression** — `run_track` (ground-truth state, no estimator): reproduce current numbers. Guards the de-torch rewrite.
2. **Estimator-in-the-loop** — new `run_track` variant using `SimTransport`: `LowState` synthesized from MuJoCo, estimator reconstructs base state, WBC runs on the *estimate*, `LowCmd` applied to MuJoCo. Pass = stays upright over the full motion on estimated state, with hand RMS within a tolerance of the ground-truth baseline. Optional injected IMU/odometer noise to characterize robustness.
3. **Unit tests** — estimator (FK z vs known pose, contact-flag thresholding, yaw-zeroing, complementary-filter convergence), joint-mapping round-trip, command-law parity (sim `kCustom` law == `MuJoCoBackend`), and the precompute-constants equivalence (numpy QP == prior torch QP to tolerance on a fixed input).

All tests run on a CPU-only host with no SDK and no torch.

## 10. Repository structure (target)

```
t1_wbc/                         # repo root (extractable, pip-installable)
├── pyproject.toml              # name=t1_wbc; deps numpy,mujoco,scipy,proxsuite; [hardware]=booster sdk
├── README.md
├── docs/superpowers/specs/2026-06-16-t1-wbc-hardware-deployment-design.md
├── assets/
│   ├── robot/t1.xml + meshes/  # vendored MuJoCo model
│   └── motion_plan.pkl         # vendored sample motion
├── src/t1_wbc/
│   ├── config.py               # package-relative asset paths; servo gains; safety/timing knobs
│   ├── model.py  dynamics.py  wbc_qp.py  solver.py  targets.py  reference.py
│   ├── controller.py           # WBController only, numpy, precomputed constants
│   ├── estimator.py            # NEW
│   ├── transport.py            # NEW: Transport, SimTransport, SdkTransport
│   ├── action_backend.py       # MuJoCoBackend + real BoosterSdkBackend + safety
│   ├── run.py  logging_utils.py
└── tests/t1_wbc/               # estimator, joint-map, command-law, regression, qp-equivalence
```

## 11. Success criteria (definition of done)

- `pip install .` in a fresh venv with **only** numpy/mujoco/scipy/proxsuite pulls a working sim; `import t1_wbc` references nothing outside the repo.
- `t1-wbc --mode track` reproduces the upright / ≈1.5 cm hand RMS / 0-infeasible baseline.
- The estimator-in-the-loop sim run stays upright over the full motion on estimated state.
- All unit tests pass on a CPU-only, torch-free, SDK-free host.
- `BoosterSdkBackend` + estimator + safety layer are code-complete, with a per-phase timing report showing the per-tick budget (and the decimation fallback wired).
- A written **on-robot bring-up procedure** (gains, ramp duration, watchdog timeout, joint-map verification, first-contact checklist).

## 12. Risks & open questions

- **Real-time budget unproven without measurement.** Mitigation: timing harness + 250 Hz decimation fallback; both in scope.
- **SDK Python package name/version** (`booster_robotic_sdk` vs `booster_robotics_sdk_python`) and the exact `LowState`/`LowCmd`/odometer API surface — confirm against `booster_robotics_sdk/python/binding.cpp` and the `low_level_*.py` examples during implementation.
- **Estimator fidelity gap.** The MuJoCo-FK port must match the C++ Pinocchio behavior closely enough; the FK contact frames (sole centres) and joint order must align. Unit tests pin this.
- **Gains/limits are robot-specific.** `T1HoldGains`, torque/position clamps, ramp duration, watchdog timeout are ported as defaults but tuned on-robot (bring-up step).
- **Mesh vendoring size** (~9.5 MB incl. ROS extras). Vendor only `t1.xml` + referenced meshes, not rviz/launch/ROS include.
- **No sim-to-real guarantee.** Sim verification with estimated state reduces but does not eliminate the gap; first on-robot runs proceed under the weight-ramp + clamps + watchdog with a human on the E-stop.

## 13. Out of scope / follow-ups

- On-robot tuning and execution (bring-up step you run).
- Locomotion / centroidal MPC / object-weight adaptation (separate future specs).
- Any re-added batched GPU / RL path.
