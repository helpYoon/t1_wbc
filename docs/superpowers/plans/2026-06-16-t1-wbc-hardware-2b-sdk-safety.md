# t1_wbc Hardware Stage 2b — SDK Backend + Safety + Torque-Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the real Booster-SDK transport, the 29-DOF joint mapping, nonzero servo gains, and the safety + torque-safety layer, so `t1_wbc` is deployable on the physical T1 — with everything except the live-robot run verified here in sim/CPU.

**Architecture:** A `SafetyLayer` wraps the WBC command (servo gains + weight-ramp + clamps + slew + watchdog + infeasible→hold) and is transport-agnostic; a `JointMap` permutes between MuJoCo actuator order and SDK index 0–28; an `SdkTransport` (lazy SDK import, mock-testable) swaps for `SimTransport`. The testable core (map, safety, torque-safety, the safety-wrapped loop) is exercised in sim; the live SDK + on-robot bring-up are documented.

**Tech Stack:** Python, numpy, mujoco==3.6.0, scipy, proxsuite, pytest. The Booster SDK Python package is an **optional** `[hardware]` dep — `SdkTransport` imports it lazily so the package + tests run without it.

**Design basis:** umbrella spec `2026-06-16-t1-wbc-hardware-deployment-design.md` §8 (SDK backend + safety) and §8.1 (torque safety); SDK API + 29-DOF joint table concretized in `2026-06-16-t1-wbc-hardware-2a-estimator-transport-design.md` §3. Builds on the Stage-2a estimator/transport/`step_track_estimated` (merged).

**Branch:** feature branch off `main` (e.g. `hw-2b-sdk-safety`). venv `.venv/bin/python` (torch-free). **Run pytest with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.** Commit: `git -c user.name='t1-wbc' -c user.email='t1-wbc@local' commit ...`.

---

## File Structure

```
src/t1_wbc/
  joint_map.py        # NEW — T1 name<->SDK-index table; MuJoCo<->SDK permutation
  safety.py           # NEW — clamp/slew primitives + SafetyLayer (ramp/watchdog/hold/gains)
  sdk_transport.py    # NEW — SdkTransport (lazy SDK, callbacks, remap, handshake); mock-testable
  config.py           # MODIFY — torque_limit_scale, tau_pd_margin, tau_slew_max, ramp/watchdog, servo-gain table
  controller.py       # MODIFY — apply torque_limit_scale + tau_pd_margin to the QP ctrlrange
  run.py              # MODIFY — run_track_estimated_safe (--mode track-est-safe), run_hw (--mode hw)
docs/
  BRINGUP-hardware.md # NEW — on-robot bring-up procedure
tests/t1_wbc/
  test_joint_map.py       # NEW
  test_safety.py          # NEW
  test_sdk_transport.py   # NEW (against a mock SDK)
  test_hw_safe_sim.py     # NEW (safety-wrapped loop upright in sim)
```

**Existing pieces (from Plan 1/2a):** `WBController` with `step_track_estimated(lowstate, t) -> (JointCommand, diag)`, `_solve_to_cmd`, `ctrlrange`, `reset/settle/attach_reference/attach_estimator`; `transport.LowState/LowCmd/SimTransport`; `estimator.StateEstimator`; `model.build_index_maps(model)["name_to_act_index"]` (joint name → MuJoCo actuator index); `config.WBCConfig` with `servo_kp/servo_kd` (currently scalars 0.0), `friction_ff`, `reg_psd`, `upright_z`.

---

## Task 1: Joint map (MuJoCo ↔ SDK index 0–28)

**Files:** Create `src/t1_wbc/joint_map.py`, `tests/t1_wbc/test_joint_map.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/t1_wbc/test_joint_map.py
import numpy as np
from t1_wbc.model import load_t1_model, build_index_maps
from t1_wbc.joint_map import T1_SDK_JOINTS, JointMap

def test_table_is_29_and_indices_0_to_28():
    assert len(T1_SDK_JOINTS) == 29
    idx = sorted(i for _, i in T1_SDK_JOINTS)
    assert idx == list(range(29))

def test_mujoco_to_sdk_roundtrip():
    model, _ = load_t1_model()
    jm = JointMap(build_index_maps(model))
    assert jm.sdk_count == 29 and jm.nu == model.nu
    mj = np.arange(model.nu, dtype=float)          # a distinct value per MuJoCo joint
    sdk = jm.mujoco_to_sdk(mj)                      # (29,)
    back = jm.sdk_to_mujoco(sdk)                    # (nu,)
    np.testing.assert_allclose(back, mj)            # lossless round-trip on shared joints

def test_names_align_with_mujoco():
    model, _ = load_t1_model()
    maps = build_index_maps(model)
    jm = JointMap(maps)
    # every SDK table name that exists in the MuJoCo model maps to that model's index
    for name, sdk_idx in T1_SDK_JOINTS:
        if name in maps["name_to_act_index"]:
            assert jm.mj_index_of_sdk[sdk_idx] == maps["name_to_act_index"][name]
```

- [ ] **Step 2: Run, expect FAIL** (module missing):
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/t1_wbc/test_joint_map.py -q`

- [ ] **Step 3: Create `joint_map.py`** (table vendored verbatim from `t1_description/T1JointGains.h`)

```python
# src/t1_wbc/joint_map.py
"""T1 joint-name <-> Booster-SDK index table (7-DOF-arm / 29-DOF variant, indices 0-28),
and the permutation between MuJoCo actuator order and SDK order. Vendored from
t1_controller .../t1_description/T1JointGains.h (kT1Joints). The SDK's vendored Python
binding only names the 23-DOF JointIndex enum, so we use raw integer indices here."""
import numpy as np

# (URDF joint name, SDK index). SERIAL cmd_type interprets indices 21,22,27,28 as ankle
# pitch/roll (serial joints), which is what we want.
T1_SDK_JOINTS = [
    ("AAHead_yaw", 0), ("Head_pitch", 1),
    ("Left_Shoulder_Pitch", 2), ("Left_Shoulder_Roll", 3), ("Left_Elbow_Pitch", 4),
    ("Left_Elbow_Yaw", 5), ("Left_Wrist_Pitch", 6), ("Left_Wrist_Yaw", 7), ("Left_Hand_Roll", 8),
    ("Right_Shoulder_Pitch", 9), ("Right_Shoulder_Roll", 10), ("Right_Elbow_Pitch", 11),
    ("Right_Elbow_Yaw", 12), ("Right_Wrist_Pitch", 13), ("Right_Wrist_Yaw", 14), ("Right_Hand_Roll", 15),
    ("Waist", 16),
    ("Left_Hip_Pitch", 17), ("Left_Hip_Roll", 18), ("Left_Hip_Yaw", 19), ("Left_Knee_Pitch", 20),
    ("Left_Ankle_Pitch", 21), ("Left_Ankle_Roll", 22),
    ("Right_Hip_Pitch", 23), ("Right_Hip_Roll", 24), ("Right_Hip_Yaw", 25), ("Right_Knee_Pitch", 26),
    ("Right_Ankle_Pitch", 27), ("Right_Ankle_Roll", 28),
]
SDK_JOINT_CNT = 29


class JointMap:
    """Permute (nu,) MuJoCo-actuator-order vectors <-> (29,) SDK-order vectors."""
    def __init__(self, index_maps):
        name_to_mj = index_maps["name_to_act_index"]
        self.nu = len(name_to_mj)
        self.sdk_count = SDK_JOINT_CNT
        # mj_index_of_sdk[sdk_idx] = MuJoCo actuator index (or -1 if the model lacks that joint)
        self.mj_index_of_sdk = np.full(SDK_JOINT_CNT, -1, dtype=int)
        for name, sdk_idx in T1_SDK_JOINTS:
            if name in name_to_mj:
                self.mj_index_of_sdk[sdk_idx] = name_to_mj[name]
        self._shared = self.mj_index_of_sdk >= 0     # SDK slots present in the model

    def mujoco_to_sdk(self, v_mj, fill=0.0):
        out = np.full(SDK_JOINT_CNT, fill, dtype=np.float64)
        out[self._shared] = np.asarray(v_mj, dtype=np.float64)[self.mj_index_of_sdk[self._shared]]
        return out

    def sdk_to_mujoco(self, v_sdk):
        out = np.zeros(self.nu, dtype=np.float64)
        v_sdk = np.asarray(v_sdk, dtype=np.float64)
        out[self.mj_index_of_sdk[self._shared]] = v_sdk[self._shared]
        return out
```

- [ ] **Step 4: Run, expect PASS** (3 tests). For the T1 7-DOF-arm model all 29 names match, so `_shared` is all True and the round-trip is exact.

- [ ] **Step 5: Commit**
```bash
git add src/t1_wbc/joint_map.py tests/t1_wbc/test_joint_map.py
git -c user.name='t1-wbc' -c user.email='t1-wbc@local' commit -m "feat(joint_map): T1 MuJoCo<->SDK index permutation (29-DOF, raw indices)"
```

---

## Task 2: Config — servo gains + torque-limit scaling + PD margin; apply to the QP

**Files:** Modify `src/t1_wbc/config.py`, `src/t1_wbc/controller.py`; Test `tests/t1_wbc/test_safety.py`

- [ ] **Step 1: Write the failing test** — `tests/t1_wbc/test_safety.py`

```python
import numpy as np
from t1_wbc.model import load_t1_model, build_index_maps
from t1_wbc.config import WBCConfig
from t1_wbc.controller import WBController
from t1_wbc.config import servo_gains_for

def test_servo_gains_table_per_group():
    model, _ = load_t1_model()
    maps = build_index_maps(model)
    kp, kd = servo_gains_for(maps)            # (nu,), (nu,)
    n2i = maps["name_to_act_index"]
    assert kp[n2i["Left_Shoulder_Pitch"]] == 20.0 and kd[n2i["Left_Shoulder_Pitch"]] == 0.5
    assert kp[n2i["Left_Hip_Pitch"]] == 200.0 and kd[n2i["Left_Hip_Pitch"]] == 5.0
    assert kp[n2i["Left_Ankle_Pitch"]] == 50.0 and kd[n2i["Left_Ankle_Pitch"]] == 3.0

def test_torque_limit_scale_tightens_qp_ctrlrange():
    model, data = load_t1_model()
    full = WBController(model, WBCConfig(torque_limit_scale=1.0, tau_pd_margin=0.0))
    half = WBController(model, WBCConfig(torque_limit_scale=0.5, tau_pd_margin=0.0))
    np.testing.assert_allclose(half.ctrlrange[:, 1], 0.5 * full.ctrlrange[:, 1])
    np.testing.assert_allclose(half.ctrlrange[:, 0], 0.5 * full.ctrlrange[:, 0])

def test_pd_margin_shrinks_limits_symmetrically():
    model, _ = load_t1_model()
    base = WBController(model, WBCConfig(torque_limit_scale=1.0, tau_pd_margin=0.0)).ctrlrange.copy()
    m = WBController(model, WBCConfig(torque_limit_scale=1.0, tau_pd_margin=2.0)).ctrlrange
    np.testing.assert_allclose(m[:, 1], base[:, 1] - 2.0)
    np.testing.assert_allclose(m[:, 0], base[:, 0] + 2.0)
```

- [ ] **Step 2: Run, expect FAIL** (`servo_gains_for`/new config fields missing).

- [ ] **Step 3: Add config fields + the gain table to `config.py`**

Add fields to `WBCConfig` (next to the existing `servo_kp`/`servo_kd`):
```python
    torque_limit_scale: float = 1.0   # (0,1]; reduce for conservative on-robot bring-up
    tau_pd_margin: float = 0.0        # Nm headroom reserved for the firmware PD on top of tau_ff
    tau_slew_max: float = 80.0        # max |delta tau| per control tick (Nm)
    ramp_seconds: float = 2.0         # weight-ramp blend hold-pose -> WBC
    watchdog_timeout_s: float = 0.05  # LowState staleness -> safe hold
```
Add the per-joint servo-gain table (the task.info `joint_pd_gains` values) as a module function:
```python
import numpy as np
_SERVO_GROUPS = {  # (kp, kd) by joint-name predicate, in priority order
    "head_arm": (20.0, 0.5), "waist_hip_knee": (200.0, 5.0), "ankle": (50.0, 3.0),
}
def servo_gains_for(index_maps):
    """Per-MuJoCo-joint (kp, kd) from the T1 task.info joint_pd_gains schedule."""
    n2i = index_maps["name_to_act_index"]
    nu = len(n2i)
    kp = np.zeros(nu); kd = np.zeros(nu)
    for name, i in n2i.items():
        if "Ankle" in name:
            g = _SERVO_GROUPS["ankle"]
        elif ("Hip" in name) or ("Knee" in name) or (name == "Waist"):
            g = _SERVO_GROUPS["waist_hip_knee"]
        else:                                  # head + all arm joints
            g = _SERVO_GROUPS["head_arm"]
        kp[i], kd[i] = g
    return kp, kd
```

- [ ] **Step 4: Apply `torque_limit_scale` + `tau_pd_margin` to the QP ctrlrange in `WBController.__init__`**

Replace the `self.ctrlrange = np.asarray(model.actuator_ctrlrange, dtype=np.float64)` line with:
```python
        cr = np.asarray(model.actuator_ctrlrange, dtype=np.float64) * cfg.torque_limit_scale
        cr[:, 1] = cr[:, 1] - cfg.tau_pd_margin
        cr[:, 0] = cr[:, 0] + cfg.tau_pd_margin
        self.ctrlrange = cr
```
(So the QP's hard torque inequality already respects the scaled, PD-headroom-reserved limit — layers 1+2+3 of spec §8.1.)

- [ ] **Step 5: Run, expect PASS** (3 tests). Then whole suite:
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/t1_wbc -q`
(The existing track regression uses `torque_limit_scale=1.0, tau_pd_margin=0.0` defaults → ctrlrange unchanged → still passes.)

- [ ] **Step 6: Commit**
```bash
git add src/t1_wbc/config.py src/t1_wbc/controller.py tests/t1_wbc/test_safety.py
git -c user.name='t1-wbc' -c user.email='t1-wbc@local' commit -m "feat(config): servo gains + torque_limit_scale + pd_margin applied to QP ctrlrange"
```

---

## Task 3: Safety primitives — torque clamp + slew limiter

**Files:** Create `src/t1_wbc/safety.py`; Modify `tests/t1_wbc/test_safety.py`

- [ ] **Step 1: Append failing tests to `tests/t1_wbc/test_safety.py`**

```python
from t1_wbc.safety import clamp_torque, slew_limit

def test_clamp_torque_bounds_per_joint():
    tau = np.array([100.0, -100.0, 5.0])
    lo = np.array([-30.0, -30.0, -30.0]); hi = np.array([30.0, 30.0, 30.0])
    np.testing.assert_allclose(clamp_torque(tau, lo, hi), [30.0, -30.0, 5.0])

def test_slew_limit_caps_delta():
    prev = np.array([0.0, 0.0])
    tau = np.array([200.0, -5.0])
    out = slew_limit(tau, prev, max_delta=10.0)
    np.testing.assert_allclose(out, [10.0, -5.0])   # first capped, second within
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Create `safety.py` with the primitives**

```python
# src/t1_wbc/safety.py
"""Hardware safety layer: per-joint clamps, torque slew limiting, weight-ramp,
watchdog, and infeasible->hold gating. Pure numpy; transport-agnostic."""
import numpy as np


def clamp_torque(tau, lo, hi):
    return np.clip(np.asarray(tau, dtype=np.float64), lo, hi)


def slew_limit(tau, prev_tau, max_delta):
    tau = np.asarray(tau, dtype=np.float64); prev_tau = np.asarray(prev_tau, dtype=np.float64)
    return prev_tau + np.clip(tau - prev_tau, -max_delta, max_delta)
```

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit**
```bash
git add src/t1_wbc/safety.py tests/t1_wbc/test_safety.py
git -c user.name='t1-wbc' -c user.email='t1-wbc@local' commit -m "feat(safety): torque clamp + slew-limit primitives"
```

---

## Task 4: SafetyLayer — weight-ramp + watchdog + infeasible→hold + servo gains

**Files:** Modify `src/t1_wbc/safety.py`, `tests/t1_wbc/test_safety.py`

- [ ] **Step 1: Append failing tests**

```python
from t1_wbc.safety import SafetyLayer
from t1_wbc.transport import LowCmd

def _mk(model, cfg, maps):
    return SafetyLayer(model, cfg, maps)

def test_hold_when_infeasible_zeros_tau_ff():
    from t1_wbc.config import WBCConfig
    from t1_wbc.model import load_t1_model, build_index_maps
    model, _ = load_t1_model(); maps = build_index_maps(model); cfg = WBCConfig()
    sl = _mk(model, cfg, maps); hold_q = np.zeros(model.nu); sl.begin(hold_q)
    raw = LowCmd(q_des=np.ones(model.nu), qd_des=np.zeros(model.nu),
                 kp=np.zeros(model.nu), kd=np.zeros(model.nu), tau_ff=np.full(model.nu, 50.0))
    out = sl.wrap(raw, ok=False, t=10.0, lowstate_age=0.0)   # past ramp, but infeasible
    np.testing.assert_allclose(out.tau_ff, 0.0)               # hold: no feedforward torque
    np.testing.assert_allclose(out.q_des, hold_q)             # PD to the hold pose
    assert np.all(out.kp > 0)                                 # servo gains engaged

def test_watchdog_stale_state_holds():
    from t1_wbc.config import WBCConfig
    from t1_wbc.model import load_t1_model, build_index_maps
    model, _ = load_t1_model(); maps = build_index_maps(model); cfg = WBCConfig()
    sl = _mk(model, cfg, maps); sl.begin(np.zeros(model.nu))
    raw = LowCmd(q_des=np.ones(model.nu), qd_des=np.zeros(model.nu),
                 kp=np.zeros(model.nu), kd=np.zeros(model.nu), tau_ff=np.full(model.nu, 50.0))
    out = sl.wrap(raw, ok=True, t=10.0, lowstate_age=0.2)     # stale beyond watchdog_timeout_s
    np.testing.assert_allclose(out.tau_ff, 0.0)

def test_weight_ramp_blends_in_tau_ff():
    from t1_wbc.config import WBCConfig
    from t1_wbc.model import load_t1_model, build_index_maps
    model, _ = load_t1_model(); maps = build_index_maps(model)
    cfg = WBCConfig(ramp_seconds=2.0)
    sl = _mk(model, cfg, maps); sl.begin(np.zeros(model.nu))
    raw = LowCmd(q_des=np.zeros(model.nu), qd_des=np.zeros(model.nu),
                 kp=np.zeros(model.nu), kd=np.zeros(model.nu), tau_ff=np.full(model.nu, 100.0))
    half = sl.wrap(raw, ok=True, t=1.0, lowstate_age=0.0)     # 50% through the ramp
    assert np.all(half.tau_ff < raw.tau_ff) and np.all(half.tau_ff > 0)
    full = sl.wrap(raw, ok=True, t=5.0, lowstate_age=0.0)     # past ramp
    np.testing.assert_allclose(full.tau_ff, raw.tau_ff)
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Add `SafetyLayer` to `safety.py`**

```python
from .config import servo_gains_for
from .transport import LowCmd


class SafetyLayer:
    """Wraps a raw WBC LowCmd into a safe one: servo gains + weight-ramp + clamps + slew,
    falling back to a PD hold (tau_ff=0) on infeasible solve or stale state."""
    def __init__(self, model, cfg, index_maps):
        self.cfg = cfg
        self.nu = model.nu
        self.servo_kp, self.servo_kd = servo_gains_for(index_maps)
        cr = np.asarray(model.actuator_ctrlrange, dtype=np.float64) * cfg.torque_limit_scale
        self.tau_lo = cr[:, 0]; self.tau_hi = cr[:, 1]
        self._prev_tau = np.zeros(self.nu)
        self._hold_q = np.zeros(self.nu)
        self._t0 = None

    def begin(self, hold_q):
        self._hold_q = np.asarray(hold_q, dtype=np.float64).copy()
        self._t0 = None
        self._prev_tau = np.zeros(self.nu)

    def wrap(self, raw, ok, t, lowstate_age):
        if self._t0 is None:
            self._t0 = t
        safe = (ok) and (lowstate_age <= self.cfg.watchdog_timeout_s)
        if not safe:                                   # hold: PD to hold pose, no feedforward
            q_des, qd_des, tau_ff = self._hold_q.copy(), np.zeros(self.nu), np.zeros(self.nu)
        else:
            alpha = min(1.0, (t - self._t0) / max(self.cfg.ramp_seconds, 1e-9))  # weight ramp
            q_des = (1 - alpha) * self._hold_q + alpha * raw.q_des
            qd_des = alpha * raw.qd_des
            tau_ff = alpha * raw.tau_ff
        tau_ff = clamp_torque(tau_ff, self.tau_lo, self.tau_hi)
        tau_ff = slew_limit(tau_ff, self._prev_tau, self.cfg.tau_slew_max)
        self._prev_tau = tau_ff.copy()
        return LowCmd(q_des=q_des, qd_des=qd_des,
                      kp=self.servo_kp.copy(), kd=self.servo_kd.copy(), tau_ff=tau_ff)
```

- [ ] **Step 4: Run, expect PASS** (the 3 new + earlier safety tests).

- [ ] **Step 5: Commit**
```bash
git add src/t1_wbc/safety.py tests/t1_wbc/test_safety.py
git -c user.name='t1-wbc' -c user.email='t1-wbc@local' commit -m "feat(safety): SafetyLayer (servo gains + weight-ramp + watchdog + infeasible->hold)"
```

---

## Task 5: Safety-wrapped loop verified in sim (`--mode track-est-safe`)

**Files:** Modify `src/t1_wbc/run.py`; Create `tests/t1_wbc/test_hw_safe_sim.py`

This exercises the FULL 2b command path (estimator + WBC + SafetyLayer with nonzero servo gains) in sim — the closest we get to the on-robot loop without the robot.

- [ ] **Step 1: Write the failing test**

```python
# tests/t1_wbc/test_hw_safe_sim.py
from t1_wbc.run import run_track_estimated_safe
from t1_wbc.config import WBCConfig

def test_safety_wrapped_loop_stays_upright():
    cfg = WBCConfig()                       # nonzero servo gains applied by SafetyLayer
    out = run_track_estimated_safe(cfg, seconds=5.0)
    assert out["upright"] is True
    assert out["infeasible"] == 0
    assert out["min_base_z"] > 0.55
    assert out["lh_rms"] < 0.05 and out["rh_rms"] < 0.05   # looser: ramp + servo PD perturb early ticks
```

- [ ] **Step 2: Run, expect FAIL** (`run_track_estimated_safe` missing).

- [ ] **Step 3: Add `run_track_estimated_safe` to `run.py`**

```python
def run_track_estimated_safe(cfg, seconds=None, log=None):
    """Estimated-state track loop wrapped by the SafetyLayer (servo gains, weight-ramp,
    clamps, slew, infeasible->hold) — the on-robot command path, run in sim."""
    from .transport import SimTransport, LowCmd
    from .estimator import StateEstimator
    from .safety import SafetyLayer
    model, data = load_t1_model(cfg.xml)
    ctrl = WBController(model, cfg); ctrl.reset(data); ncon = ctrl.settle(data)
    maps = build_index_maps(model)
    ctrl.attach_reference(ReferenceTrajectory(model, maps, ctrl.q_home, cfg, 0.0, 0.0, 0.0))
    ctrl.attach_estimator(StateEstimator(model, maps))
    tr = SimTransport(model, data)
    safety = SafetyLayer(model, cfg, maps); safety.begin(ctrl.q_home)
    horizon = ctrl.ref.duration if seconds is None else seconds
    dt = model.opt.timestep; t = 0.0; infeas = 0; zmin = 1e9; lh = []; rh = []; last = None
    for i in range(int(horizon / dt)):
        mujoco.mj_step1(model, data)
        if i % cfg.control_decimation == 0:
            cmd, diag = ctrl.step_track_estimated(tr.read_lowstate(), t)
            raw = LowCmd(q_des=cmd.q_des, qd_des=cmd.qd_des, kp=cmd.kp, kd=cmd.kd, tau_ff=cmd.tau_ff)
            last = safety.wrap(raw, ok=diag["ok"], t=t, lowstate_age=0.0)
            infeas += int(not diag["ok"]); zmin = min(zmin, float(data.qpos[2]))
            lh.append(diag["lh_err"]); rh.append(diag["rh_err"])
        tr.write_lowcmd(last)
        mujoco.mj_step2(model, data)
        t += dt
    return dict(ncon=ncon, infeasible=infeas, min_base_z=zmin, upright=zmin > cfg.upright_z,
                lh_rms=float(np.mean(lh)), rh_rms=float(np.mean(rh)))
```
In `main()`: add `"track-est-safe"` to `--mode` choices + `elif args.mode == "track-est-safe": print(run_track_estimated_safe(cfg, args.seconds, log=args.log))`.

- [ ] **Step 4: Run, expect PASS** (allow a couple minutes):
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/t1_wbc/test_hw_safe_sim.py -q -s`
If NOT upright: the nonzero servo gains (200/5 on legs) on top of `tau_ff` may be over-stiff at the ramp seam — report `out` and STOP; the fix is gain/ramp tuning, not loosening "upright". Record the printed dict.

- [ ] **Step 5: Commit**
```bash
git add src/t1_wbc/run.py tests/t1_wbc/test_hw_safe_sim.py
git -c user.name='t1-wbc' -c user.email='t1-wbc@local' commit -m "feat: safety-wrapped estimated loop verified upright in sim (--mode track-est-safe)"
```

---

## Task 6: SdkTransport (lazy SDK, mock-testable)

**Files:** Create `src/t1_wbc/sdk_transport.py`, `tests/t1_wbc/test_sdk_transport.py`

- [ ] **Step 1: Write the failing test** (against an injected mock SDK — no real SDK needed)

```python
# tests/t1_wbc/test_sdk_transport.py
import numpy as np
from t1_wbc.model import load_t1_model, build_index_maps
from t1_wbc.joint_map import JointMap
from t1_wbc.sdk_transport import SdkTransport
from t1_wbc.transport import LowState, LowCmd

class _MotorState:
    def __init__(self, q, dq): self.q = q; self.dq = dq; self.tau_est = 0.0
class _Imu:
    def __init__(self): self.rpy = [0.0, 0.05, 0.2]; self.gyro = [0.0, 0.0, 0.1]; self.acc = [0, 0, 9.81]
class _LowStateMsg:
    def __init__(self, n): self.imu_state = _Imu(); self.motor_state_serial = [_MotorState(0.1*i, 0.0) for i in range(n)]
class _OdomMsg:
    x, y, theta = 1.0, 2.0, 0.2
class _MotorCmd:
    def __init__(self): self.q=self.dq=self.tau=self.kp=self.kd=self.weight=0.0
class _LowCmd:
    def __init__(self): self.cmd_type=None; self.motor_cmd=[]
class _Pub:
    def __init__(self): self.written=[]
    def InitChannel(self): pass
    def Write(self, c): self.written.append(c)
class _Loco:
    def __init__(self): self.modes=[]
    def Init(self): pass
    def ChangeMode(self, m): self.modes.append(m)

def _mocks():
    return dict(low_cmd_pub=_Pub(), loco=_Loco(), LowCmd=_LowCmd, MotorCmd=_MotorCmd,
                cmd_type_serial="SERIAL", kPrepare="kPrepare", kCustom="kCustom", joint_cnt=29)

def test_read_lowstate_remaps_sdk_to_mujoco():
    model, _ = load_t1_model(); maps = build_index_maps(model); jm = JointMap(maps)
    tr = SdkTransport(model, maps, _sdk=_mocks())
    tr._on_state(_LowStateMsg(29)); tr._on_odom(_OdomMsg())
    ls = tr.read_lowstate()
    assert isinstance(ls, LowState) and ls.joint_q.shape == (model.nu,)
    # SDK index i carried 0.1*i; after remap, the MuJoCo slot for SDK idx 5 holds 0.5
    assert abs(ls.joint_q[jm.mj_index_of_sdk[5]] - 0.5) < 1e-9
    np.testing.assert_allclose(ls.imu_rpy, [0.0, 0.05, 0.2])
    np.testing.assert_allclose(ls.odom_xytheta, [1.0, 2.0, 0.2])

def test_write_lowcmd_fills_serial_motorcmd_by_sdk_index():
    model, _ = load_t1_model(); maps = build_index_maps(model); jm = JointMap(maps)
    tr = SdkTransport(model, maps, _sdk=_mocks())
    cmd = LowCmd(q_des=np.arange(model.nu, dtype=float), qd_des=np.zeros(model.nu),
                 kp=np.full(model.nu, 7.0), kd=np.full(model.nu, 0.7), tau_ff=np.full(model.nu, 1.0))
    tr.write_lowcmd(cmd)
    written = tr._sdk["low_cmd_pub"].written[-1]
    assert written.cmd_type == "SERIAL" and len(written.motor_cmd) == 29
    mj5 = jm.mj_index_of_sdk[5]
    assert abs(written.motor_cmd[5].q - cmd.q_des[mj5]) < 1e-9
    assert written.motor_cmd[5].kp == 7.0 and written.motor_cmd[5].tau == 1.0

def test_start_does_prepare_then_custom():
    model, _ = load_t1_model(); maps = build_index_maps(model)
    tr = SdkTransport(model, maps, _sdk=_mocks())
    tr.start()
    assert tr._sdk["loco"].modes == ["kPrepare", "kCustom"]
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Create `sdk_transport.py`** (the `_sdk` dict is injectable for tests; in production it's built from the real SDK via a lazy import)

```python
# src/t1_wbc/sdk_transport.py
"""Booster-SDK transport: LowState/Odometer subscribe -> LowState; LowCmd publish (SERIAL,
kCustom). The real SDK is imported lazily so the package/tests run without it; tests inject
a mock `_sdk` dict. Joints are remapped MuJoCo<->SDK index 0-28 via JointMap."""
import time
import numpy as np
from .transport import Transport, LowState, LowCmd
from .joint_map import JointMap, SDK_JOINT_CNT


def _real_sdk():
    import booster_robotics_sdk_python as B
    B.ChannelFactory.Instance().Init(0)
    loco = B.B1LocoClient(); loco.Init()
    pub = B.B1LowCmdPublisher(); pub.InitChannel()
    return dict(low_cmd_pub=pub, loco=loco, LowCmd=B.LowCmd, MotorCmd=B.MotorCmd,
                cmd_type_serial=B.LowCmdType.SERIAL, kPrepare=B.RobotMode.kPrepare,
                kCustom=B.RobotMode.kCustom, kDamping=B.RobotMode.kDamping,
                joint_cnt=B.B1JointCnt, _module=B)


class SdkTransport(Transport):
    def __init__(self, model, index_maps, _sdk=None):
        self.model = model
        self.jm = JointMap(index_maps)
        self._sdk = _sdk if _sdk is not None else _real_sdk()
        self.n = int(self._sdk["joint_cnt"])
        assert self.n == SDK_JOINT_CNT, (
            f"SDK exposes {self.n} joints but the T1 map needs {SDK_JOINT_CNT} (7-DOF arm). "
            "The vendored binding is likely the 23-DOF build — see docs/BRINGUP-hardware.md.")
        self._latest_state = None
        self._latest_odom = None
        self._state_t = 0.0
        if _sdk is None:                       # real run: subscribe (callbacks)
            B = self._sdk["_module"]
            self._state_sub = B.B1LowStateSubscriber(self._on_state); self._state_sub.InitChannel()
            self._odom_sub = B.B1OdometerStateSubscriber(self._on_odom); self._odom_sub.InitChannel()

    def _on_state(self, msg):
        self._latest_state = msg; self._state_t = time.monotonic()

    def _on_odom(self, msg):
        self._latest_odom = msg

    def start(self):
        """kPrepare -> kCustom handshake."""
        self._sdk["loco"].ChangeMode(self._sdk["kPrepare"]); time.sleep(2.0)
        self._sdk["loco"].ChangeMode(self._sdk["kCustom"]); time.sleep(0.5)

    def stop(self):
        if "kDamping" in self._sdk:
            self._sdk["loco"].ChangeMode(self._sdk["kDamping"])

    def state_age(self):
        return time.monotonic() - self._state_t

    def read_lowstate(self) -> LowState:
        st = self._latest_state; od = self._latest_odom
        q_sdk = np.array([st.motor_state_serial[i].q for i in range(self.n)])
        dq_sdk = np.array([st.motor_state_serial[i].dq for i in range(self.n)])
        imu = st.imu_state
        return LowState(imu_rpy=np.array(imu.rpy, dtype=np.float64),
                        imu_gyro=np.array(imu.gyro, dtype=np.float64),
                        imu_acc=np.array(imu.acc, dtype=np.float64),
                        joint_q=self.jm.sdk_to_mujoco(q_sdk),
                        joint_dq=self.jm.sdk_to_mujoco(dq_sdk),
                        odom_xytheta=np.array([od.x, od.y, od.theta], dtype=np.float64))

    def write_lowcmd(self, cmd: LowCmd) -> None:
        q = self.jm.mujoco_to_sdk(cmd.q_des); dq = self.jm.mujoco_to_sdk(cmd.qd_des)
        kp = self.jm.mujoco_to_sdk(cmd.kp); kd = self.jm.mujoco_to_sdk(cmd.kd)
        tau = self.jm.mujoco_to_sdk(cmd.tau_ff)
        n = self.n
        low = self._sdk["LowCmd"](); low.cmd_type = self._sdk["cmd_type_serial"]
        mc = [self._sdk["MotorCmd"]() for _ in range(n)]
        for i in range(n):
            mc[i].q = float(q[i]); mc[i].dq = float(dq[i]); mc[i].tau = float(tau[i])
            mc[i].kp = float(kp[i]); mc[i].kd = float(kd[i]); mc[i].weight = 0.0
        low.motor_cmd = mc
        self._sdk["low_cmd_pub"].Write(low)
```

- [ ] **Step 4: Run, expect PASS** (3 tests, no real SDK). Whole suite green.

- [ ] **Step 5: Commit**
```bash
git add src/t1_wbc/sdk_transport.py tests/t1_wbc/test_sdk_transport.py
git -c user.name='t1-wbc' -c user.email='t1-wbc@local' commit -m "feat(sdk_transport): SdkTransport (lazy SDK, mock-tested, SDK<->MuJoCo remap, handshake)"
```

---

## Task 7: Hardware entry point `run_hw` + on-robot bring-up doc

**Files:** Modify `src/t1_wbc/run.py`; Create `docs/BRINGUP-hardware.md`; Test (mock-transport smoke) in `tests/t1_wbc/test_sdk_transport.py`

- [ ] **Step 1: Write the failing smoke test** (the hw loop runs against an injected mock SdkTransport — no robot)

```python
# append to tests/t1_wbc/test_sdk_transport.py
def test_run_hw_loop_with_mock_transport_writes_commands():
    from t1_wbc.run import run_hw_loop
    from t1_wbc.config import WBCConfig
    model, data = load_t1_model(); maps = build_index_maps(model)
    tr = SdkTransport(model, maps, _sdk=_mocks())
    # seed a state so read_lowstate works before the first callback
    tr._on_state(_LowStateMsg(29)); tr._on_odom(_OdomMsg())
    n_written = run_hw_loop(WBCConfig(), model, data, maps, tr, ticks=5)
    assert n_written == 5
    assert len(tr._sdk["low_cmd_pub"].written) == 5
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Add `run_hw_loop` (testable core) + `run_hw` (real entry) to `run.py`**

```python
def run_hw_loop(cfg, model, data, maps, transport, ticks=None):
    """The hardware control loop: read LowState -> estimate+WBC -> SafetyLayer -> write LowCmd.
    Transport-agnostic (real SdkTransport on robot; mock/SimTransport in tests). Returns the
    number of commands written. `data` is only used to settle/seed q_home."""
    from .estimator import StateEstimator
    from .safety import SafetyLayer
    from .transport import LowCmd
    ctrl = WBController(model, cfg); ctrl.reset(data); ctrl.settle(data)
    ctrl.attach_reference(ReferenceTrajectory(model, maps, ctrl.q_home, cfg, 0.0, 0.0, 0.0))
    ctrl.attach_estimator(StateEstimator(model, maps))
    safety = SafetyLayer(model, cfg, maps); safety.begin(ctrl.q_home)
    dt = model.opt.timestep; t = 0.0; n = 0
    horizon_ticks = ticks if ticks is not None else int(ctrl.ref.duration / dt)
    age_fn = getattr(transport, "state_age", lambda: 0.0)
    for i in range(horizon_ticks):
        ls = transport.read_lowstate()
        cmd, diag = ctrl.step_track_estimated(ls, t)
        raw = LowCmd(q_des=cmd.q_des, qd_des=cmd.qd_des, kp=cmd.kp, kd=cmd.kd, tau_ff=cmd.tau_ff)
        safe = safety.wrap(raw, ok=diag["ok"], t=t, lowstate_age=age_fn())
        transport.write_lowcmd(safe); n += 1; t += dt
    return n


def run_hw(cfg):
    """REAL on-robot entry. Requires the Booster SDK + a connected T1. Builds an SdkTransport,
    runs the kPrepare->kCustom handshake, then the hw loop; on exit, falls back to kDamping."""
    from .sdk_transport import SdkTransport
    model, data = load_t1_model(cfg.xml)
    maps = build_index_maps(model)
    tr = SdkTransport(model, maps)        # lazy-imports the SDK
    import time as _t; _t.sleep(0.2)      # let the first LowState callback arrive
    tr.start()
    try:
        run_hw_loop(cfg, model, data, maps, tr)
    finally:
        tr.stop()
```
In `main()`: add `"hw"` to `--mode` choices + `elif args.mode == "hw": run_hw(cfg)`.

- [ ] **Step 4: Run the smoke test, expect PASS** (5 commands written via the mock).

- [ ] **Step 5: Write `docs/BRINGUP-hardware.md`** — the on-robot procedure (this is documentation, not a test). Include, concretely:
  - **Install:** `pip install -e .[hardware]` on the robot's onboard computer; confirm `python -c "import booster_robotics_sdk_python"`.
  - **29-DOF binding check (the §3 risk):** verify the installed binding exposes 29 joints — `python -c "import booster_robotics_sdk_python as B; print(B.B1JointCnt)"`. If it prints `23`, the vendored binding is the 4-DOF-arm build; rebuild it against the SDK's `JointIndexWith7DofArm`/`kJointCnt7DofArm`, or confirm with Booster that `motor_cmd` of length 29 is accepted. **Do not run with a count mismatch.**
  - **First-run safety config:** start with `WBCConfig(torque_limit_scale=0.3, tau_pd_margin=2.0, ramp_seconds=5.0, watchdog_timeout_s=0.05)`, a human on the E-stop, robot on a gantry/harness.
  - **Sequence:** power on → `--mode hw` → confirm `kPrepare` reached its hold pose → confirm `kCustom` entered → the 5 s weight ramp blends the WBC in → watch CoP margin / torques; abort to `kDamping` on any anomaly.
  - **Ramp-up:** once stable at 0.3, raise `torque_limit_scale` toward 1.0, lower `ramp_seconds`, and tune servo gains from the 20/200/50 starting point.
  - **Joint-map smoke (before balancing):** command a tiny single-joint move and confirm the *correct physical joint* moves (validates the MuJoCo↔SDK permutation on the real robot).

- [ ] **Step 6: Whole suite green, then commit**
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/t1_wbc -q
git add src/t1_wbc/run.py docs/BRINGUP-hardware.md tests/t1_wbc/test_sdk_transport.py
git -c user.name='t1-wbc' -c user.email='t1-wbc@local' commit -m "feat: run_hw loop + entry point (mock-tested) + on-robot bring-up doc"
```

---

## Definition of Done (Stage 2b)

- `pytest tests/t1_wbc` green (2a suite + joint-map + safety + sdk-transport(mock) + safety-wrapped-sim), all **without** the real SDK installed.
- `--mode track-est-safe` keeps the robot upright in sim with the full hardware command path (nonzero servo gains + weight-ramp + clamps + slew + infeasible→hold).
- `SdkTransport` + `run_hw` are code-complete and exercised against a mock SDK; the live SDK is a lazy import.
- `docs/BRINGUP-hardware.md` documents the on-robot procedure, including the 29-DOF binding verification and the conservative first-run config.

**On-robot (your step, needs the physical T1):** install `[hardware]`, verify the 29-DOF binding, run `--mode hw` under the conservative config with an E-stop, do the joint-map smoke, then ramp up. This plan makes that a config-and-run exercise, not a code exercise.
