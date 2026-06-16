# On-robot bring-up: `t1_wbc` whole-body QP on the Booster T1

This is the procedure for running the WBC on a **real T1**. The control loop is
`run_hw` / `run_hw_loop` in `t1_wbc.run`: read `LowState` → estimate + WBC →
`SafetyLayer` → write `LowCmd`. Everything below assumes the robot is **on a
gantry / harness** and a **human is on the physical E-stop**.

> Do not run free-standing until you have completed every check on this page.

---

## 1. Install (on the robot's onboard computer)

```bash
pip install -e ".[hardware]"
python -c "import booster_robotics_sdk_python"   # must import cleanly
```

If the import fails, the Booster SDK / its Python binding is not installed for
this interpreter — fix that before going further.

## 2. 29-DOF binding check (CRITICAL)

`t1_wbc` maps **29 joints** (7-DOF arms). The vendored SDK binding *must* expose
the same count, or the MuJoCo↔SDK permutation is wrong and commands land on the
wrong actuators.

```bash
python -c "import booster_robotics_sdk_python as B; print(B.B1JointCnt)"
```

- Prints `29` → good, proceed.
- Prints `23` → this is the **4-DOF-arm build**. `SdkTransport.__init__` will
  (correctly) refuse to start with an `AssertionError` — do **not** force it.
  Either:
  - rebuild the binding against the SDK's `JointIndexWith7DofArm` /
    `kJointCnt7DofArm = 29`, **or**
  - confirm with Booster that a length-29 `motor_cmd` is accepted on your unit.

  **Never run with a count mismatch.**

## 3. First-run safety config (conservative)

For the very first power-up, build the config defensively:

```python
from t1_wbc.config import WBCConfig
cfg = WBCConfig(
    torque_limit_scale=0.3,    # only 30% of the torque envelope
    tau_pd_margin=2.0,         # Nm reserved for the firmware PD on top of tau_ff
    ramp_seconds=5.0,          # slow weight-ramp blend (hold-pose -> WBC)
    watchdog_timeout_s=0.05,   # 50 ms LowState staleness -> safe hold
)
```

Plus, physically: **robot on a gantry/harness**, **human on the E-stop**.

## 4. Bring-up sequence

1. **Power on** the robot; bring up the loco client.
2. Start the controller: `t1-wbc --mode hw` (or call `run_hw(cfg)` from Python).
3. The loco client moves to **`kPrepare`** (the hold pose) — the robot servos to
   the prepare posture and holds.
4. The client enters **`kCustom`** — custom `LowCmd` is now accepted.
5. The **weight ramp** blends the WBC in over `ramp_seconds` (start hold-pose,
   end full WBC).
6. **Watch** the CoP / centre-of-pressure margin and joint torques throughout.
   On **any** anomaly, abort to **`kDamping`** — `SdkTransport.stop()` does this
   automatically on loop exit (the `finally` in `run_hw`), and the E-stop is the
   hard backstop.

## 5. Joint-map smoke test (BEFORE any balancing)

With the robot **supported**, command a **tiny single-joint offset** and confirm
the **correct physical joint** moves. This validates the MuJoCo↔SDK permutation
on the real robot — the one thing that cannot be checked in sim. Do this for at
least one joint per limb before trusting full-body commands.

## 6. Ramp-up

Once stable at `torque_limit_scale=0.3`:

- raise `torque_limit_scale` toward `1.0` in steps,
- shorten `ramp_seconds`,
- tune the servo gains from the **20 / 200 / 50** starting point
  (head+arm `kp=20`, waist+hip+knee `kp=200`, ankle `kp=50`; see
  `config.servo_gains_for`).

Re-verify CoP margin and torques at each step; back off if anything degrades.
