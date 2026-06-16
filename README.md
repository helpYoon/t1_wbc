# t1_wbc

A self-contained, **torch-free** single-robot whole-body QP controller for the
Booster T1. It runs a per-tick inverse-dynamics QP in MuJoCo to track a recorded
motion plan, built entirely on **numpy + proxsuite** (no RL, no OCS2, no
Pinocchio, no GPU/batched runtime). The robot MJCF and motion plan ship inside
the package, so it has no sibling-repo dependencies.

The hardware-deployment path (floating-base state estimator, sensor/command
transport, Booster-SDK backend, and the safety + torque-safety layer) is **built
and verified in simulation** — only the live on-robot run remains (see
[Hardware](#hardware)).

## Install

```bash
pip install -e .
```

Dependencies: `numpy`, `mujoco==3.6.0`, `scipy`, `proxsuite`. An optional
`[hardware]` extra pulls the Booster SDK (`pip install -e ".[hardware]"`), needed
only for the on-robot `--mode hw` path.

## Run

Simulation modes (no robot needed):

```bash
python -m t1_wbc.run --mode track            # track the motion on ground-truth state (baseline)
python -m t1_wbc.run --mode balance          # settle, then hold/balance at home
python -m t1_wbc.run --mode track-est        # track on ESTIMATED base state (IMU+odom+encoders)
python -m t1_wbc.run --mode track-est-safe   # track-est wrapped by the full safety layer
                                             #   (servo gains, weight-ramp, clamps, slew, hold)
```

Flags: `--seconds N` (run N seconds), `--viewer` (live MuJoCo window — needs a
display), `--time-scale S` (motion playback speed; default 5 = quasi-static),
`--log out.csv` (per-tick diagnostics), `--no-friction-ff` (disable the Coulomb
friction feedforward).

On-robot (requires `[hardware]` + a connected T1):

```bash
python -m t1_wbc.run --mode hw               # real Booster-SDK run; see docs/BRINGUP-hardware.md
```

(Also installed as the `t1-wbc` console script.)

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/t1_wbc -q
```

The suite runs entirely torch-free and **without the Booster SDK installed** (the
SDK is imported lazily; its transport is unit-tested against a mock). Coverage
includes the QP (verified bit-for-bit against a captured reference fixture),
end-to-end tracking on ground-truth and on estimated state, the estimator,
transport, joint map, and the safety layer.

## How it works

Per control tick: read sensors (`LowState`) → reconstruct the floating-base state
(`estimator.py`, MuJoCo FK) → assemble a `MjData` and extract `M, h, J`
(`dynamics.py`) → build and solve the whole-body QP over joint accelerations +
contact wrenches (`wbc_qp.py`, `solver.py`, proxsuite) → recover the inverse-
dynamics torque → wrap it in the safety layer (`safety.py`) → emit a command
(`LowCmd`). The only difference between sim and hardware is the **transport**
(`transport.SimTransport` ↔ `sdk_transport.SdkTransport`); the estimator, WBC,
and safety layer are identical on both.

## Hardware

The deployment path is code-complete and sim-verified:

- `estimator.py` — floating-base state estimator (IMU + planar odometry + joint
  encoders → base pose/velocity/contacts via MuJoCo FK).
- `transport.py` / `sdk_transport.py` — the `Transport` abstraction; `SimTransport`
  (sim) and `SdkTransport` (real SDK: `kPrepare→kCustom` handshake, 29-DOF
  MuJoCo↔SDK joint remap, SERIAL/`kCustom` command).
- `safety.py` — the `SafetyLayer`: nonzero servo gains, startup weight-ramp,
  per-joint torque/slew limits, watchdog, and infeasible→hold. Torque is bounded
  inside the QP via `config.torque_limit_scale` / `tau_pd_margin`.

`--mode track-est-safe` exercises this entire command path in MuJoCo (the robot
stays upright on estimated state through the safety layer). To run on the
physical robot, follow **`docs/BRINGUP-hardware.md`** — install `[hardware]`,
**verify the SDK exposes 29 joints** (`B1JointCnt`; `SdkTransport` refuses to
start otherwise), then run `--mode hw` under a conservative config
(`torque_limit_scale=0.3`, slow ramp) with a human on the E-stop.

Design docs: `docs/superpowers/specs/2026-06-16-t1-wbc-hardware-deployment-design.md`
(umbrella) and the stage specs/plans under `docs/superpowers/`.
