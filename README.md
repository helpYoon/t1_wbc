# t1_wbc

A self-contained, **torch-free** single-robot whole-body QP controller for the
Booster T1. It runs a per-tick inverse-dynamics QP in MuJoCo to track a recorded
motion plan, built entirely on **numpy + proxsuite** (no RL, no OCS2, no
Pinocchio, no GPU/batched runtime). The robot MJCF and motion plan ship inside
the package, so it has no sibling-repo dependencies.

## Install

```bash
pip install -e .
```

Dependencies: `numpy`, `mujoco==3.6.0`, `scipy`, `proxsuite`. An optional
`[hardware]` extra pulls the Booster SDK (`pip install -e .[hardware]`); the
hardware path is not yet wired (see below).

## Run

```bash
python -m t1_wbc.run --mode track          # settle, then track the motion plan
python -m t1_wbc.run --mode balance        # settle, then hold/balance at home
python -m t1_wbc.run --mode track --seconds 5      # run for N seconds
python -m t1_wbc.run --mode track --log out.csv    # per-tick diagnostics CSV
python -m t1_wbc.run --mode track --no-friction-ff # disable Coulomb friction feedforward
```

(Also installed as the `t1-wbc` console script.)

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/t1_wbc -q
```

## Hardware

The hardware path — the Booster SDK action backend, state estimator, and safety
layer — is specified in
`docs/superpowers/specs/2026-06-16-t1-wbc-hardware-deployment-design.md` and is a
follow-up. The `BoosterSdkBackend` in `action_backend.py` is an intentional stub
until then.
