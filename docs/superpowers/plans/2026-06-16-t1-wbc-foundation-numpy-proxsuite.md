# t1_wbc Foundation — Self-Contained numpy + proxsuite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `t1_wbc` into a self-contained, torch-free repository whose B=1 sim controller runs on numpy + proxsuite and reproduces the current tracking baseline (upright, ≈1.5 cm hand RMS, 0 infeasible).

**Architecture:** Delete all batching (mujoco_warp / GPU / mjlab) and the torch + themis_mpc dependencies; rewrite the per-tick hot path (`dynamics`, `wbc_qp`, `solver`, `targets`, `reference.sample`, `controller`) in numpy with the leading batch dim removed; make proxsuite the sole, warm-started QP backend; vendor the MuJoCo model + sample motion into the repo and resolve them via `importlib.resources`. Correctness is guarded by a **golden-QP equivalence fixture** captured from the current torch code, plus an end-to-end **tracking regression**.

**Tech Stack:** Python 3.13, numpy, mujoco==3.6.0, scipy, proxsuite, pytest. (No torch, no mujoco-warp, no jax, no themis_mpc after this plan.)

**Spec:** `docs/superpowers/specs/2026-06-16-t1-wbc-hardware-deployment-design.md` (§3, §5, §6, §10 — the self-contained + de-torch scope. §7–§9 hardware are Plan 2.)

**Branch:** Execute on a feature branch off `main` (e.g. `t1-wbc-self-contained`), or an isolated worktree via `superpowers:using-git-worktrees`. Do NOT work directly on `main`.

**`<uv>` in run commands** = the project interpreter `/home/yoonwoo/.holosoma_deps/miniconda3/envs/hsmujoco/bin/uv`. After Task 2's `pip install -e .` into a venv, commands can equivalently be the venv's `python`/`pytest` directly.

**Convention for de-torch tasks:** The math is **unchanged** — these tasks transform torch ops to numpy with the batch dim removed (`torch.bmm(A,B)`→`A @ B`; `x.transpose(1,2)`→`x.swapaxes(-1,-2)` / `.T`; `torch.as_tensor(a)`→`np.asarray(a)`; drop every leading `B`/`[0]`/`.unsqueeze(0)`; `torch.zeros`→`np.zeros`; `.cpu().numpy()` boundaries vanish). The **golden-QP equivalence test (Task 1 / Task 7)** is the hard acceptance gate for every de-torch step — if the matrices match to `1e-9`, the transform is correct.

---

## File Structure (Plan 1 target)

```
t1_wbc/                              # repo root (extractable)
├── pyproject.toml                   # NEW — name=t1_wbc; deps numpy,mujoco==3.6.0,scipy,proxsuite
├── README.md                        # MODIFY — drop batched/uv sections
├── src/t1_wbc/
│   ├── assets/                      # package data (ships inside the wheel)
│   │   ├── robot/t1.xml + meshes/   # VENDORED from t1_controller
│   │   └── motion_plan.pkl          # VENDORED from t1_controller/data
│   ├── __init__.py
│   ├── _assets.py                   # NEW — importlib.resources paths
│   ├── config.py                    # MODIFY — package-relative paths; drop batched fields
│   ├── model.py                     # MODIFY — drop warp loaders; keep load_t1_model + index maps
│   ├── dynamics.py                  # MODIFY — numpy extract; drop torch wrap; drop warp
│   ├── solver.py                    # REWRITE — proxsuite-only numpy, warm-started
│   ├── wbc_qp.py                    # REWRITE — numpy assemble + recover
│   ├── targets.py                   # MODIFY — numpy
│   ├── reference.py                 # MODIFY — numpy sample path
│   ├── controller.py               # MODIFY — WBController only, numpy, precomputed constants
│   ├── action_backend.py            # MODIFY — numpy MuJoCoBackend (BoosterSdkBackend stub stays)
│   ├── run.py                       # MODIFY — sim modes only
│   └── logging_utils.py             # unchanged
└── tests/t1_wbc/
    ├── fixtures/golden_qp.npz       # NEW — captured from current torch code (Task 1)
    ├── conftest.py                  # NEW — shared model/controller fixtures
    ├── test_packaging.py            # NEW
    ├── test_no_heavy_deps.py        # NEW
    ├── test_solver.py               # NEW
    ├── test_dynamics.py             # NEW
    ├── test_qp_equivalence.py       # NEW
    └── test_regression_track.py     # NEW
```

**Deleted:** `dynamics_warp.py`; `BatchedWarpController`/`MjlabBatchedWarpController`/`_warp`/`wp_*` (controller.py); `load_t1_warp_model`/`build_warp_handles`/`T1_NCONMAX`/`T1_NJMAX` (model.py); `run_batched_balance`/`run_mjlab_balance`/`benchmark_throughput` + batched CLI (run.py); `backend/device/dtype/num_envs/admm_*` (config.py); `AdmmBackend`/`_per_env_residual`/themis import (solver.py).

---

## Task 1: Capture the golden-QP equivalence fixture (from the CURRENT torch code)

Do this **first**, before any edit, so we have an oracle for the de-torch.

**Files:**
- Create: `scripts/capture_golden_qp.py`
- Create: `tests/t1_wbc/fixtures/golden_qp.npz`

- [ ] **Step 1: Write the capture script**

```python
# scripts/capture_golden_qp.py
"""Capture a deterministic QP from the CURRENT (torch) t1_wbc on a fixed tick.
Run once, before the de-torch, to produce the equivalence oracle."""
import numpy as np, mujoco, torch
from t1_wbc.model import load_t1_model, build_index_maps
from t1_wbc.config import WBCConfig
from t1_wbc.controller import WBController
from t1_wbc.reference import ReferenceTrajectory
from t1_wbc.wbc_qp import assemble_wbc_qp, recover_tau

cfg = WBCConfig()
model, data = load_t1_model(cfg.xml)
ctrl = WBController(model, cfg)
ctrl.reset(data); ctrl.settle(data)   # current settle() signature is settle(self, data)
ref = ReferenceTrajectory(model, build_index_maps(model), ctrl.q_home, cfg, 0.0, 0.0, 0.0)
ctrl.attach_reference(ref)
dt = model.opt.timestep
# advance to a fixed, non-trivial tick (t=1.0 s of reference) so arms/CoM are off-home
for i in range(int(1.0 / dt)):
    mujoco.mj_step1(model, data)
    cmd, _ = ctrl.step_track(data, i * dt)
    from t1_wbc.action_backend import MuJoCoBackend
    MuJoCoBackend(model).apply(ctrl._last, data)
    mujoco.mj_step2(model, data)

d = ctrl.dyn.extract(data)
rs = ctrl.ref.sample(1.0)
from t1_wbc.targets import tracking_targets_from_refsample
q_act, qd_act = ctrl._act_state(data)
tg = tracking_targets_from_refsample(rs, q_act, qd_act, ctrl.q_home, dtype=torch.float64)
qp = assemble_wbc_qp(d, tg, cfg, ctrl.ctrlrange, ctrl.nv, ctrl.nu)
z, ok = ctrl.solver.solve(qp)
tau = recover_tau(z, d, cfg, ctrl.nv)

def np1(x): return x.detach().cpu().numpy()[0] if torch.is_tensor(x) else np.asarray(x)
np.savez("tests/t1_wbc/fixtures/golden_qp.npz",
         H=np1(qp.H), g=np1(qp.g), A_eq=np1(qp.A_eq), b_eq=np1(qp.b_eq),
         G=np1(qp.G), b=np1(qp.b), z=np1(z), tau=np1(tau),
         # raw inputs so the numpy assemble can be re-driven identically:
         **{f"dyn__{k}": np1(v) for k, v in d.items() if torch.is_tensor(v)},
         x_half=np.float64(d["x_half"]), y_half=np.float64(d["y_half"]),
         adof=d["actuated_dof"].cpu().numpy(),
         ctrlrange=ctrl.ctrlrange.cpu().numpy(),
         q_ref=np1(tg.q_ref), qd_ref=np1(tg.qd_ref), tracked=np1(tg.tracked),
         q_act=np1(tg.q_act), qd_act=np1(tg.qd_act), com_ref=np1(tg.com_ref),
         lh_pos=np1(tg.lh_pos), lh_quat=np1(tg.lh_quat_xyzw), lh_vel=np1(tg.lh_vel),
         rh_pos=np1(tg.rh_pos), rh_quat=np1(tg.rh_quat_xyzw), rh_vel=np1(tg.rh_vel),
         base_quat=np1(tg.base_quat_xyzw), base_omega=np1(tg.base_omega_world))
print("wrote golden_qp.npz")
```

- [ ] **Step 2: Run it**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 <uv> run python scripts/capture_golden_qp.py`
Expected: prints `wrote golden_qp.npz`; file exists.

- [ ] **Step 3: Sanity-check the fixture**

Run: `<uv> run python -c "import numpy as np; d=np.load('tests/t1_wbc/fixtures/golden_qp.npz'); print(sorted(d.files)); print('nz', d['H'].shape, 'infeas', np.max(np.maximum(d['G']@d['z']-d['b'],0)))"`
Expected: lists all keys; `H` is `(nz,nz)`; constraint violation ≈ 0 (the captured solution is feasible).

- [ ] **Step 4: Commit**

```bash
git add scripts/capture_golden_qp.py tests/t1_wbc/fixtures/golden_qp.npz
git commit -m "test: capture golden QP fixture from current torch t1_wbc"
```

---

## Task 2: Repo scaffolding — pyproject, src/ layout, vendored assets, package-relative paths

**Files:**
- Create: `pyproject.toml`, `src/t1_wbc/_assets.py`
- Move: every `src/t1_wbc/*.py` stays under `src/t1_wbc/` (already there); add repo-root `pyproject.toml`, `assets/`
- Modify: `src/t1_wbc/config.py:5-7`
- Test: `tests/t1_wbc/test_packaging.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/t1_wbc/test_packaging.py
import os, t1_wbc
from t1_wbc.config import WBCConfig
from t1_wbc.model import load_t1_model

def test_assets_resolve_inside_repo():
    cfg = WBCConfig()
    assert os.path.exists(cfg.xml), cfg.xml
    assert os.path.exists(cfg.motion), cfg.motion
    # no absolute path into a sibling repo
    assert "t1_controller" not in cfg.xml and "t1_controller" not in cfg.motion

def test_model_loads_from_vendored_xml():
    model, data = load_t1_model(WBCConfig().xml)
    assert model.nu == 29 and model.nv == 35
```

- [ ] **Step 2: Run it (fails — paths still point at t1_controller)**

Run: `<uv> run pytest tests/t1_wbc/test_packaging.py -q`
Expected: FAIL (`t1_controller` in path / file under sibling repo).

- [ ] **Step 3: Vendor assets**

```bash
SRC=/home/yoonwoo/humanoid_mpc_ws/src/t1_controller/robot_models/booster_t1/t1_description
mkdir -p src/t1_wbc/assets/robot
cp "$SRC/urdf/t1.xml" src/t1_wbc/assets/robot/t1.xml
cp -r "$SRC/meshes" src/t1_wbc/assets/robot/meshes
cp /home/yoonwoo/humanoid_mpc_ws/src/t1_controller/data/motion_plan.pkl src/t1_wbc/assets/motion_plan.pkl
```
Then fix mesh paths inside `src/t1_wbc/assets/robot/t1.xml`: ensure `<compiler meshdir=...>` (or the `<asset>` mesh `file=`) points at `meshes/` relative to the xml. Run `<uv> run python -c "import mujoco; mujoco.MjModel.from_xml_path('src/t1_wbc/assets/robot/t1.xml')"` and resolve any mesh-not-found by editing `meshdir`.

- [ ] **Step 4: Add the assets path helper**

```python
# src/t1_wbc/_assets.py
"""Package-relative asset paths (self-contained — assets ship inside the package)."""
from importlib.resources import files

def asset(*parts: str) -> str:
    """Absolute path to a vendored asset, e.g. asset('robot', 't1.xml')."""
    return str(files("t1_wbc").joinpath("assets", *parts))
```

- [ ] **Step 5: Point config at the vendored assets**

In `src/t1_wbc/config.py`, replace lines 5-7:
```python
from ._assets import asset
T1_XML = asset("robot", "t1.xml")
PKL = asset("motion_plan.pkl")
```

- [ ] **Step 6: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "t1_wbc"
version = "0.1.0"
description = "Self-contained whole-body QP controller for the Booster T1"
requires-python = ">=3.10"
dependencies = ["numpy", "mujoco==3.6.0", "scipy", "proxsuite>=0.7.3"]

[project.optional-dependencies]
hardware = ["booster-robotics-sdk"]   # exact name resolved in Plan 2

[project.scripts]
t1-wbc = "t1_wbc.run:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
t1_wbc = ["assets/robot/t1.xml", "assets/robot/meshes/*", "assets/motion_plan.pkl"]
```

- [ ] **Step 7: Run the test (passes)**

Run: `<uv> run pytest tests/t1_wbc/test_packaging.py -q`
Expected: PASS (both tests).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/t1_wbc/_assets.py src/t1_wbc/config.py assets tests/t1_wbc/test_packaging.py
git commit -m "feat: vendor T1 model + motion and resolve assets package-relative"
```

---

## Task 3: Delete batching

**Files:**
- Delete: `src/t1_wbc/dynamics_warp.py`
- Modify: `controller.py`, `model.py`, `run.py`, `config.py`
- Test: `tests/t1_wbc/test_no_heavy_deps.py` (partial — full torch check is Task 11)

- [ ] **Step 1: Write the failing test**

```python
# tests/t1_wbc/test_no_heavy_deps.py
import subprocess, sys, pathlib
SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "t1_wbc"

def test_no_warp_or_themis_or_mjlab_imports():
    bad = []
    for f in SRC.glob("*.py"):
        t = f.read_text()
        for tok in ("mujoco_warp", "themis_mpc", "mjlab", "import warp"):
            if tok in t:
                bad.append(f"{f.name}: {tok}")
    assert not bad, bad

def test_no_batched_files():
    assert not (SRC / "dynamics_warp.py").exists()
```

- [ ] **Step 2: Run it (fails)**

Run: `<uv> run pytest tests/t1_wbc/test_no_heavy_deps.py::test_no_warp_or_themis_or_mjlab_imports -q`
Expected: FAIL (dynamics_warp.py + batched controller/model reference warp/themis).

- [ ] **Step 3: Delete the batched code**

```bash
git rm src/t1_wbc/dynamics_warp.py
```
Then edit:
- `controller.py`: delete `class BatchedWarpController`, `class MjlabBatchedWarpController`, and the `_warp()/wp_to_torch/wp_from_torch_f32/mjw_forward/mjw_step` helpers and their imports (`from .model import (... load_t1_warp_model, build_warp_handles, T1_NCONMAX, T1_NJMAX)` → keep only `build_index_maps, load_t1_model` once model.py is trimmed; `from .dynamics_warp import WarpDynamics`; `from .solver import ... AdmmBackend`). Keep `WBController`.
- `model.py`: delete `load_t1_warp_model`, `build_warp_handles`, `T1_NCONMAX`, `T1_NJMAX` and any `import` of warp.
- `run.py`: delete `run_batched_balance`, `run_mjlab_balance`, `benchmark_throughput`; in `main()` remove the `batched-balance/mjlab-balance/benchmark` choices and the `--num-envs/--device/--admm-max-iter/--batch-sizes` args and their handling; keep `balance` and `track`.
- `config.py`: delete fields `backend, device, dtype, num_envs, admm_max_iter, admm_rho, admm_rho_eq, admm_eps_abs, admm_eps_rel, admm_feas_tol`.

- [ ] **Step 4: Run the test (the warp/themis part passes; import may still fail until Task 4)**

Run: `<uv> run pytest tests/t1_wbc/test_no_heavy_deps.py::test_no_batched_files tests/t1_wbc/test_no_heavy_deps.py::test_no_warp_or_themis_or_mjlab_imports -q`
Expected: `test_no_batched_files` PASS; the import-token test PASS (note: `solver.py` still imports `themis_mpc` — fix in Task 4; if this token test fails only on `solver.py`, that's expected and cleared next task).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: delete batching (warp/mjlab/batched controllers + config)"
```

---

## Task 4: Rewrite the solver — proxsuite-only, numpy, warm-started

**Files:**
- Rewrite: `src/t1_wbc/solver.py`
- Test: `tests/t1_wbc/test_solver.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/t1_wbc/test_solver.py
import numpy as np
from t1_wbc.solver import ProxsuiteSolver
from t1_wbc.wbc_qp import BatchedQP   # now a numpy dataclass

def test_box_qp_against_closed_form():
    # min 1/2 x'x - g'x  s.t.  x <= b  (one-sided), no eq -> x = clip(g, ., b)
    n = 4
    qp = BatchedQP(H=np.eye(n), g=np.array([1., 2., 3., 4.]),
                   A_eq=np.zeros((0, n)), b_eq=np.zeros(0),
                   G=np.eye(n), b=np.array([10., 1.5, 10., 10.]))
    s = ProxsuiteSolver()
    z, ok = s.solve(qp)
    assert ok
    np.testing.assert_allclose(z, [1., 1.5, 3., 4.], atol=1e-6)

def test_equality_constrained():
    n = 2
    qp = BatchedQP(H=np.eye(n), g=np.zeros(n),
                   A_eq=np.array([[1., 1.]]), b_eq=np.array([2.]),
                   G=np.zeros((0, n)), b=np.zeros(0))
    z, ok = ProxsuiteSolver().solve(qp)
    assert ok
    np.testing.assert_allclose(z, [1., 1.], atol=1e-6)  # min ||x|| s.t. sum=2
```

- [ ] **Step 2: Run it (fails — module/class not defined)**

Run: `<uv> run pytest tests/t1_wbc/test_solver.py -q`
Expected: FAIL (ImportError).

- [ ] **Step 3: Rewrite `solver.py`**

```python
# src/t1_wbc/solver.py
"""WBC QP solver — proxsuite (dense, exact, warm-started). numpy-native."""
import numpy as np
from proxsuite import proxqp


def external_residual(z, A_eq, b_eq, G, b):
    eq = float(np.max(np.abs(A_eq @ z - b_eq))) if A_eq.shape[0] else 0.0
    viol = float(np.max(np.maximum(G @ z - b, 0.0))) if G.shape[0] else 0.0
    return eq, viol


class ProxsuiteSolver:
    """Persistent dense proxsuite QP, warm-started across ticks.
    Solves: min 1/2 z'Hz + g'z  s.t.  A_eq z = b_eq,  -inf <= G z <= b."""
    def __init__(self, feas_tol: float = 1e-4):
        self.feas_tol = feas_tol
        self._qp = None
        self._dims = None

    def solve(self, qp):
        n = qp.H.shape[0]; n_eq = qp.A_eq.shape[0]; n_in = qp.G.shape[0]
        dims = (n, n_eq, n_in)
        lo = np.full(n_in, -1e20)
        if self._qp is None or self._dims != dims:
            self._qp = proxqp.dense.QP(n, n_eq, n_in)
            self._qp.init(qp.H, qp.g, qp.A_eq, qp.b_eq, qp.G, lo, qp.b)
            self._dims = dims
        else:
            self._qp.update(H=qp.H, g=qp.g, A=qp.A_eq, b=qp.b_eq, C=qp.G, l=lo, u=qp.b)
        self._qp.solve()                       # warm-started from the previous result
        z = np.asarray(self._qp.results.x, dtype=np.float64)
        eq, viol = external_residual(z, qp.A_eq, qp.b_eq, qp.G, qp.b)
        ok = (eq < self.feas_tol) and (viol < self.feas_tol)
        return z, ok


def make_solver(cfg=None):
    return ProxsuiteSolver()
```

- [ ] **Step 4: Run the test (passes)**

Run: `<uv> run pytest tests/t1_wbc/test_solver.py -q`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add src/t1_wbc/solver.py tests/t1_wbc/test_solver.py
git commit -m "feat: numpy proxsuite-only solver, warm-started across ticks"
```

---

## Task 5: De-torch dynamics extraction

**Files:**
- Modify: `src/t1_wbc/dynamics.py` (`CpuDynamics.extract` returns numpy)
- Test: `tests/t1_wbc/test_dynamics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/t1_wbc/test_dynamics.py
import numpy as np, mujoco
from t1_wbc.model import load_t1_model, build_index_maps
from t1_wbc.dynamics import CpuDynamics

def test_extract_returns_numpy_with_correct_shapes():
    model, data = load_t1_model()
    mujoco.mj_resetDataKeyframe(model, data, 0); mujoco.mj_forward(model, data)
    d = CpuDynamics(model, build_index_maps(model)).extract(data)
    assert isinstance(d["M"], np.ndarray) and d["M"].shape == (model.nv, model.nv)
    np.testing.assert_allclose(d["M"], d["M"].T, atol=1e-9)          # symmetric
    assert d["Jcom"].shape == (3, model.nv)
    assert d["Jfoot_L"].shape == (6, model.nv) and d["Jfoot_R"].shape == (6, model.nv)
    assert d["h"].shape == (model.nv,) and d["qvel"].shape == (model.nv,)
    assert d["base_quat_xyzw"].shape == (4,)
```

(`load_t1_model()` must default to the vendored xml — add that default in model.py if absent.)

- [ ] **Step 2: Run it (fails — extract returns `(1,...)` torch tensors)**

Run: `<uv> run pytest tests/t1_wbc/test_dynamics.py -q`
Expected: FAIL (torch tensor / shape `(1, nv, nv)`).

- [ ] **Step 3: De-torch `CpuDynamics`**

In `dynamics.py`: delete `import torch`. Change `CpuDynamics.__init__` to store `self._adof = np.asarray(handles["actuated_dof"])`. Rewrite `extract` to return the existing numpy `extract_dynamics(...)` dict directly (no `torch.as_tensor(...).unsqueeze(0)`), adding `qvel`, `tau_fric_coeff`, `actuated_dof`, `x_half`, `y_half`:
```python
    def extract(self, data):
        raw = extract_dynamics(self.model, data, self.handles)   # numpy dict
        raw["qvel"] = data.qvel.copy()
        raw["actuated_dof"] = self._adof
        raw["x_half"] = self.handles["x_half"]; raw["y_half"] = self.handles["y_half"]
        return raw
```
(`extract_dynamics` is already numpy; leave it unchanged. `dtype` arg on `CpuDynamics` is removed.)

- [ ] **Step 4: Run the test (passes)**

Run: `<uv> run pytest tests/t1_wbc/test_dynamics.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/t1_wbc/dynamics.py src/t1_wbc/model.py tests/t1_wbc/test_dynamics.py
git commit -m "refactor: CpuDynamics.extract returns numpy (drop torch wrap)"
```

---

## Task 6: De-torch the QP core (`wbc_qp.py`) + numpy `Targets`

**Files:**
- Rewrite: `src/t1_wbc/wbc_qp.py`
- Modify: `src/t1_wbc/targets.py` (numpy `Targets`)
- Test: covered by Task 7's equivalence test

- [ ] **Step 1: Make `BatchedQP` and `ori_error_world` numpy**

In `wbc_qp.py`: delete `import torch`; `import numpy as np`. Rewrite the dataclass field types to numpy and `ori_error_world` to numpy:
```python
@dataclass
class BatchedQP:
    H: np.ndarray; g: np.ndarray          # (nz,nz),(nz,)
    A_eq: np.ndarray; b_eq: np.ndarray    # (neq,nz),(neq,)
    G: np.ndarray; b: np.ndarray          # (nineq,nz),(nineq,)

def ori_error_world(q_des_xyzw, q_cur_xyzw):
    """world-frame orientation error 2·log(q_des ⊗ q_cur⁻¹) -> (3,) rotation vector."""
    qd = np.asarray(q_des_xyzw); qc = np.asarray(q_cur_xyzw)
    cx, cy, cz, cw = -qc[0], -qc[1], -qc[2], qc[3]
    dx, dy, dz, dw = qd[0], qd[1], qd[2], qd[3]
    ew = dw*cw - dx*cx - dy*cy - dz*cz
    ex = dw*cx + dx*cw + dy*cz - dz*cy
    ey = dw*cy - dx*cz + dy*cw + dz*cx
    ez = dw*cz + dx*cy - dy*cx + dz*cw
    sgn = -1.0 if ew < 0 else 1.0
    ew, ex, ey, ez = ew*sgn, ex*sgn, ey*sgn, ez*sgn
    vnorm = np.sqrt(ex*ex + ey*ey + ez*ez)
    angle = 2.0 * np.arctan2(vnorm, ew)
    scale = angle / vnorm if vnorm > 1e-12 else 0.0
    return np.array([ex*scale, ey*scale, ez*scale])
```

- [ ] **Step 2: Rewrite the helpers + `assemble_wbc_qp` + `recover_tau` in numpy**

Transform the existing torch functions (`_contact_transpose`, `_finite_contact_rows`, `_torque_rows`, `assemble_wbc_qp`, `recover_tau` — the current `src/t1_wbc/wbc_qp.py:33-142`) line-for-line per the convention (drop `B`; `torch.bmm`→`@`; `.transpose(1,2)`→`.T`; `torch.zeros`→`np.zeros`; `torch.as_tensor`→`np.asarray`; the posture scatter becomes `H[adof, adof] += w_j; g[adof] += -(w_j*vdot_j)` with `adof = np.asarray(dyn["actuated_dof"])`). Shapes after transform: `_contact_transpose`→`(nv,12)`; `_finite_contact_rows`→`(m,nz)`; `_torque_rows`→`(2nu,nz)`; `A_base`→`(6,nz)`, `A_feet`→`(12,nz)`. The `_add_task` becomes:
```python
    def _add_task(J, a_des, w):
        JWl = J.T * w
        H[:nv, :nv] += JWl @ J
        g[:nv] += -(JWl @ a_des)
```
and CoM/hand/base-ori tasks call it. Use `cfg.reg_psd` for the final jitter (`H = 0.5*(H+H.T) + cfg.reg_psd*np.eye(nz)`) and gate `tau_fric` on `cfg.friction_ff` (already in config). `recover_tau` returns `(M @ z[:nv] + h + tau_fric - JcT @ z[nv:])[adof]`.

- [ ] **Step 3: Make `targets.Targets` numpy**

In `targets.py`: delete torch; `Targets` fields are numpy arrays; `balance_targets`/`tracking_targets_from_refsample` drop the `dtype`/batch args and `.unsqueeze(0)`, returning `(nu,)`/`(3,)`/`(4,)` numpy arrays. `tracked` is a bool numpy array.

- [ ] **Step 4: Defer test to Task 7** (the equivalence test drives steps 2-3). Commit after Task 7 passes.

---

## Task 7: QP equivalence test (acceptance gate for Tasks 5-6)

**Files:**
- Create: `tests/t1_wbc/test_qp_equivalence.py`

- [ ] **Step 1: Write the test**

```python
# tests/t1_wbc/test_qp_equivalence.py
import numpy as np
from t1_wbc.config import WBCConfig
from t1_wbc.wbc_qp import assemble_wbc_qp, recover_tau, BatchedQP
from t1_wbc.targets import Targets

F = np.load("tests/t1_wbc/fixtures/golden_qp.npz")

def _dyn():
    d = {k[len("dyn__"):]: F[k] for k in F.files if k.startswith("dyn__")}
    d["x_half"] = float(F["x_half"]); d["y_half"] = float(F["y_half"])
    d["actuated_dof"] = F["adof"]
    return d

def _targets():
    return Targets(q_ref=F["q_ref"], qd_ref=F["qd_ref"], tracked=F["tracked"].astype(bool),
                   q_act=F["q_act"], qd_act=F["qd_act"], com_ref=F["com_ref"],
                   lh_pos=F["lh_pos"], lh_quat_xyzw=F["lh_quat"], lh_vel=F["lh_vel"],
                   rh_pos=F["rh_pos"], rh_quat_xyzw=F["rh_quat"], rh_vel=F["rh_vel"],
                   base_quat_xyzw=F["base_quat"], base_omega_world=F["base_omega"],
                   q_home=F["q_ref"])

def test_numpy_assemble_matches_torch_golden():
    d = _dyn(); nv = d["M"].shape[0]; nu = d["actuated_dof"].shape[0]
    qp = assemble_wbc_qp(d, _targets(), WBCConfig(), F["ctrlrange"], nv, nu)
    np.testing.assert_allclose(qp.H, F["H"], atol=1e-9)
    np.testing.assert_allclose(qp.g, F["g"], atol=1e-9)
    np.testing.assert_allclose(qp.A_eq, F["A_eq"], atol=1e-9)
    np.testing.assert_allclose(qp.b_eq, F["b_eq"], atol=1e-9)
    np.testing.assert_allclose(qp.G, F["G"], atol=1e-9)
    np.testing.assert_allclose(qp.b, F["b"], atol=1e-9)

def test_recover_tau_matches_golden():
    d = _dyn(); nv = d["M"].shape[0]
    tau = recover_tau(F["z"], d, WBCConfig(), nv)
    np.testing.assert_allclose(tau, F["tau"], atol=1e-9)
```

(If `Targets` field names differ, align the test to the dataclass defined in Task 6 — keep names identical between Task 6 and this test.)

- [ ] **Step 2: Run (fails until Task 6 implemented; then iterate Task 6 until green)**

Run: `<uv> run pytest tests/t1_wbc/test_qp_equivalence.py -q`
Expected: ultimately PASS — numpy QP matches the torch golden to `1e-9`.

- [ ] **Step 3: Commit Tasks 6+7 together**

```bash
git add src/t1_wbc/wbc_qp.py src/t1_wbc/targets.py tests/t1_wbc/test_qp_equivalence.py
git commit -m "refactor: numpy wbc_qp + targets, verified equal to torch golden (1e-9)"
```

---

## Task 8: De-torch `reference.sample` and `controller.WBController` (+ precompute constants)

**Files:**
- Modify: `src/t1_wbc/reference.py` (`sample` returns numpy `RefSample`)
- Modify: `src/t1_wbc/controller.py` (numpy; precompute friction-cone/selector/weights once)
- Modify: `src/t1_wbc/action_backend.py` (`MuJoCoBackend` numpy)
- Test: `tests/t1_wbc/conftest.py` + a step smoke in `test_regression_track.py` (Task 9)

- [ ] **Step 1: De-torch `reference.RefSample`/`sample`**

`reference.py` is already numpy internally; in `sample()` return a `RefSample` of numpy arrays (it already builds numpy arrays — just drop any torch coercion in `__init__` where `q_home` may arrive as torch: replace the `hasattr(q_home, "detach")` branch with `q_home = np.asarray(q_home, dtype=np.float64).reshape(-1)`).

- [ ] **Step 2: De-torch `WBController`**

In `controller.py`: delete `import torch`; numpy throughout. `self.ctrlrange = np.asarray(model.actuator_ctrlrange)`. `_act_state` returns numpy `(nu,)`. `_solve_to_cmd` builds the numpy QP, solves, `recover_tau`, integrates `q_des = q + qd*dt + 0.5*vdot*dt**2` (numpy), packs `JointCommand` of numpy arrays. `settle` already numpy. **Precompute once in `__init__`** the loop-invariant constants and pass them into `assemble_wbc_qp` (or compute inside `assemble` but cache on the controller): friction-cone `Gc,bc` (depends only on `cfg.mu, cfg.fz_min, x_half, y_half`), `np.eye(nv/12/nz)`, the base-orientation selector. Wire `assemble_wbc_qp` to accept the precomputed `Gc,bc` (add optional args defaulting to building them) so the equivalence test (Task 7) still passes.

- [ ] **Step 3: De-torch `MuJoCoBackend`**

```python
class MuJoCoBackend(ActionBackend):
    def __init__(self, model):
        self.model = model
        self.lo = model.actuator_ctrlrange[:, 0].copy(); self.hi = model.actuator_ctrlrange[:, 1].copy()
    def apply(self, cmd, data):
        nu = self.model.nu
        q = data.qpos[7:7+nu]; qd = data.qvel[6:6+nu]
        tau = cmd.kp*(cmd.q_des - q) + cmd.kd*(cmd.qd_des - qd) + cmd.tau_ff
        data.ctrl[:] = np.clip(tau, self.lo, self.hi)
```
and make `JointCommand` fields numpy `(nu,)` arrays (drop the `(B,nu)` docstring/shape).

- [ ] **Step 4: Smoke — one tracked tick produces a numpy command**

Add to `tests/t1_wbc/conftest.py`:
```python
import pytest, mujoco
from t1_wbc.model import load_t1_model, build_index_maps
from t1_wbc.config import WBCConfig
from t1_wbc.controller import WBController
from t1_wbc.reference import ReferenceTrajectory

@pytest.fixture
def tracking_controller():
    cfg = WBCConfig(); model, data = load_t1_model(cfg.xml)
    ctrl = WBController(model, cfg); ctrl.reset(data); ctrl.settle(data)
    ref = ReferenceTrajectory(model, build_index_maps(model), ctrl.q_home, cfg, 0.0, 0.0, 0.0)
    ctrl.attach_reference(ref)
    return cfg, model, data, ctrl
```
```python
# in test_regression_track.py
import numpy as np
def test_one_track_tick_is_numpy(tracking_controller):
    cfg, model, data, ctrl = tracking_controller
    cmd, diag = ctrl.step_track(data, 0.0)
    assert isinstance(cmd.tau_ff, np.ndarray) and cmd.tau_ff.shape == (model.nu,)
    assert diag["ok"]
```

- [ ] **Step 5: Run the smoke (passes)**

Run: `<uv> run pytest tests/t1_wbc/test_regression_track.py::test_one_track_tick_is_numpy -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/t1_wbc/reference.py src/t1_wbc/controller.py src/t1_wbc/action_backend.py tests/t1_wbc/conftest.py tests/t1_wbc/test_regression_track.py
git commit -m "refactor: numpy WBController + reference.sample + MuJoCoBackend; precompute constants"
```

---

## Task 9: End-to-end tracking regression (the quality gate)

**Files:**
- Modify: `tests/t1_wbc/test_regression_track.py`

- [ ] **Step 1: Write the regression test**

```python
# add to tests/t1_wbc/test_regression_track.py
import numpy as np
from t1_wbc.run import run_track
from t1_wbc.config import WBCConfig

def test_track_reproduces_baseline():
    cfg = WBCConfig()
    out = run_track(cfg, seconds=5.0)            # headless, no viewer
    assert out["upright"] is True
    assert out["infeasible"] == 0
    assert out["min_base_z"] > 0.55              # leans to ~0.58-0.64, never falls
    assert out["lh_rms"] < 0.03 and out["rh_rms"] < 0.03   # ~1-2 cm
```

- [ ] **Step 2: Run it**

Run: `<uv> run pytest tests/t1_wbc/test_regression_track.py::test_track_reproduces_baseline -q`
Expected: PASS — proxsuite/numpy path keeps the robot upright with ≈1-2 cm hand RMS, 0 infeasible (equal-or-better than the torch-ADMM baseline; exact solve).

- [ ] **Step 3: Manual full-motion confirmation**

Run: `<uv> run python -m t1_wbc.run --mode track --log /tmp/post_detorch.csv`
Expected: summary dict with `upright: True`, `infeasible: 0`, `lh_rms`≈0.013-0.017, `rh_rms`≈0.013, comparable to the pre-refactor full-motion numbers.

- [ ] **Step 4: Commit**

```bash
git add tests/t1_wbc/test_regression_track.py
git commit -m "test: end-to-end tracking regression on numpy/proxsuite path"
```

---

## Task 10: Final dependency severance + README

**Files:**
- Modify: `pyproject.toml` (confirm no torch), `README.md`
- Test: `tests/t1_wbc/test_no_heavy_deps.py` (add the torch check)

- [ ] **Step 1: Add the torch-free assertion**

```python
# add to tests/t1_wbc/test_no_heavy_deps.py
def test_core_import_pulls_no_torch():
    import importlib, sys
    for m in ("t1_wbc.controller", "t1_wbc.wbc_qp", "t1_wbc.solver",
              "t1_wbc.dynamics", "t1_wbc.run"):
        importlib.import_module(m)
    assert "torch" not in sys.modules
    assert "themis_mpc" not in sys.modules
```

- [ ] **Step 2: Run it (fails if any stray torch import remains)**

Run: `<uv> run pytest tests/t1_wbc/test_no_heavy_deps.py::test_core_import_pulls_no_torch -q`
Expected: FAIL → grep `git grep -n "import torch" src/t1_wbc` and remove the last stragglers (logging_utils, __init__, run); then PASS.

- [ ] **Step 3: Update README** — remove the `uv`/batched/GPU sections; document `pip install .`, `python -m t1_wbc.run --mode track`, and that the package is torch-free and self-contained.

- [ ] **Step 4: Full test suite green**

Run: `<uv> run pytest tests/t1_wbc -q`
Expected: PASS (packaging, no-heavy-deps, solver, dynamics, qp-equivalence, regression).

- [ ] **Step 5: Fresh-venv self-containment check**

Run:
```bash
python -m venv /tmp/t1venv && /tmp/t1venv/bin/pip install -e . && \
/tmp/t1venv/bin/python -c "import t1_wbc, sys; from t1_wbc.run import run_track; print('torch' in sys.modules)"
```
Expected: installs with only numpy/mujoco/scipy/proxsuite; prints `False`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml README.md tests/t1_wbc/test_no_heavy_deps.py
git commit -m "chore: drop torch dependency; t1_wbc is self-contained + torch-free"
```

---

## Definition of Done (Plan 1)

- `pip install -e .` in a clean venv with only numpy/mujoco/scipy/proxsuite produces a working sim controller; `import t1_wbc` pulls no torch/themis.
- `t1-wbc --mode track` reproduces the baseline (upright, ≈1.3-1.7 cm hand RMS, 0 infeasible).
- `pytest tests/t1_wbc` green (packaging, no-heavy-deps, solver, dynamics, qp-equivalence@1e-9, regression).
- Batching, torch, mujoco-warp, jax, themis_mpc, and all `t1_controller` paths are gone.

**Next:** Plan 2 (`2026-06-16-t1-wbc-hardware-path.md`) — estimator, transport abstraction, `BoosterSdkBackend`, safety + torque-safety (§7–§9 of the spec) — written against the numpy APIs this plan establishes.
