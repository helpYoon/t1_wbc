import pathlib
SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "t1_wbc"

def test_no_warp_or_themis_or_mjlab_imports():
    bad = []
    for f in SRC.glob("*.py"):
        text = f.read_text()
        for tok in ("mujoco_warp", "themis_mpc", "mjlab", "import warp"):
            if tok in text:
                bad.append(f"{f.name}: {tok}")
    assert not bad, bad

def test_no_batched_files():
    assert not (SRC / "dynamics_warp.py").exists()


def test_core_import_pulls_no_torch():
    import importlib, sys
    for m in ("t1_wbc", "t1_wbc.config", "t1_wbc.model", "t1_wbc.dynamics",
              "t1_wbc.wbc_qp", "t1_wbc.solver", "t1_wbc.targets", "t1_wbc.reference",
              "t1_wbc.controller", "t1_wbc.action_backend", "t1_wbc.run"):
        importlib.import_module(m)
    assert "torch" not in sys.modules
    assert "themis_mpc" not in sys.modules
    assert "mujoco_warp" not in sys.modules
