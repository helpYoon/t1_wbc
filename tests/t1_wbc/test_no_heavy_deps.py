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
