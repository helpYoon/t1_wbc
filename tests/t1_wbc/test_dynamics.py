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
    assert d["actuated_dof"].shape == (model.nu,)
    assert isinstance(d["x_half"], float) and isinstance(d["y_half"], float)
    # no torch tensors anywhere in the dict
    import sys
    assert "torch" not in sys.modules
