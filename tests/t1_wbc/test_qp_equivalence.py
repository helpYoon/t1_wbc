import numpy as np
from t1_wbc.config import WBCConfig
from t1_wbc.wbc_qp import assemble_wbc_qp, recover_tau
from t1_wbc.targets import Targets

F = np.load("tests/t1_wbc/fixtures/golden_qp.npz")

def _dyn():
    d = {k[len("dyn__"):]: F[k] for k in F.files if k.startswith("dyn__")}
    d["x_half"] = float(F["x_half"]); d["y_half"] = float(F["y_half"])
    d["actuated_dof"] = F["adof"]
    return d

def _targets():
    return Targets(q_home=F["q_home"], q_act=F["q_act"], qd_act=F["qd_act"],
                   com_ref=F["com_ref"], tracked=F["tracked"].astype(bool),
                   q_ref=F["q_ref"], qd_ref=F["qd_ref"],
                   base_quat_xyzw=F["base_quat"], base_omega_world=F["base_omega"],
                   lh_pos=F["lh_pos"], lh_quat_xyzw=F["lh_quat"], lh_vel=F["lh_vel"],
                   rh_pos=F["rh_pos"], rh_quat_xyzw=F["rh_quat"], rh_vel=F["rh_vel"])

def test_numpy_assemble_matches_torch_golden():
    d = _dyn(); nv = d["M"].shape[0]; nu = d["actuated_dof"].shape[0]
    qp = assemble_wbc_qp(d, _targets(), WBCConfig(), F["ctrlrange"], nv, nu)
    for name, got in [("H", qp.H), ("g", qp.g), ("A_eq", qp.A_eq),
                      ("b_eq", qp.b_eq), ("G", qp.G), ("b", qp.b)]:
        np.testing.assert_allclose(got, F[name], atol=1e-9, rtol=1e-9, err_msg=name)

def test_recover_tau_matches_golden():
    d = _dyn(); nv = d["M"].shape[0]
    tau = recover_tau(F["z"], d, WBCConfig(), nv)
    np.testing.assert_allclose(tau, F["tau"], atol=1e-9, rtol=1e-9)
