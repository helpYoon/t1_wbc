import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
from t1_wbc.model import load_t1_model
from t1_wbc.transport import SimTransport, LowState, LowCmd

def _sim():
    model, data = load_t1_model()
    mujoco.mj_resetDataKeyframe(model, data, 0); mujoco.mj_forward(model, data)
    return model, data

def test_read_lowstate_shapes_and_joint_values():
    model, data = _sim()
    ls = SimTransport(model, data).read_lowstate()
    assert isinstance(ls, LowState)
    assert ls.imu_rpy.shape == (3,) and ls.imu_gyro.shape == (3,) and ls.imu_acc.shape == (3,)
    assert ls.joint_q.shape == (model.nu,) and ls.joint_dq.shape == (model.nu,)
    assert ls.odom_xytheta.shape == (3,)
    np.testing.assert_allclose(ls.joint_q, data.qpos[7:7 + model.nu])
    np.testing.assert_allclose(ls.imu_acc, [0, 0, 9.81], atol=0.2)   # ~gravity reaction up, body frame

def test_read_lowstate_imu_matches_base_orientation():
    model, data = _sim()
    ls = SimTransport(model, data).read_lowstate()
    bq = data.qpos[3:7]
    exp = R.from_quat([bq[1], bq[2], bq[3], bq[0]]).as_euler("ZYX")  # yaw,pitch,roll
    np.testing.assert_allclose(ls.imu_rpy, exp[::-1], atol=1e-9)     # rpy = roll,pitch,yaw

def test_write_lowcmd_applies_kcustom_law():
    model, data = _sim()
    nu = model.nu
    tr = SimTransport(model, data)
    cmd = LowCmd(q_des=data.qpos[7:7+nu].copy(), qd_des=np.zeros(nu),
                 kp=np.zeros(nu), kd=np.zeros(nu), tau_ff=np.full(nu, 1.5))
    tr.write_lowcmd(cmd)
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    np.testing.assert_allclose(data.ctrl, np.clip(np.full(nu, 1.5), lo, hi))
