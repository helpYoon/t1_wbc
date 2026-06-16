import numpy as np
from scipy.spatial.transform import Rotation as R
from t1_wbc.model import load_t1_model, build_index_maps
from t1_wbc.estimator import StateEstimator

def _est():
    model, _ = load_t1_model()
    return StateEstimator(model, build_index_maps(model))

def test_initial_yaw_is_zeroed():
    est = _est()
    est.update_imu(rpy=[0.0, 0.1, 1.3], gyro=[0, 0, 0], acc=[0, 0, 9.81], t=0.0)
    q = est.quat_xyzw()                       # xyzw
    yaw, pitch, roll = R.from_quat(q).as_euler("ZYX")
    assert abs(yaw) < 1e-9                     # startup yaw subtracted
    assert abs(pitch - 0.1) < 1e-9 and abs(roll - 0.0) < 1e-9

def test_yaw_is_relative_to_first_sample():
    est = _est()
    est.update_imu([0, 0, 1.3], [0, 0, 0], [0, 0, 9.81], 0.0)
    est.update_imu([0, 0, 1.3 + 0.4], [0, 0, 0], [0, 0, 9.81], 0.002)
    yaw = R.from_quat(est.quat_xyzw()).as_euler("ZYX")[0]
    assert abs(yaw - 0.4) < 1e-9

def test_ang_vel_passthrough():
    est = _est()
    est.update_imu([0, 0, 0], gyro=[0.1, -0.2, 0.3], acc=[0, 0, 9.81], t=0.0)
    np.testing.assert_allclose(est.ang_vel(), [0.1, -0.2, 0.3])
