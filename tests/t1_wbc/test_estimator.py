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

import mujoco
def test_fk_base_z_pins_lower_foot_to_ground():
    model, data = load_t1_model()
    mujoco.mj_resetDataKeyframe(model, data, 0); mujoco.mj_forward(model, data)
    true_base_z = float(data.qpos[2])
    bq = data.qpos[3:7]                                 # wxyz
    rpy = R.from_quat([bq[1], bq[2], bq[3], bq[0]]).as_euler("ZYX")[::-1]  # roll,pitch,yaw
    jq = data.qpos[7:7 + model.nu].copy()
    est = StateEstimator(model, build_index_maps(model))
    est.update_imu(rpy, [0, 0, 0], [0, 0, 9.81], 0.0)
    est.update_base_pose_and_contacts(jq)
    # home keyframe has both feet flat on the ground -> recovered base z ~= true base z
    assert abs(est.position()[2] - true_base_z) < 5e-3
    assert est.contact_flags().tolist() == [True, True]

def test_contact_flag_clears_when_foot_lifted():
    model, data = load_t1_model()
    mujoco.mj_resetDataKeyframe(model, data, 0); mujoco.mj_forward(model, data)
    jq = data.qpos[7:7 + model.nu].copy()
    maps = build_index_maps(model)
    li = maps["name_to_act_index"]["Left_Knee_Pitch"]
    jq[li] += 0.6                                        # bend left knee -> lift left foot
    est = StateEstimator(model, maps)
    est.update_imu([0, 0, 0], [0, 0, 0], [0, 0, 9.81], 0.0)
    est.update_base_pose_and_contacts(jq)
    cf = est.contact_flags().tolist()
    assert cf == [False, True] or cf == [True, False]
