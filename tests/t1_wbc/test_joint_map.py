import numpy as np
from t1_wbc.model import load_t1_model, build_index_maps
from t1_wbc.joint_map import T1_SDK_JOINTS, JointMap

def test_table_is_29_and_indices_0_to_28():
    assert len(T1_SDK_JOINTS) == 29
    idx = sorted(i for _, i in T1_SDK_JOINTS)
    assert idx == list(range(29))

def test_mujoco_to_sdk_roundtrip():
    model, _ = load_t1_model()
    jm = JointMap(build_index_maps(model))
    assert jm.sdk_count == 29 and jm.nu == model.nu
    mj = np.arange(model.nu, dtype=float)          # a distinct value per MuJoCo joint
    sdk = jm.mujoco_to_sdk(mj)                      # (29,)
    back = jm.sdk_to_mujoco(sdk)                    # (nu,)
    np.testing.assert_allclose(back, mj)            # lossless round-trip on shared joints

def test_names_align_with_mujoco():
    model, _ = load_t1_model()
    maps = build_index_maps(model)
    jm = JointMap(maps)
    for name, sdk_idx in T1_SDK_JOINTS:
        if name in maps["name_to_act_index"]:
            assert jm.mj_index_of_sdk[sdk_idx] == maps["name_to_act_index"][name]
