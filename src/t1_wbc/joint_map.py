"""T1 joint-name <-> Booster-SDK index table (7-DOF-arm / 29-DOF variant, indices 0-28),
and the permutation between MuJoCo actuator order and SDK order. Vendored from
t1_controller .../t1_description/T1JointGains.h (kT1Joints). The SDK's vendored Python
binding only names the 23-DOF JointIndex enum, so we use raw integer indices here."""
import numpy as np

# (URDF joint name, SDK index). SERIAL cmd_type interprets indices 21,22,27,28 as ankle
# pitch/roll (serial joints), which is what we want.
T1_SDK_JOINTS = [
    ("AAHead_yaw", 0), ("Head_pitch", 1),
    ("Left_Shoulder_Pitch", 2), ("Left_Shoulder_Roll", 3), ("Left_Elbow_Pitch", 4),
    ("Left_Elbow_Yaw", 5), ("Left_Wrist_Pitch", 6), ("Left_Wrist_Yaw", 7), ("Left_Hand_Roll", 8),
    ("Right_Shoulder_Pitch", 9), ("Right_Shoulder_Roll", 10), ("Right_Elbow_Pitch", 11),
    ("Right_Elbow_Yaw", 12), ("Right_Wrist_Pitch", 13), ("Right_Wrist_Yaw", 14), ("Right_Hand_Roll", 15),
    ("Waist", 16),
    ("Left_Hip_Pitch", 17), ("Left_Hip_Roll", 18), ("Left_Hip_Yaw", 19), ("Left_Knee_Pitch", 20),
    ("Left_Ankle_Pitch", 21), ("Left_Ankle_Roll", 22),
    ("Right_Hip_Pitch", 23), ("Right_Hip_Roll", 24), ("Right_Hip_Yaw", 25), ("Right_Knee_Pitch", 26),
    ("Right_Ankle_Pitch", 27), ("Right_Ankle_Roll", 28),
]
SDK_JOINT_CNT = 29


class JointMap:
    """Permute (nu,) MuJoCo-actuator-order vectors <-> (29,) SDK-order vectors."""
    def __init__(self, index_maps):
        name_to_mj = index_maps["name_to_act_index"]
        self.nu = len(name_to_mj)
        self.sdk_count = SDK_JOINT_CNT
        # mj_index_of_sdk[sdk_idx] = MuJoCo actuator index (or -1 if the model lacks that joint)
        self.mj_index_of_sdk = np.full(SDK_JOINT_CNT, -1, dtype=int)
        for name, sdk_idx in T1_SDK_JOINTS:
            if name in name_to_mj:
                self.mj_index_of_sdk[sdk_idx] = name_to_mj[name]
        self._shared = self.mj_index_of_sdk >= 0     # SDK slots present in the model

    def mujoco_to_sdk(self, v_mj, fill=0.0):
        out = np.full(SDK_JOINT_CNT, fill, dtype=np.float64)
        out[self._shared] = np.asarray(v_mj, dtype=np.float64)[self.mj_index_of_sdk[self._shared]]
        return out

    def sdk_to_mujoco(self, v_sdk):
        out = np.zeros(self.nu, dtype=np.float64)
        v_sdk = np.asarray(v_sdk, dtype=np.float64)
        out[self.mj_index_of_sdk[self._shared]] = v_sdk[self._shared]
        return out
