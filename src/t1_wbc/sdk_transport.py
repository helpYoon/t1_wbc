"""Booster-SDK transport: LowState/Odometer subscribe -> LowState; LowCmd publish (SERIAL,
kCustom). The real SDK is imported lazily so the package/tests run without it; tests inject
a mock `_sdk` dict. Joints are remapped MuJoCo<->SDK index 0-28 via JointMap. Fails fast if
the SDK exposes a joint count != 29 (the 23-DOF-binding risk)."""
import time
import numpy as np
from .transport import Transport, LowState, LowCmd
from .joint_map import JointMap, SDK_JOINT_CNT


def _real_sdk():
    import booster_robotics_sdk_python as B
    B.ChannelFactory.Instance().Init(0)
    loco = B.B1LocoClient(); loco.Init()
    pub = B.B1LowCmdPublisher(); pub.InitChannel()
    return dict(low_cmd_pub=pub, loco=loco, LowCmd=B.LowCmd, MotorCmd=B.MotorCmd,
                cmd_type_serial=B.LowCmdType.SERIAL, kPrepare=B.RobotMode.kPrepare,
                kCustom=B.RobotMode.kCustom, kDamping=B.RobotMode.kDamping,
                joint_cnt=B.B1JointCnt, _module=B)


class SdkTransport(Transport):
    def __init__(self, model, index_maps, _sdk=None):
        self.model = model
        self.jm = JointMap(index_maps)
        self._sdk = _sdk if _sdk is not None else _real_sdk()
        self.n = int(self._sdk["joint_cnt"])
        assert self.n == SDK_JOINT_CNT, (
            f"SDK exposes {self.n} joints but the T1 map needs {SDK_JOINT_CNT} (7-DOF arm). "
            "The vendored binding is likely the 23-DOF build — see docs/BRINGUP-hardware.md.")
        self._latest_state = None
        self._latest_odom = None
        self._state_t = 0.0
        if _sdk is None:                       # real run: subscribe (callbacks)
            B = self._sdk["_module"]
            self._state_sub = B.B1LowStateSubscriber(self._on_state); self._state_sub.InitChannel()
            self._odom_sub = B.B1OdometerStateSubscriber(self._on_odom); self._odom_sub.InitChannel()

    def _on_state(self, msg):
        self._latest_state = msg; self._state_t = time.monotonic()

    def _on_odom(self, msg):
        self._latest_odom = msg

    def start(self):
        """kPrepare -> kCustom handshake."""
        self._sdk["loco"].ChangeMode(self._sdk["kPrepare"]); time.sleep(2.0)
        self._sdk["loco"].ChangeMode(self._sdk["kCustom"]); time.sleep(0.5)

    def stop(self):
        if "kDamping" in self._sdk:
            self._sdk["loco"].ChangeMode(self._sdk["kDamping"])

    def state_age(self):
        return time.monotonic() - self._state_t

    def read_lowstate(self) -> LowState:
        st = self._latest_state; od = self._latest_odom
        q_sdk = np.array([st.motor_state_serial[i].q for i in range(self.n)])
        dq_sdk = np.array([st.motor_state_serial[i].dq for i in range(self.n)])
        imu = st.imu_state
        return LowState(imu_rpy=np.array(imu.rpy, dtype=np.float64),
                        imu_gyro=np.array(imu.gyro, dtype=np.float64),
                        imu_acc=np.array(imu.acc, dtype=np.float64),
                        joint_q=self.jm.sdk_to_mujoco(q_sdk),
                        joint_dq=self.jm.sdk_to_mujoco(dq_sdk),
                        odom_xytheta=np.array([od.x, od.y, od.theta], dtype=np.float64))

    def write_lowcmd(self, cmd: LowCmd) -> None:
        q = self.jm.mujoco_to_sdk(cmd.q_des); dq = self.jm.mujoco_to_sdk(cmd.qd_des)
        kp = self.jm.mujoco_to_sdk(cmd.kp); kd = self.jm.mujoco_to_sdk(cmd.kd)
        tau = self.jm.mujoco_to_sdk(cmd.tau_ff)
        n = self.n
        low = self._sdk["LowCmd"](); low.cmd_type = self._sdk["cmd_type_serial"]
        mc = [self._sdk["MotorCmd"]() for _ in range(n)]
        for i in range(n):
            mc[i].q = float(q[i]); mc[i].dq = float(dq[i]); mc[i].tau = float(tau[i])
            mc[i].kp = float(kp[i]); mc[i].kd = float(kd[i]); mc[i].weight = 0.0
        low.motor_cmd = mc
        self._sdk["low_cmd_pub"].Write(low)
