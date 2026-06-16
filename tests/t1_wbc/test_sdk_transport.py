import numpy as np
from t1_wbc.model import load_t1_model, build_index_maps
from t1_wbc.joint_map import JointMap
from t1_wbc.sdk_transport import SdkTransport
from t1_wbc.transport import LowState, LowCmd

class _MotorState:
    def __init__(self, q, dq): self.q = q; self.dq = dq; self.tau_est = 0.0
class _Imu:
    def __init__(self): self.rpy = [0.0, 0.05, 0.2]; self.gyro = [0.0, 0.0, 0.1]; self.acc = [0, 0, 9.81]
class _LowStateMsg:
    def __init__(self, n): self.imu_state = _Imu(); self.motor_state_serial = [_MotorState(0.1*i, 0.0) for i in range(n)]
class _OdomMsg:
    x, y, theta = 1.0, 2.0, 0.2
class _MotorCmd:
    def __init__(self): self.q=self.dq=self.tau=self.kp=self.kd=self.weight=0.0
class _LowCmd:
    def __init__(self): self.cmd_type=None; self.motor_cmd=[]
class _Pub:
    def __init__(self): self.written=[]
    def InitChannel(self): pass
    def Write(self, c): self.written.append(c)
class _Loco:
    def __init__(self): self.modes=[]
    def Init(self): pass
    def ChangeMode(self, m): self.modes.append(m)

def _mocks():
    return dict(low_cmd_pub=_Pub(), loco=_Loco(), LowCmd=_LowCmd, MotorCmd=_MotorCmd,
                cmd_type_serial="SERIAL", kPrepare="kPrepare", kCustom="kCustom", joint_cnt=29)

def test_read_lowstate_remaps_sdk_to_mujoco():
    model, _ = load_t1_model(); maps = build_index_maps(model); jm = JointMap(maps)
    tr = SdkTransport(model, maps, _sdk=_mocks())
    tr._on_state(_LowStateMsg(29)); tr._on_odom(_OdomMsg())
    ls = tr.read_lowstate()
    assert isinstance(ls, LowState) and ls.joint_q.shape == (model.nu,)
    assert abs(ls.joint_q[jm.mj_index_of_sdk[5]] - 0.5) < 1e-9   # SDK idx 5 carried 0.1*5=0.5
    np.testing.assert_allclose(ls.imu_rpy, [0.0, 0.05, 0.2])
    np.testing.assert_allclose(ls.odom_xytheta, [1.0, 2.0, 0.2])

def test_write_lowcmd_fills_serial_motorcmd_by_sdk_index():
    model, _ = load_t1_model(); maps = build_index_maps(model); jm = JointMap(maps)
    tr = SdkTransport(model, maps, _sdk=_mocks())
    cmd = LowCmd(q_des=np.arange(model.nu, dtype=float), qd_des=np.zeros(model.nu),
                 kp=np.full(model.nu, 7.0), kd=np.full(model.nu, 0.7), tau_ff=np.full(model.nu, 1.0))
    tr.write_lowcmd(cmd)
    written = tr._sdk["low_cmd_pub"].written[-1]
    assert written.cmd_type == "SERIAL" and len(written.motor_cmd) == 29
    mj5 = jm.mj_index_of_sdk[5]
    assert abs(written.motor_cmd[5].q - cmd.q_des[mj5]) < 1e-9
    assert written.motor_cmd[5].kp == 7.0 and written.motor_cmd[5].tau == 1.0

def test_start_does_prepare_then_custom():
    model, _ = load_t1_model(); maps = build_index_maps(model)
    tr = SdkTransport(model, maps, _sdk=_mocks())
    tr.start()
    assert tr._sdk["loco"].modes == ["kPrepare", "kCustom"]

def test_wrong_joint_count_fails_fast():
    import pytest
    model, _ = load_t1_model(); maps = build_index_maps(model)
    bad = _mocks(); bad["joint_cnt"] = 23
    with pytest.raises(AssertionError):
        SdkTransport(model, maps, _sdk=bad)

def test_run_hw_loop_with_mock_transport_writes_commands():
    from t1_wbc.run import run_hw_loop
    from t1_wbc.config import WBCConfig
    model, data = load_t1_model(); maps = build_index_maps(model)
    tr = SdkTransport(model, maps, _sdk=_mocks())
    tr._on_state(_LowStateMsg(29)); tr._on_odom(_OdomMsg())   # seed a state for read_lowstate
    n_written = run_hw_loop(WBCConfig(), model, data, maps, tr, ticks=5)
    assert n_written == 5
    assert len(tr._sdk["low_cmd_pub"].written) == 5
