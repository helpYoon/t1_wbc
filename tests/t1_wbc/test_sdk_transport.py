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
class _OneShotState:
    """Simulates a real DDS msg whose buffer is valid ONLY during the callback: accessing
    motor_state_serial more than once raises (use-after-free), like the live binding did."""
    def __init__(self, n):
        self._ms = [_MotorState(0.1 * i, 0.0) for i in range(n)]
        self.imu_state = _Imu(); self._reads = 0
    @property
    def motor_state_serial(self):
        self._reads += 1
        if self._reads > 1:
            raise MemoryError("Could not allocate list object!")
        return self._ms
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
class _GetModeResponse:
    def __init__(self): self.mode = None
class _Loco:
    """mode = what GetMode reports; custom_rcs = rc sequence returned by ChangeMode(kCustom)
    (default all 0 = success). Records every requested mode in .modes; records when each
    ChangeMode happened relative to writes via a shared event log."""
    def __init__(self, mode="kPrepare", mode_seq=None, custom_rcs=None, events=None):
        self.modes = []; self._mode = mode
        self._mode_seq = list(mode_seq) if mode_seq is not None else None
        self._custom_rcs = list(custom_rcs) if custom_rcs is not None else None
        self._events = events
    def Init(self): pass
    def GetMode(self, resp):
        if self._mode_seq:                 # advance through a transition, then hold the last
            self._mode = self._mode_seq.pop(0)
        resp.mode = self._mode
        return 0
    def ChangeMode(self, m):
        self.modes.append(m)
        if self._events is not None: self._events.append(("mode", m))
        if m == "kCustom" and self._custom_rcs:
            return self._custom_rcs.pop(0)
        return 0

def _mocks(loco=None, events=None):
    pub = _Pub()
    if events is not None:
        _w = pub.Write
        pub.Write = lambda c: (events.append(("write", None)), _w(c))[1]
    return dict(low_cmd_pub=pub, loco=loco or _Loco(events=events), LowCmd=_LowCmd, MotorCmd=_MotorCmd,
                cmd_type_serial="SERIAL", kPrepare="kPrepare", kCustom="kCustom",
                kDamping="kDamping", GetModeResponse=_GetModeResponse)

_NOSLEEP = lambda *_a, **_k: None
def _seeded(tr, n=29):
    tr._on_state(_LowStateMsg(n)); return tr

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

def test_construction_uses_29_not_binding_constant():
    # The installed binding's B1JointCnt is 23; t1_wbc must NOT depend on it. Construction
    # works with an _sdk dict that carries no joint_cnt at all, and self.n is the model's 29.
    model, _ = load_t1_model(); maps = build_index_maps(model)
    sdk = _mocks(); sdk.pop("joint_cnt", None)
    tr = SdkTransport(model, maps, _sdk=sdk)
    from t1_wbc.joint_map import SDK_JOINT_CNT
    assert tr.n == SDK_JOINT_CNT == 29

def test_start_verifies_prepare_then_kcustom():
    # Robot must already be in kPrepare (operator-set); start() verifies via GetMode and
    # does NOT itself command kPrepare. Only kCustom is requested.
    model, _ = load_t1_model(); maps = build_index_maps(model)
    tr = _seeded(SdkTransport(model, maps, _sdk=_mocks()))    # _Loco default mode = kPrepare
    tr.start(sleep=_NOSLEEP, clock=lambda: 0.0)
    assert tr._sdk["loco"].modes == ["kCustom"]

def test_start_waits_for_prepare_transition_then_engages():
    # Operator hits Prepare and launches immediately; the firmware is still in kDamping for
    # a moment, then settles to kPrepare. start() should WAIT (grace) and then engage.
    model, _ = load_t1_model(); maps = build_index_maps(model)
    loco = _Loco(mode_seq=["kDamping", "kDamping", "kPrepare"])
    tr = _seeded(SdkTransport(model, maps, _sdk=_mocks(loco=loco)))
    clk = {"t": 0.0}
    def clock(): v = clk["t"]; clk["t"] += 0.1; return v
    tr.start(prepare_timeout=3.0, sleep=_NOSLEEP, clock=clock)
    assert "kCustom" in loco.modes                            # engaged after the transition

def test_start_refuses_if_never_reaches_prepare():
    import pytest
    model, _ = load_t1_model(); maps = build_index_maps(model)
    loco = _Loco(mode="kDamping")                             # stays in kDamping (not armed)
    tr = _seeded(SdkTransport(model, maps, _sdk=_mocks(loco=loco)))
    clk = {"t": 0.0}
    def clock(): v = clk["t"]; clk["t"] += 0.5; return v      # advance past the grace timeout
    with pytest.raises(RuntimeError):
        tr.start(prepare_timeout=1.0, sleep=_NOSLEEP, clock=clock)
    assert "kCustom" not in loco.modes                        # never handed control
    assert "kDamping" not in loco.modes                       # refused pre-engage, robot untouched

def test_start_publishes_initial_cmd_before_kcustom():
    # t1_controller: "firmware needs the LowCmd queued before accepting kCustom, else it reverts"
    model, _ = load_t1_model(); maps = build_index_maps(model)
    events = []
    tr = _seeded(SdkTransport(model, maps, _sdk=_mocks(events=events)))
    init = LowCmd(q_des=np.zeros(model.nu), qd_des=np.zeros(model.nu),
                  kp=np.zeros(model.nu), kd=np.zeros(model.nu), tau_ff=np.zeros(model.nu))
    tr.start(initial_cmd=init, sleep=_NOSLEEP, clock=lambda: 0.0)
    kinds = [e[0] for e in events]
    assert "write" in kinds and "mode" in kinds
    assert kinds.index("write") < kinds.index("mode")        # LowCmd queued before kCustom

def test_start_retries_kcustom_until_accepted():
    model, _ = load_t1_model(); maps = build_index_maps(model)
    loco = _Loco(custom_rcs=[501, 501, 0])                    # firmware refuses twice, then accepts
    tr = _seeded(SdkTransport(model, maps, _sdk=_mocks(loco=loco)))
    tr.start(sleep=_NOSLEEP, clock=lambda: 0.0)
    assert loco.modes.count("kCustom") == 3

def test_start_raises_without_damping_if_kcustom_refused():
    # Firmware refuses kCustom -> we NEVER took control -> robot stays in kPrepare.
    # We MUST NOT command kDamping (that would drop a firmware-held robot -> fall).
    import pytest
    model, _ = load_t1_model(); maps = build_index_maps(model)
    loco = _Loco(custom_rcs=[501] * 5)
    tr = _seeded(SdkTransport(model, maps, _sdk=_mocks(loco=loco)))
    with pytest.raises(RuntimeError):
        tr.start(mode_retries=5, sleep=_NOSLEEP, clock=lambda: 0.0)
    assert loco.modes.count("kCustom") == 5                  # tried 5x
    assert "kDamping" not in loco.modes                      # never engaged -> never drop the robot

def test_start_rejects_wrong_live_dof():
    import pytest
    model, _ = load_t1_model(); maps = build_index_maps(model)
    loco = _Loco()
    tr = _seeded(SdkTransport(model, maps, _sdk=_mocks(loco=loco)), n=23)  # 23-DOF: never reaches 29
    clk = {"t": 0.0}
    def clock(): clk["t"] += 0.5; return clk["t"]           # advance past timeout
    with pytest.raises(RuntimeError):
        tr.start(state_timeout=0.5, sleep=_NOSLEEP, clock=clock)
    assert "kCustom" not in loco.modes                       # never handed control
    assert "kDamping" not in loco.modes                      # not engaged -> stays in kPrepare

def test_stop_is_noop_if_never_engaged():
    # stop() before kCustom must NOT command ANY mode (robot is firmware-held in kPrepare).
    model, _ = load_t1_model(); maps = build_index_maps(model)
    loco = _Loco()
    tr = SdkTransport(model, maps, _sdk=_mocks(loco=loco))
    tr.stop()
    assert loco.modes == []

def test_stop_releases_to_prepare_after_engaging():
    # kDamping drops this robot, so release hands control back to the firmware (kPrepare),
    # which re-holds the pose and keeps the robot standing.
    model, _ = load_t1_model(); maps = build_index_maps(model)
    loco = _Loco()
    tr = _seeded(SdkTransport(model, maps, _sdk=_mocks(loco=loco)))
    tr.start(sleep=_NOSLEEP, clock=lambda: 0.0)              # engages kCustom
    tr.stop()
    assert loco.modes[-1] == "kPrepare"                      # had control -> release to kPrepare
    assert "kDamping" not in loco.modes

def test_read_lowstate_tolerates_partial_message():
    # The first DDS sample(s) can carry a short motor_state_serial. read_lowstate must NOT
    # IndexError; missing joints fall back to the last fully-populated reading.
    model, _ = load_t1_model(); maps = build_index_maps(model)
    tr = SdkTransport(model, maps, _sdk=_mocks())
    tr._on_state(_LowStateMsg(29)); tr._on_odom(_OdomMsg())
    full = tr.read_lowstate().joint_q.copy()                 # seeds last-good
    tr._on_state(_LowStateMsg(5))                            # partial sample arrives
    ls = tr.read_lowstate()                                  # must not raise
    assert ls.joint_q.shape == (model.nu,)
    np.testing.assert_allclose(ls.joint_q, full)             # missing joints reuse last-good

def test_read_lowstate_tolerates_missing_odom():
    # Odometer is a separate DDS topic that can lag the first LowState. read_lowstate must
    # NOT AttributeError on a None odom (that aborted the hold startup); default to zeros.
    model, _ = load_t1_model(); maps = build_index_maps(model)
    tr = SdkTransport(model, maps, _sdk=_mocks())
    tr._on_state(_LowStateMsg(29))                  # joint state present, NO odom yet
    ls = tr.read_lowstate()                         # must not raise
    np.testing.assert_allclose(ls.odom_xytheta, [0.0, 0.0, 0.0])


def test_on_state_parses_in_callback_no_use_after_free():
    # The msg buffer is only valid during the callback. _on_state must parse it THERE; later
    # read_lowstate calls must read the cache and never touch the (freed) msg again.
    model, _ = load_t1_model(); maps = build_index_maps(model); jm = JointMap(maps)
    tr = SdkTransport(model, maps, _sdk=_mocks())
    tr._on_state(_OneShotState(29)); tr._on_odom(_OdomMsg())
    ls1 = tr.read_lowstate()                 # must NOT re-access the msg (no MemoryError)
    ls2 = tr.read_lowstate()
    assert ls1.joint_q.shape == (model.nu,)
    assert abs(ls1.joint_q[jm.mj_index_of_sdk[5]] - 0.5) < 1e-9
    np.testing.assert_allclose(ls1.joint_q, ls2.joint_q)


def test_await_valid_state_waits_for_full_motor_array():
    model, _ = load_t1_model(); maps = build_index_maps(model)
    tr = SdkTransport(model, maps, _sdk=_mocks())
    tr._on_state(_LowStateMsg(5))                            # only a partial state present
    clk = {"t": 0.0}
    def clock(): clk["t"] += 0.2; return clk["t"]
    assert not tr._await_valid_state(0.5, _NOSLEEP, clock)   # never reaches 29
    tr._on_state(_LowStateMsg(29))
    assert tr._await_valid_state(0.5, _NOSLEEP, lambda: 0.0)

def test_start_raises_if_no_lowstate():
    import pytest
    model, _ = load_t1_model(); maps = build_index_maps(model)
    loco = _Loco()
    tr = SdkTransport(model, maps, _sdk=_mocks(loco=loco))    # no state ever seeded
    clk = {"t": 0.0}
    def clock(): clk["t"] += 1.0; return clk["t"]            # blow past the timeout immediately
    with pytest.raises(RuntimeError):
        tr.start(state_timeout=0.5, sleep=_NOSLEEP, clock=clock)
    assert "kCustom" not in loco.modes

def test_run_hw_loop_with_mock_transport_writes_commands():
    from t1_wbc.run import run_hw_loop
    from t1_wbc.config import WBCConfig
    model, data = load_t1_model(); maps = build_index_maps(model)
    tr = SdkTransport(model, maps, _sdk=_mocks())
    tr._on_state(_LowStateMsg(29)); tr._on_odom(_OdomMsg())   # seed a state for read_lowstate
    res = run_hw_loop(WBCConfig(), model, data, maps, tr, ticks=5,
                      clock=lambda: 0.0, sleep=lambda s: None)
    assert res["n"] == 5 and res["overruns"] >= 0
    assert len(tr._sdk["low_cmd_pub"].written) == 5
