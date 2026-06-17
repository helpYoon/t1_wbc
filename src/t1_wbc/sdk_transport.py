"""Booster-SDK transport: LowState/Odometer subscribe -> LowState; LowCmd publish (SERIAL,
kCustom). The real SDK is imported lazily so the package/tests run without it; tests inject
a mock `_sdk` dict. Joints are remapped MuJoCo<->SDK index 0-28 via JointMap.

Two safety invariants, mirroring t1_controller's BoosterT1HwInterface:
  * we only command a release (kPrepare) if we actually ENGAGED kCustom (took control); a
    failure before handover leaves the robot in its firmware-held mode instead of dropping it.
    Release is kPrepare (firmware re-holds, robot stays standing), NOT kDamping (which drops it);
  * the joint count + IMU/odom are PARSED in the callback (the msg buffer is valid only there);
    read_lowstate reads numpy snapshots, never the raw msg (no use-after-free)."""
import time
import numpy as np
from .transport import Transport, LowState, LowCmd
from .joint_map import JointMap, SDK_JOINT_CNT

# NOTE: t1_controller needed a z-axis tau_ff sign flip (flipWaistTauFf: Waist/Hip_Yaw) for its
# Pinocchio inverse dynamics. t1_wbc computes tau_ff via MuJoCo, which is correctly signed for
# the real robot — confirmed ON HARDWARE (2026-06-17) that flipping makes it WORSE. So we do
# NOT flip tau_ff here.


def _real_sdk():
    import booster_robotics_sdk_python as B
    B.ChannelFactory.Instance().Init(0)
    loco = B.B1LocoClient(); loco.Init()
    pub = B.B1LowCmdPublisher(); pub.InitChannel()
    # NOTE: B.B1JointCnt is a compile-time constant of the installed binding (23 for the
    # 5-DOF-arm build) and is NOT a reading of the connected robot, so we deliberately do
    # not use it. The real joint count is validated against the live LowState in start().
    return dict(low_cmd_pub=pub, loco=loco, LowCmd=B.LowCmd, MotorCmd=B.MotorCmd,
                cmd_type_serial=B.LowCmdType.SERIAL, kPrepare=B.RobotMode.kPrepare,
                kCustom=B.RobotMode.kCustom, kDamping=B.RobotMode.kDamping,
                GetModeResponse=B.GetModeResponse, _module=B)


class SdkTransport(Transport):
    def __init__(self, model, index_maps, _sdk=None):
        self.model = model
        self.jm = JointMap(index_maps)
        self._sdk = _sdk if _sdk is not None else _real_sdk()
        # t1_wbc drives the 29-joint / 7-DOF-arm layout. We use that count directly (NOT the
        # binding's B1JointCnt, which is a stale compile-time 23) and validate it against the
        # robot's first published LowState in start() before any command is sent.
        self.n = SDK_JOINT_CNT
        self._state_t = 0.0
        self._engaged = False                  # True only after a successful kCustom handover
        # State is PARSED inside the DDS callback (the msg buffer is valid only there — storing
        # the raw msg and reading it later is use-after-free). We keep plain numpy snapshots:
        self._have_full = False                # parsed a full (>=29-joint) reading at least once
        self._q_sdk = None                     # last full joint positions, SDK order (sticky)
        self._dq_sdk = None
        self._imu_rpy = np.zeros(3); self._imu_gyro = np.zeros(3); self._imu_acc = np.zeros(3)
        self._odom_xyt = None                  # last odom (separate DDS topic; may lag)
        if _sdk is None:                       # real run: subscribe (callbacks)
            B = self._sdk["_module"]
            self._state_sub = B.B1LowStateSubscriber(self._on_state); self._state_sub.InitChannel()
            self._odom_sub = B.B1OdometerStateSubscriber(self._on_odom); self._odom_sub.InitChannel()

    def _on_state(self, msg):
        # Parse HERE — the msg buffer is only valid during the callback. Reading the stored
        # msg later raised "Could not allocate list object!" (use-after-free) and dropped the
        # robot. Mirrors t1_controller's lowStateCallback.
        ms = msg.motor_state_serial
        if len(ms) >= self.n:                                  # ignore short/partial samples
            self._q_sdk = np.array([ms[i].q for i in range(self.n)], dtype=np.float64)
            self._dq_sdk = np.array([ms[i].dq for i in range(self.n)], dtype=np.float64)
            self._have_full = True
        imu = msg.imu_state
        self._imu_rpy = np.array(imu.rpy, dtype=np.float64)
        self._imu_gyro = np.array(imu.gyro, dtype=np.float64)
        self._imu_acc = np.array(imu.acc, dtype=np.float64)
        self._state_t = time.monotonic()

    def _on_odom(self, msg):
        self._odom_xyt = np.array([msg.x, msg.y, msg.theta], dtype=np.float64)

    def _await_valid_state(self, timeout, sleep, clock):
        """Wait up to `timeout` for a fully-populated (>=29-joint) LowState to have been parsed
        in the callback. The first DDS sample(s) can be short/empty, so 'a state arrived' is
        not sufficient. Returns True once a full reading exists, else False on timeout."""
        deadline = clock() + timeout
        while clock() < deadline:
            if self._have_full:
                return True
            sleep(0.01)
        return False

    def _await_prepare(self, timeout, sleep, clock):
        """Poll GetMode until it reports kPrepare, up to `timeout`. Returns (ok, last_mode).
        Tolerates the brief kDamping->kPrepare transition right after the operator arms the
        remote, and a GetMode RPC that isn't ready yet (rc != 0)."""
        loco = self._sdk["loco"]; deadline = clock() + timeout; last = None
        while clock() < deadline:
            resp = self._sdk["GetModeResponse"]()
            rc = loco.GetMode(resp)
            if rc == 0:
                last = resp.mode
                if resp.mode == self._sdk["kPrepare"]:
                    return True, last
            sleep(0.1)
        return False, last

    def start(self, initial_cmd=None, require_prepare=True, mode_retries=5,
              state_timeout=2.0, prepare_timeout=3.0, sleep=time.sleep, clock=time.monotonic):
        """Validate the robot, queue an initial LowCmd, then hand over with kCustom.

        Mirrors the proven booster_t1 hardware bring-up sequence:
          1. wait for a LowState reporting all 29 joints (7-DOF arm); short first samples wait;
          2. wait (up to prepare_timeout) for GetMode to report kPrepare — we do NOT command
             kPrepare; this tolerates the operator arming Prepare just before launch;
          3. publish `initial_cmd` (if given) so the firmware has a LowCmd stream queued
             BEFORE kCustom (else the firmware reverts to kPrepare);
          4. ChangeMode(kCustom), retrying while rc != 0 (firmware refuses ~code 501 after a fall);
          5. only on success set _engaged=True — the flag that lets stop() release via kPrepare.

        Every failure path raises WITHOUT commanding a mode: we have NOT taken control, so
        the robot must stay in its firmware-held mode (kPrepare) — damping it here would drop
        it. Callers wrap start()+loop in try/finally: stop(); stop() damps only if engaged.
        `sleep`/`clock` are injectable for deterministic tests."""
        loco = self._sdk["loco"]
        if not self._await_valid_state(state_timeout, sleep, clock):
            raise RuntimeError(
                f"no LowState with {SDK_JOINT_CNT} joints within {state_timeout}s; "
                "robot absent or not a 7-DOF-arm unit — refusing kCustom")
        if require_prepare:
            ok, observed = self._await_prepare(prepare_timeout, sleep, clock)
            if not ok:
                raise RuntimeError(
                    f"robot not in kPrepare within {prepare_timeout}s (last GetMode: {observed}); "
                    "set Prepare on the remote and retry")
        if initial_cmd is not None:                  # queue a LowCmd before kCustom
            self.write_lowcmd(initial_cmd); sleep(0.05)
        rc = -1
        for _ in range(max(1, mode_retries)):
            rc = loco.ChangeMode(self._sdk["kCustom"])
            if rc == 0:
                break
            sleep(0.5)
        if rc != 0:
            # Firmware refused — we never took control, robot stays in kPrepare. Do NOT damp.
            raise RuntimeError(
                f"ChangeMode(kCustom) refused after {mode_retries} attempts (rc={rc}); robot left "
                "in kPrepare (not damped). Power-cycle or do a damping/prepare cycle and retry")
        self._engaged = True                         # now under our control -> stop() may damp
        # NOTE: do NOT sleep here. The caller must begin streaming LowCmd immediately — the
        # firmware expects a continuous stream once in kCustom (a gap can trip its watchdog).

    def stop(self):
        """Release control back to the firmware via kPrepare (it re-holds the pose, so the
        robot stays STANDING), and only if we actually engaged kCustom — never command a mode
        on a robot we never controlled. We do NOT use kDamping on release: it drops this robot
        (the hardware E-stop is the path for an emergency limp)."""
        if self._engaged and "kPrepare" in self._sdk:
            self._sdk["loco"].ChangeMode(self._sdk["kPrepare"])
        self._engaged = False

    def state_age(self):
        return time.monotonic() - self._state_t

    def read_lowstate(self) -> LowState:
        # Reads only the numpy snapshots parsed in the callbacks — never the raw DDS msg, so
        # there is no use-after-free. Local refs make the read atomic vs a concurrent callback.
        q_sdk = self._q_sdk if self._q_sdk is not None else np.zeros(self.n)
        dq_sdk = self._dq_sdk if self._dq_sdk is not None else np.zeros(self.n)
        odom = self._odom_xyt if self._odom_xyt is not None else np.zeros(3)
        return LowState(imu_rpy=self._imu_rpy.copy(),
                        imu_gyro=self._imu_gyro.copy(),
                        imu_acc=self._imu_acc.copy(),
                        joint_q=self.jm.sdk_to_mujoco(q_sdk),
                        joint_dq=self.jm.sdk_to_mujoco(dq_sdk),
                        odom_xytheta=np.asarray(odom, dtype=np.float64).copy())

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
