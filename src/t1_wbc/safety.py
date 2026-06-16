"""Hardware safety layer: per-joint clamps, torque slew limiting, weight-ramp,
watchdog, and infeasible->hold gating. Pure numpy; transport-agnostic."""
import numpy as np
from .config import servo_gains_for
from .transport import LowCmd


def clamp_torque(tau, lo, hi):
    return np.clip(np.asarray(tau, dtype=np.float64), lo, hi)


def slew_limit(tau, prev_tau, max_delta):
    tau = np.asarray(tau, dtype=np.float64); prev_tau = np.asarray(prev_tau, dtype=np.float64)
    return prev_tau + np.clip(tau - prev_tau, -max_delta, max_delta)


class SafetyLayer:
    """Wraps a raw WBC LowCmd into a safe one: servo gains + weight-ramp + clamps + slew,
    falling back to a PD hold (tau_ff=0) on infeasible solve or stale state."""
    def __init__(self, model, cfg, index_maps):
        self.cfg = cfg
        self.nu = model.nu
        self.servo_kp, self.servo_kd = servo_gains_for(index_maps)
        cr = np.asarray(model.actuator_ctrlrange, dtype=np.float64) * cfg.torque_limit_scale
        self.tau_lo = cr[:, 0]; self.tau_hi = cr[:, 1]
        self._prev_tau = np.zeros(self.nu)
        self._hold_q = np.zeros(self.nu)
        self._t0 = 0.0

    def begin(self, hold_q):
        self._hold_q = np.asarray(hold_q, dtype=np.float64).copy()
        self._t0 = 0.0
        self._prev_tau = np.zeros(self.nu)

    def wrap(self, raw, ok, t, lowstate_age):
        safe = (ok) and (lowstate_age <= self.cfg.watchdog_timeout_s)
        if not safe:                                   # hold: PD to hold pose, no feedforward
            q_des, qd_des, tau_ff = self._hold_q.copy(), np.zeros(self.nu), np.zeros(self.nu)
        else:
            alpha = min(1.0, (t - self._t0) / max(self.cfg.ramp_seconds, 1e-9))  # weight ramp
            q_des = (1 - alpha) * self._hold_q + alpha * raw.q_des
            qd_des = alpha * raw.qd_des
            tau_ff = alpha * raw.tau_ff
        tau_ff = clamp_torque(tau_ff, self.tau_lo, self.tau_hi)
        tau_ff = slew_limit(tau_ff, self._prev_tau, self.cfg.tau_slew_max)
        self._prev_tau = tau_ff.copy()
        return LowCmd(q_des=q_des, qd_des=qd_des,
                      kp=self.servo_kp.copy(), kd=self.servo_kd.copy(), tau_ff=tau_ff)
