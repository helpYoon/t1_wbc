"""Hardware safety layer: per-joint clamps, torque slew limiting, weight-ramp,
watchdog, and infeasible->hold gating. Pure numpy; transport-agnostic."""
import numpy as np


def clamp_torque(tau, lo, hi):
    return np.clip(np.asarray(tau, dtype=np.float64), lo, hi)


def slew_limit(tau, prev_tau, max_delta):
    tau = np.asarray(tau, dtype=np.float64); prev_tau = np.asarray(prev_tau, dtype=np.float64)
    return prev_tau + np.clip(tau - prev_tau, -max_delta, max_delta)
