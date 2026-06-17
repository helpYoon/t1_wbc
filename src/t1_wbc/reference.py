"""Standalone numpy port of the t1 motion-reference pipeline (NO ROS). Verified.
Ported: pkl_loader.py (load + joint-vel recompute), anchoring.py (yaw-only), state_packing.py."""
import pickle
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as R

from .config import PKL


class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def _load_raw(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        with open(path, "rb") as f:
            return _NumpyCompatUnpickler(f).load()


@dataclass(frozen=True)
class MotionSegment:
    arm: str; dt: float; T: int
    trunk_xyz: np.ndarray; trunk_quat_xyzw: np.ndarray
    left_arm: np.ndarray; right_arm: np.ndarray
    left_hand_pos: np.ndarray; left_hand_quat_xyzw: np.ndarray
    right_hand_pos: np.ndarray; right_hand_quat_xyzw: np.ndarray
    hip_pitch: np.ndarray; knee_pitch: np.ndarray; ankle_pitch: np.ndarray
    waist_pos: np.ndarray                       # = -pkl trunk_yaw
    trunk_xyz_dot: np.ndarray; trunk_omega_world: np.ndarray
    left_arm_vel: np.ndarray; right_arm_vel: np.ndarray
    hip_pitch_vel: np.ndarray; knee_pitch_vel: np.ndarray
    ankle_pitch_vel: np.ndarray; waist_vel: np.ndarray
    left_hand_vel: np.ndarray; right_hand_vel: np.ndarray
    left_hand_omega: np.ndarray; right_hand_omega: np.ndarray


def load_motion(path=PKL, segments=None):
    """Load motion segments from `path`. If `segments` is an iterable of indices,
    return only those segments, in the given order (validated against the count)."""
    raw = _load_raw(path)
    if not isinstance(raw, dict) or "segments" not in raw:
        raise ValueError(f"{path}: no 'segments' key")
    raw_segs = raw["segments"]
    if segments is not None:
        n = len(raw_segs)
        for i in segments:
            if not (0 <= i < n):
                raise IndexError(f"segment index {i} out of range [0,{n}) in {path}")
        raw_segs = [raw_segs[i] for i in segments]
    f64 = lambda a: np.ascontiguousarray(a, dtype=np.float64)
    segs = []
    for rs in raw_segs:
        pos = rs["position"]; vel = rs.get("velocity", {}) or {}
        T = int(rs["T"]); dt = float(rs["dt"])
        hip = f64(pos.get("trunk_pitch", np.zeros(T)))       # NO negation -> both Hip_Pitch
        knee = f64(pos.get("knee_pitch", np.zeros(T)))
        ankle = f64(pos.get("ankle_pitch", np.zeros(T)))
        waist = -f64(pos.get("trunk_yaw", np.zeros(T)))      # NEGATE -> Waist
        ddt = lambda a: (np.gradient(a, dt, axis=0) if a.shape[0] >= 2 else np.zeros_like(a))
        segs.append(MotionSegment(
            arm=str(rs["arm"]), dt=dt, T=T,
            trunk_xyz=f64(pos["trunk_xyz"]), trunk_quat_xyzw=f64(pos["trunk_quat_xyzw"]),
            left_arm=f64(pos["left_arm"]), right_arm=f64(pos["right_arm"]),
            left_hand_pos=f64(pos["left_hand_xyz"]),
            left_hand_quat_xyzw=f64(pos["left_hand_quat_xyzw"]),
            right_hand_pos=f64(pos["right_hand_xyz"]),
            right_hand_quat_xyzw=f64(pos["right_hand_quat_xyzw"]),
            hip_pitch=hip, knee_pitch=knee, ankle_pitch=ankle, waist_pos=waist,
            trunk_xyz_dot=f64(vel.get("trunk_xyz_dot", np.zeros((T, 3)))),
            trunk_omega_world=f64(vel.get("trunk_angular_velocity_world", np.zeros((T, 3)))),
            left_arm_vel=ddt(f64(pos["left_arm"])), right_arm_vel=ddt(f64(pos["right_arm"])),
            hip_pitch_vel=ddt(hip), knee_pitch_vel=ddt(knee),
            ankle_pitch_vel=ddt(ankle), waist_vel=ddt(waist),
            left_hand_vel=f64(vel.get("left_hand_xyz_dot", np.zeros((T, 3)))),
            right_hand_vel=f64(vel.get("right_hand_xyz_dot", np.zeros((T, 3)))),
            left_hand_omega=f64(vel.get("left_hand_angular_velocity_world", np.zeros((T, 3)))),
            right_hand_omega=f64(vel.get("right_hand_angular_velocity_world", np.zeros((T, 3)))),
        ))
    return segs


def anchor_pose(pkl_xyz, pkl_quat_xyzw, x0, y0, yaw0):
    """Yaw-only anchor: XY rotated by R_z(yaw0) then +(x0,y0); Z passes through; quat pre-composed."""
    Ry = R.from_euler("z", yaw0)
    xy = Ry.apply(np.array([pkl_xyz[0], pkl_xyz[1], 0.0]))[:2] + np.array([x0, y0])
    return (np.array([xy[0], xy[1], pkl_xyz[2]]),
            (Ry * R.from_quat(pkl_quat_xyzw)).as_quat())


def _anchor_stream_quat(arr, yaw0):
    return (R.from_euler("z", yaw0) * R.from_quat(arr)).as_quat()


def _anchor_stream_vec(arr, yaw0):
    """Rotate world-frame velocity/omega vectors by yaw only (translation-invariant)."""
    return R.from_euler("z", yaw0).apply(arr)


# arm joint order within left_arm/right_arm (matches MuJoCo arm order, verified)
_ARM = ["Shoulder_Pitch", "Shoulder_Roll", "Elbow_Pitch", "Elbow_Yaw",
        "Wrist_Pitch", "Wrist_Yaw", "Hand_Roll"]


def frame_to_qref(seg, k, index_maps, q_home):
    """Map pkl frame k -> (q_ref[nu], qd_ref[nu], tracked_mask[nu]) in MuJoCo actuator order.
    Tracked: 14 arm + Waist + both Hip_Pitch + both Knee_Pitch + both Ankle_Pitch (21).
    Untracked (head x2, hip roll/yaw x4, ankle roll x2) hold q_home with zero ref vel."""
    n2i = index_maps["name_to_act_index"]
    nu = len(q_home)
    q_ref = q_home.copy(); qd_ref = np.zeros(nu); tracked = np.zeros(nu, dtype=bool)

    def setj(name, val, vel):
        i = n2i[name]; q_ref[i] = val; qd_ref[i] = vel; tracked[i] = True

    for side, arm, armv in (("Left", seg.left_arm, seg.left_arm_vel),
                            ("Right", seg.right_arm, seg.right_arm_vel)):
        for j, suff in enumerate(_ARM):
            setj(f"{side}_{suff}", arm[k, j], armv[k, j])
    setj("Waist", seg.waist_pos[k], seg.waist_vel[k])
    for side in ("Left", "Right"):
        setj(f"{side}_Hip_Pitch", seg.hip_pitch[k], seg.hip_pitch_vel[k])
        setj(f"{side}_Knee_Pitch", seg.knee_pitch[k], seg.knee_pitch_vel[k])
        setj(f"{side}_Ankle_Pitch", seg.ankle_pitch[k], seg.ankle_pitch_vel[k])
    return q_ref, qd_ref, tracked


from dataclasses import dataclass as _dc
import mujoco


@_dc
class RefSample:
    q_ref: np.ndarray; qd_ref: np.ndarray; tracked: np.ndarray
    base_quat_xyzw: np.ndarray; base_omega_world: np.ndarray
    com_ref: np.ndarray
    left_hand_pos: np.ndarray; left_hand_quat_xyzw: np.ndarray; left_hand_vel: np.ndarray
    right_hand_pos: np.ndarray; right_hand_quat_xyzw: np.ndarray; right_hand_vel: np.ndarray


class ReferenceTrajectory:
    """Anchored, dedup'd, time-scaled reference with per-frame CoM (FK on a scratch MjData)."""

    def __init__(self, model, index_maps, q_home, cfg, x0=0.0, y0=0.0, yaw0=0.0):
        # q_home arrives as a numpy (nu,) array (WBController.q_home); frame_to_qref
        # needs a numpy (nu,) array it can .copy() and index by joint, so coerce here.
        q_home = np.asarray(q_home, dtype=np.float64).reshape(-1)
        self.model = model; self.maps = index_maps; self.q_home = q_home; self.cfg = cfg
        self._scratch = mujoco.MjData(model)
        segs = load_motion(cfg.motion, getattr(cfg, "segments", None))
        # global time axis from per-segment dt, with seam dedup applied to EVERY stream
        times, frames, segref, kref = [], [], [], []
        t = 0.0
        for si, s in enumerate(segs):
            for k in range(s.T):
                times.append(t); segref.append(si); kref.append(k); t += s.dt
        # build per-frame q_ref/qd_ref/tracked and anchored hand/base streams
        self._q = []; self._qd = []; self.tracked = None
        self._bq = []; self._bw = []; self._com = []
        self._lhp = []; self._lhq = []; self._lhv = []
        self._rhp = []; self._rhq = []; self._rhv = []
        for si, k in zip(segref, kref):
            s = segs[si]
            q_ref, qd_ref, tracked = frame_to_qref(s, k, index_maps, q_home)
            self.tracked = tracked
            bq = _anchor_stream_quat(s.trunk_quat_xyzw[k:k+1], yaw0)[0]
            bw = _anchor_stream_vec(s.trunk_omega_world[k:k+1], yaw0)[0]
            lhp, lhq = anchor_pose(s.left_hand_pos[k], s.left_hand_quat_xyzw[k], x0, y0, yaw0)
            rhp, rhq = anchor_pose(s.right_hand_pos[k], s.right_hand_quat_xyzw[k], x0, y0, yaw0)
            lhv = _anchor_stream_vec(s.left_hand_vel[k:k+1], yaw0)[0]
            rhv = _anchor_stream_vec(s.right_hand_vel[k:k+1], yaw0)[0]
            self._q.append(q_ref); self._qd.append(qd_ref)
            self._bq.append(bq); self._bw.append(bw); self._com.append(self._fk_com(q_ref, bq))
            self._lhp.append(lhp); self._lhq.append(lhq); self._lhv.append(lhv)
            self._rhp.append(rhp); self._rhq.append(rhq); self._rhv.append(rhv)
        # dedup seam frames using trunk position equality of the *raw* stream
        keep = self._dedup_mask([segs[si].trunk_xyz[k] for si, k in zip(segref, kref)])
        self._times = np.array(times)[keep]
        for name in ["_q", "_qd", "_bq", "_bw", "_com", "_lhp", "_lhq", "_lhv",
                     "_rhp", "_rhq", "_rhv"]:
            setattr(self, name, np.array(getattr(self, name))[keep])
        self.duration = float(self._times[-1] * cfg.time_scale)

    def _dedup_mask(self, trunks):
        keep = np.ones(len(trunks), dtype=bool)
        for i in range(1, len(trunks)):
            if np.allclose(trunks[i], trunks[i-1]):
                keep[i] = False
        return keep

    def _fk_com(self, q_ref, base_quat_xyzw):
        """World CoM of the reference config: home base xyz + reference base orientation
        (lean-consistent) + reference joints. Anchored with x0=y0=yaw0=0, so this world
        CoM aligns with the robot's home support."""
        d = self._scratch
        mujoco.mj_resetDataKeyframe(self.model, d, 0)          # base xyz = [0,0,0.6735]
        d.qpos[3:7] = [base_quat_xyzw[3], base_quat_xyzw[0],
                       base_quat_xyzw[1], base_quat_xyzw[2]]   # xyzw -> wxyz
        d.qpos[7:7+self.model.nu] = q_ref
        mujoco.mj_forward(self.model, d)
        return d.subtree_com[self.maps["base_body_id"]].copy()

    def sample(self, t_wall):
        """Nearest-frame sample at wall time (divided by time_scale, clamped). Returns RefSample."""
        t_ref = np.clip(t_wall / self.cfg.time_scale, 0.0, self._times[-1])
        i = int(np.searchsorted(self._times, t_ref))
        i = min(i, len(self._times) - 1)
        return RefSample(
            q_ref=self._q[i], qd_ref=self._qd[i] / self.cfg.time_scale, tracked=self.tracked,
            base_quat_xyzw=self._bq[i], base_omega_world=self._bw[i] / self.cfg.time_scale,
            com_ref=self._com[i],
            left_hand_pos=self._lhp[i], left_hand_quat_xyzw=self._lhq[i],
            left_hand_vel=self._lhv[i] / self.cfg.time_scale,
            right_hand_pos=self._rhp[i], right_hand_quat_xyzw=self._rhq[i],
            right_hand_vel=self._rhv[i] / self.cfg.time_scale,
        )
