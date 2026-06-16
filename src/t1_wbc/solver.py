"""WBC QP data + solver — proxsuite (dense, exact, warm-started). numpy-native."""
from dataclasses import dataclass
import numpy as np
from proxsuite import proxqp


@dataclass
class BatchedQP:
    """Single-robot QP data (numpy). Name kept for continuity; B=1, no batch dim.
    min 1/2 z'Hz + g'z  s.t.  A_eq z = b_eq,  G z <= b."""
    H: np.ndarray; g: np.ndarray          # (nz,nz),(nz,)
    A_eq: np.ndarray; b_eq: np.ndarray    # (neq,nz),(neq,)
    G: np.ndarray; b: np.ndarray          # (nineq,nz),(nineq,)


def external_residual(z, A_eq, b_eq, G, b):
    """(max |A_eq z - b_eq|, max(G z - b, 0)) — the trusted feasibility check."""
    eq = float(np.max(np.abs(A_eq @ z - b_eq))) if A_eq.shape[0] else 0.0
    viol = float(np.max(np.maximum(G @ z - b, 0.0))) if G.shape[0] else 0.0
    return eq, viol


class ProxsuiteSolver:
    """Persistent dense proxsuite QP, warm-started across ticks.
    Solves  min 1/2 z'Hz + g'z  s.t.  A_eq z = b_eq,  -inf <= G z <= b."""
    def __init__(self, feas_tol: float = 1e-4):
        self.feas_tol = feas_tol
        self._qp = None
        self._dims = None

    def solve(self, qp):
        H = np.ascontiguousarray(qp.H, dtype=np.float64)
        g = np.ascontiguousarray(qp.g, dtype=np.float64)
        A = np.ascontiguousarray(qp.A_eq, dtype=np.float64)
        beq = np.ascontiguousarray(qp.b_eq, dtype=np.float64)
        G = np.ascontiguousarray(qp.G, dtype=np.float64)
        b = np.ascontiguousarray(qp.b, dtype=np.float64)
        n = H.shape[0]; n_eq = A.shape[0]; n_in = G.shape[0]
        lo = np.full(n_in, -1e20)
        dims = (n, n_eq, n_in)
        if self._qp is None or self._dims != dims:
            self._qp = proxqp.dense.QP(n, n_eq, n_in)
            self._qp.settings.eps_abs = 1e-9  # tighten so |residual| < feas_tol on warm-start
            self._qp.init(H, g, A, beq, G, lo, b)
            self._dims = dims
        else:
            self._qp.update(H=H, g=g, A=A, b=beq, C=G, l=lo, u=b)
        self._qp.solve()
        z = np.asarray(self._qp.results.x, dtype=np.float64)
        eq, viol = external_residual(z, A, beq, G, b)
        ok = (eq < self.feas_tol) and (viol < self.feas_tol)
        return z, ok


def make_solver(cfg=None):
    return ProxsuiteSolver()
