"""WBC QP solver backends. proxsuite (B=1 exact)."""
from abc import ABC, abstractmethod
import numpy as np, torch

class SolverBackend(ABC):
    @abstractmethod
    def solve(self, qp):  # -> (z (B,nz), converged (B,) bool)
        ...

class ProxsuiteBackend(SolverBackend):
    """B=1 exact. Maps one-sided G z <= b into proxsuite (lb=-inf, ub=b)."""
    def solve(self, qp):
        from proxsuite import proxqp
        assert qp.H.shape[0] == 1, "ProxsuiteBackend is B=1 only"
        npd = lambda t: t[0].detach().cpu().numpy().astype(np.float64)
        H, g, A, beq, G, b = map(npd, (qp.H, qp.g, qp.A_eq, qp.b_eq, qp.G, qp.b))
        n, n_eq, n_in = H.shape[0], A.shape[0], G.shape[0]
        q = proxqp.dense.QP(n, n_eq, n_in)
        lo = np.full(n_in, -1e20); q.init(H, g, A, beq, G, lo, b)
        q.solve()
        z = torch.as_tensor(np.asarray(q.results.x), dtype=qp.H.dtype, device=qp.H.device).unsqueeze(0)
        ok = torch.tensor([str(q.results.info.status) == "QPSolverOutput.PROXQP_SOLVED"], device=qp.H.device)
        return z, ok

def make_solver(cfg):
    return ProxsuiteBackend()
