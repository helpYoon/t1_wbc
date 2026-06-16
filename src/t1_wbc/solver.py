"""WBC QP solver backends. ADMM (default, batched) + proxsuite (B=1 exact)."""
from abc import ABC, abstractmethod
import numpy as np, torch
from themis_mpc.admm_qp import ADMMSolver, QPData

def _per_env_residual(z, A_eq, b_eq, G, b):
    eq = (torch.bmm(A_eq, z.unsqueeze(-1)).squeeze(-1) - b_eq).abs().amax(dim=1)
    viol = torch.clamp(torch.bmm(G, z.unsqueeze(-1)).squeeze(-1) - b, min=0.0).amax(dim=1)
    return eq, viol

class SolverBackend(ABC):
    @abstractmethod
    def solve(self, qp):  # -> (z (B,nz), converged (B,) bool)
        ...

class AdmmBackend(SolverBackend):
    def __init__(self, cfg):
        self.cfg = cfg
        self.solver = ADMMSolver(max_iter=cfg.admm_max_iter, rho=cfg.admm_rho,
                                 rho_eq=cfg.admm_rho_eq, eps_abs=cfg.admm_eps_abs, eps_rel=cfg.admm_eps_rel)
    def solve(self, qp):
        B, n = qp.g.shape; dtype = qp.H.dtype; dev = qp.H.device
        neg = torch.full((B, n), -float("inf"), dtype=dtype, device=dev)
        pos = torch.full((B, n), float("inf"), dtype=dtype, device=dev)
        qd = QPData(H=qp.H, h=qp.g, G=qp.G, b=qp.b, lb=neg, ub=pos, A_eq=qp.A_eq, b_eq=qp.b_eq)
        sol = self.solver.solve(qd)
        eq, viol = _per_env_residual(sol.z, qp.A_eq, qp.b_eq, qp.G, qp.b)   # TRUST THIS, not sol.converged
        ok = (eq < self.cfg.admm_feas_tol) & (viol < self.cfg.admm_feas_tol)
        return sol.z, ok

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
    return ProxsuiteBackend() if cfg.backend == "proxsuite" else AdmmBackend(cfg)
