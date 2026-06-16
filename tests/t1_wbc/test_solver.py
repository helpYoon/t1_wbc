import numpy as np
from t1_wbc.solver import ProxsuiteSolver, BatchedQP, external_residual

def test_box_qp_against_closed_form():
    # min 1/2 x'x + g'x  s.t.  x <= b  ->  x = min(-g, b)  (unconstrained min is -g)
    n = 4
    qp = BatchedQP(H=np.eye(n), g=-np.array([1., 2., 3., 4.]),
                   A_eq=np.zeros((0, n)), b_eq=np.zeros(0),
                   G=np.eye(n), b=np.array([10., 1.5, 10., 10.]))
    z, ok = ProxsuiteSolver().solve(qp)
    assert ok
    np.testing.assert_allclose(z, [1., 1.5, 3., 4.], atol=1e-6)

def test_equality_constrained():
    # min 1/2 x'x  s.t.  x0 + x1 = 2  ->  x = [1, 1]
    n = 2
    qp = BatchedQP(H=np.eye(n), g=np.zeros(n),
                   A_eq=np.array([[1., 1.]]), b_eq=np.array([2.]),
                   G=np.zeros((0, n)), b=np.zeros(0))
    z, ok = ProxsuiteSolver().solve(qp)
    assert ok
    np.testing.assert_allclose(z, [1., 1.], atol=1e-6)

def test_warm_start_same_dims_resolves():
    n = 3
    s = ProxsuiteSolver()
    qp = BatchedQP(H=np.eye(n), g=-np.ones(n), A_eq=np.zeros((0, n)), b_eq=np.zeros(0),
                   G=np.eye(n), b=np.full(n, 0.5))
    z1, _ = s.solve(qp)
    qp.b = np.full(n, 0.25)            # change bounds, same dims -> uses update() path
    z2, ok = s.solve(qp)
    assert ok
    np.testing.assert_allclose(z2, [0.25, 0.25, 0.25], atol=1e-6)

def test_external_residual_zero_on_feasible():
    z = np.array([1., 1.])
    eq, viol = external_residual(z, np.array([[1., 1.]]), np.array([2.]),
                                 np.eye(2), np.array([5., 5.]))
    assert eq < 1e-9 and viol == 0.0
