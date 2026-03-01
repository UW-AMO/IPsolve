"""
Tests for the PLQ penalty library – verifies each penalty against CVXPY.

Each test:
1. Generates a random problem  min_x  ρ(z − H·x)
2. Solves with IPsolve
3. Solves with CVXPY
4. Asserts objectives match to within tolerance
"""

import numpy as np
import pytest
import scipy.sparse as sp

from ipsolve import solve, l1, l2, huber, hinge, vapnik, qreg, infnorm

# Try importing cvxpy; skip tests if unavailable
cvxpy = pytest.importorskip("cvxpy")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_problem(m=60, n=20, seed=42):
    """Random over-determined linear system with noise."""
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((m, n))
    x_true = rng.standard_normal(n)
    z = H @ x_true + 0.1 * rng.standard_normal(m)
    return H, z, x_true


def _rel_diff(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


# ---------------------------------------------------------------------------
# L2 (least squares)
# ---------------------------------------------------------------------------

class TestL2:
    def test_basic(self):
        H, z, _ = _make_problem()
        x_ip = solve(H, z, meas="l2", silent=True)

        # CVXPY
        x_cv = cvxpy.Variable(H.shape[1])
        prob = cvxpy.Problem(cvxpy.Minimize(0.5 * cvxpy.sum_squares(z - H @ x_cv)))
        prob.solve(solver=cvxpy.SCS, verbose=False)

        assert _rel_diff(prob.value, 0.5 * np.sum((z - H @ x_ip) ** 2)) < 1e-3


# ---------------------------------------------------------------------------
# Huber
# ---------------------------------------------------------------------------

class TestHuber:
    def test_basic(self):
        H, z, _ = _make_problem()
        kappa = 1.0
        x_ip = solve(H, z, meas=huber(kappa=kappa), silent=True)

        x_cv = cvxpy.Variable(H.shape[1])
        # CVXPY huber(x, M) = x² for |x|≤M, 2M|x|−M² for |x|>M
        # Our huber = (above) / 2,  so minimize 0.5*cvxpy.huber
        prob = cvxpy.Problem(cvxpy.Minimize(0.5 * cvxpy.sum(cvxpy.huber(z - H @ x_cv, M=kappa))))
        prob.solve(solver=cvxpy.SCS, verbose=False)

        obj_ip = np.sum(np.where(
            np.abs(z - H @ x_ip) > kappa,
            np.abs(z - H @ x_ip) - 0.5 * kappa,
            0.5 * (z - H @ x_ip) ** 2 / kappa,
        ))
        assert _rel_diff(prob.value, obj_ip) < 1e-2


# ---------------------------------------------------------------------------
# Lasso  (L2 + L1)
# ---------------------------------------------------------------------------

class TestLasso:
    def test_basic(self):
        m, n = 80, 40
        H, z, _ = _make_problem(m, n)
        lam = 0.5

        x_ip = solve(H, z, proc=l1(lam=lam), silent=True)

        x_cv = cvxpy.Variable(n)
        prob = cvxpy.Problem(cvxpy.Minimize(
            0.5 * cvxpy.sum_squares(z - H @ x_cv) + lam * cvxpy.norm1(x_cv)
        ))
        prob.solve(solver=cvxpy.SCS, verbose=False)

        obj_ip = 0.5 * np.sum((z - H @ x_ip) ** 2) + lam * np.sum(np.abs(x_ip))
        assert _rel_diff(prob.value, obj_ip) < 1e-2


# ---------------------------------------------------------------------------
# Huber + L1
# ---------------------------------------------------------------------------

class TestHuberL1:
    def test_basic(self):
        m, n = 80, 40
        H, z, _ = _make_problem(m, n)
        kappa = 1.0
        lam = 0.3

        x_ip = solve(H, z, meas=huber(kappa=kappa), proc=l1(lam=lam), silent=True)

        x_cv = cvxpy.Variable(n)
        prob = cvxpy.Problem(cvxpy.Minimize(
            0.5 * cvxpy.sum(cvxpy.huber(z - H @ x_cv, M=kappa))
            + lam * cvxpy.norm1(x_cv)
        ))
        prob.solve(solver=cvxpy.SCS, verbose=False)

        def huber_val(v, k):
            return np.sum(np.where(np.abs(v) > k, np.abs(v) - 0.5 * k, 0.5 * v ** 2 / k))

        obj_ip = huber_val(z - H @ x_ip, kappa) + lam * np.sum(np.abs(x_ip))
        assert _rel_diff(prob.value, obj_ip) < 5e-2


# ---------------------------------------------------------------------------
# Hinge loss (SVM)
# ---------------------------------------------------------------------------

class TestSVM:
    def test_binary_classification(self):
        rng = np.random.default_rng(123)
        n_inst, n_vars = 200, 20
        X = rng.standard_normal((n_inst, n_vars))
        w_true = rng.standard_normal(n_vars)
        y_labels = np.sign(X @ w_true)
        # Pre-multiply by labels
        yX = np.diag(y_labels) @ X
        lam_svm = 0.01

        # Hinge penalty: pos(1 - yX @ w)  with L2 regulariser (0.5/mMult)*||w||^2
        x_ip = solve(yX, np.ones(n_inst), meas=hinge(), proc=l2(lam=lam_svm), silent=True)

        x_cv = cvxpy.Variable(n_vars)
        prob = cvxpy.Problem(cvxpy.Minimize(
            cvxpy.sum(cvxpy.pos(1 - yX @ x_cv))
            + 0.5 * lam_svm * cvxpy.sum_squares(x_cv)
        ))
        prob.solve(solver=cvxpy.CLARABEL, verbose=False)

        obj_ip = np.sum(np.maximum(1 - yX @ x_ip, 0)) + 0.5 * lam_svm * np.sum(x_ip ** 2)
        obj_cv = prob.value
        assert _rel_diff(obj_cv, obj_ip) < 5e-2


# ---------------------------------------------------------------------------
# L1 fitting (LAD)
# ---------------------------------------------------------------------------

class TestL1Fitting:
    def test_basic(self):
        H, z, _ = _make_problem()
        x_ip = solve(H, z, meas=l1(), silent=True)

        x_cv = cvxpy.Variable(H.shape[1])
        prob = cvxpy.Problem(cvxpy.Minimize(cvxpy.norm1(z - H @ x_cv)))
        prob.solve(solver=cvxpy.SCS, verbose=False)

        obj_ip = np.sum(np.abs(z - H @ x_ip))
        assert _rel_diff(prob.value, obj_ip) < 1e-2


# ---------------------------------------------------------------------------
# Quantile regression
# ---------------------------------------------------------------------------

class TestQuantileRegression:
    def test_basic(self):
        H, z, _ = _make_problem()
        tau = 0.25

        x_ip = solve(H, z, meas=qreg(tau=tau), silent=True)

        x_cv = cvxpy.Variable(H.shape[1])
        resid = z - H @ x_cv
        # Standard quantile (check) loss: τ·pos(r) + (1−τ)·pos(−r)
        prob = cvxpy.Problem(cvxpy.Minimize(
            tau * cvxpy.sum(cvxpy.pos(resid)) + (1 - tau) * cvxpy.sum(cvxpy.pos(-resid))
        ))
        prob.solve(solver=cvxpy.CLARABEL, verbose=False)

        def qreg_val(v, t):
            return t * np.sum(np.maximum(v, 0)) + (1 - t) * np.sum(np.maximum(-v, 0))

        obj_ip = qreg_val(z - H @ x_ip, tau)
        assert _rel_diff(prob.value, obj_ip) < 1e-2


# ---------------------------------------------------------------------------
# L1 + L1
# ---------------------------------------------------------------------------

class TestL1L1:
    def test_basic(self):
        m, n = 80, 40
        H, z, _ = _make_problem(m, n)
        lam = 0.5

        x_ip = solve(H, z, meas=l1(), proc=l1(lam=lam), silent=True)

        x_cv = cvxpy.Variable(n)
        prob = cvxpy.Problem(cvxpy.Minimize(
            cvxpy.norm1(z - H @ x_cv) + lam * cvxpy.norm1(x_cv)
        ))
        prob.solve(solver=cvxpy.SCS, verbose=False)

        obj_ip = np.sum(np.abs(z - H @ x_ip)) + lam * np.sum(np.abs(x_ip))
        assert _rel_diff(prob.value, obj_ip) < 1e-2


# ---------------------------------------------------------------------------
# Vapnik ε-insensitive
# ---------------------------------------------------------------------------

class TestVapnik:
    def test_basic(self):
        H, z, _ = _make_problem()
        eps = 0.3
        lam = 1.0

        x_ip = solve(H, z, meas=vapnik(lam=lam, eps=eps), silent=True)

        x_cv = cvxpy.Variable(H.shape[1])
        resid = z - H @ x_cv
        prob = cvxpy.Problem(cvxpy.Minimize(
            lam * cvxpy.sum(cvxpy.pos(cvxpy.abs(resid) - eps))
        ))
        prob.solve(solver=cvxpy.SCS, verbose=False)

        obj_ip = lam * np.sum(np.maximum(np.abs(z - H @ x_ip) - eps, 0))
        assert _rel_diff(prob.value, obj_ip) < 1e-2


# ---------------------------------------------------------------------------
# With linear constraints
# ---------------------------------------------------------------------------

class TestConstrained:
    def test_l2_box_constraints(self):
        """LS with box constraints on x."""
        H, z, _ = _make_problem(m=60, n=20)
        box = 0.5
        n = H.shape[1]

        x_ip = solve(H, z, bounds=(-box, box), silent=True)

        x_cv = cvxpy.Variable(n)
        prob = cvxpy.Problem(
            cvxpy.Minimize(0.5 * cvxpy.sum_squares(z - H @ x_cv)),
            [x_cv >= -box, x_cv <= box]
        )
        prob.solve(solver=cvxpy.SCS, verbose=False)

        obj_ip = 0.5 * np.sum((z - H @ x_ip) ** 2)
        assert _rel_diff(prob.value, obj_ip) < 1e-2
        # Also check box bounds are respected
        assert np.all(x_ip <= box + 1e-4)
        assert np.all(x_ip >= -box - 1e-4)


# ---------------------------------------------------------------------------
# Infinity-norm regularisation (mirrors 515Examples/infNorm.m)
# ---------------------------------------------------------------------------

class TestInfNorm:
    def test_l2_infnorm(self):
        """min 0.5‖z − Hx‖² + λ‖x‖∞  vs CVXPY."""
        H, z, _ = _make_problem(m=60, n=20)
        lam = np.max(np.abs(H.T @ z)) / 5.0

        x_ip = solve(H, z, proc=infnorm(lam=lam),
                      opt_tol=1e-6, silent=True)

        n = H.shape[1]
        x_cv = cvxpy.Variable(n)
        prob = cvxpy.Problem(
            cvxpy.Minimize(0.5 * cvxpy.sum_squares(z - H @ x_cv)
                           + lam * cvxpy.norm(x_cv, "inf"))
        )
        prob.solve(solver=cvxpy.SCS, verbose=False)

        obj_ip = 0.5 * np.sum((z - H @ x_ip) ** 2) + lam * np.max(np.abs(x_ip))
        assert _rel_diff(prob.value, obj_ip) < 1e-2


# ---------------------------------------------------------------------------
# Kalman smoother (two-block with K_proc)
# ---------------------------------------------------------------------------

class TestKalman:
    def test_l2_l2_smoother(self):
        """Standard Kalman smoother (ℓ₂/ℓ₂) vs direct LS solution."""
        rng = np.random.default_rng(77)
        N = 20      # time steps
        n = 2       # state dim
        sigma = 0.1
        gamma = 0.5
        dt = 0.1

        # True signal
        x_true = np.sin(np.arange(N) * dt)

        # Noisy measurements
        z_raw = x_true + sigma * rng.standard_normal(N)

        # Measurement: pick out second state, scale by 1/sigma
        rinvk = 1.0 / sigma**2
        rh = np.sqrt(rinvk) * np.array([[0.0, 1.0]])
        H_mat = sp.kron(sp.eye(N, format="csc"), sp.csc_matrix(rh), format="csc")
        z_meas = np.sqrt(rinvk) * z_raw

        # Process: identity main diagonal, -I sub-diagonal (random walk)
        qk = gamma * np.array([[dt, dt**2/2], [dt**2/2, dt**3/3]])
        sqrtQinv = np.linalg.inv(np.linalg.cholesky(qk))
        gk = sqrtQinv @ np.array([[1.0, 0.0], [dt, 1.0]])

        # Build block tridiagonal (lower-bidiagonal)
        from examples.kalman_demo import blktridiag
        G_mat = blktridiag(sqrtQinv @ np.eye(n), -gk, N)
        G_mat[:n, :n] = 10.0 * sp.eye(n)
        w = np.zeros(n * N)
        w[:n] = 10.0 * x_true[:1]

        y_ip = solve(H_mat, z_meas, proc=l2(),
                     K_proc=G_mat, k_proc=w, silent=True)

        # CVXPY reference: min 0.5‖z_meas − H y‖² + 0.5‖G y − w‖²
        y_cv = cvxpy.Variable(n * N)
        prob = cvxpy.Problem(
            cvxpy.Minimize(
                0.5 * cvxpy.sum_squares(z_meas - H_mat @ y_cv)
                + 0.5 * cvxpy.sum_squares(G_mat @ y_cv - w)
            )
        )
        prob.solve(solver=cvxpy.SCS, verbose=False)

        obj_ip = (0.5 * np.sum((z_meas - H_mat @ y_ip)**2)
                  + 0.5 * np.sum((G_mat @ y_ip - w)**2))
        assert _rel_diff(prob.value, obj_ip) < 1e-2
