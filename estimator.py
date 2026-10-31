from const import EstimatorConstant
import numpy as np


NUM_STATES = 9


def _initial_moments(c: EstimatorConstant) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of the initial state distribution."""
    xm = np.zeros(NUM_STATES)
    xm[2] = c.z0
    xm[4] = 0.5 * (c.theta0_min + c.theta0_max)
    xm[5] = 0.5 * (c.p0_min + c.p0_max)

    # Uniform disk of radius R: Var[x] = Var[y] = R^2 / 4
    # Uniform on [a, b]: Var = (b - a)^2 / 12
    Pm = np.diag([
        c.R ** 2 / 4.0,
        c.R ** 2 / 4.0,
        0.0,
        (2.0 * c.psi_bound) ** 2 / 12.0,
        (c.theta0_max - c.theta0_min) ** 2 / 12.0,
        (c.p0_max - c.p0_min) ** 2 / 12.0,
        0.0,
        0.0,
        0.0,
    ])
    return xm, Pm


def _theta_star(c: EstimatorConstant, k: int) -> float:
    """Target pitch of the dive schedule at time step k."""
    return c.theta_dive if k < c.N * c.k_switch_frac else -c.theta_dive


class EKF:
    """
    Extended Kalman Filter for the whale tracking problem under Gaussian noise.

    Args:
        estimator_constant : EstimatorConstant
            Constants known to the estimator (initial state bounds, process
            noise paramaters, etc...)
    """

    def __init__(
            self,
            estimator_constant: EstimatorConstant,
    ):
        self.constant = estimator_constant

        # Posterior mean of the state estimate
        self.xm = None
         # Posterior covariance of the state estimate
        self.Pm = None

        # Time index of the current posterior
        self.k = 0

        c = self.constant
        self.Q = np.diag([0.0] * 6 + [c.sigma_r_psi ** 2,
                                      c.sigma_r_theta ** 2,
                                      c.sigma_a ** 2])

        # Measurement matrices: USBL + sonar, and sonar only (USBL dropout)
        self.H_full = np.zeros((5, NUM_STATES))
        self.H_full[[0, 1, 2, 3, 4], [0, 1, 0, 1, 2]] = 1.0
        self.H_sonar = self.H_full[2:]

    def initialize(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Initialize the estimator. Set up any internal state required by the
        filter and return the initial posterior mean and covariance.

        Returns:
            xm : np.ndarray, dim: (num_states,)
                The initial posterior state mean. The order of states is
                x = [x, y, z, psi, theta, p, r_psi, r_theta, a].
            Pm : np.ndarray, dim: (num_states, num_states)
                The initial posterior state covariance.
        """
        xm, Pm = _initial_moments(self.constant)
        self.xm, self.Pm, self.k = xm, Pm, 0

        return xm.copy(), Pm.copy()

    def _predict(self) -> tuple[np.ndarray, np.ndarray]:
        """Prior update from k-1 to k (depth clipping ignored)."""
        c = self.constant
        Ts = c.Ts
        x, y, z, psi, th, p, r_psi, r_th, a = self.xm

        cps, sps = np.cos(psi), np.sin(psi)
        cth, sth = np.cos(th), np.sin(th)

        xp = np.array([
            x + p * cth * cps * Ts,
            y + p * cth * sps * Ts,
            z + p * sth * Ts,
            psi + r_psi * Ts,
            th + r_th * Ts,
            p + (a - c.cd * p) * Ts,
            r_psi + c.K_yaw * (c.omega_yaw - r_psi),
            r_th + c.K_theta_p * (_theta_star(c, self.k) - th)
                 - c.K_theta_d * r_th,
            a + c.K_p_p * (c.p_cruise - p) - c.K_p_d * a,
        ])

        A = np.eye(NUM_STATES)
        A[0, 3] = -p * cth * sps * Ts
        A[0, 4] = -p * sth * cps * Ts
        A[0, 5] = cth * cps * Ts
        A[1, 3] = p * cth * cps * Ts
        A[1, 4] = -p * sth * sps * Ts
        A[1, 5] = cth * sps * Ts
        A[2, 4] = p * cth * Ts
        A[2, 5] = sth * Ts
        A[3, 6] = Ts
        A[4, 7] = Ts
        A[5, 5] = 1.0 - c.cd * Ts
        A[5, 8] = Ts
        A[6, 6] = 1.0 - c.K_yaw
        A[7, 4] = -c.K_theta_p
        A[7, 7] = 1.0 - c.K_theta_d
        A[8, 5] = -c.K_p_p
        A[8, 8] = 1.0 - c.K_p_d

        Pp = A @ self.Pm @ A.T + self.Q
        return xp, Pp

    def estimate(
            self,
            measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Update the estimator with a new measurement and return the posterior
        mean and covariance at the current time step.

        Args:
            measurement : np.ndarray, dim: (num_measurements,)
                Sensor measurements from time step k, z(k). The order of
                measurements is z = [z_Ux, z_Uy, z_Sx, z_Sy, z_Sz], where
                z_Ux, z_Uy are the USBL measurements (np.nan during dropout)
                and z_Sx, z_Sy, z_Sz are the sonar measurements.

        Returns:
            xm : np.ndarray, dim: (num_states,)
                The posterior state mean at time step k. The order of
                states is x = [x, y, z, psi, theta, p, r_psi, r_theta, a].
            Pm : np.ndarray, dim: (num_states, num_states)
                The posterior state covariance at time step k.
        """
        c = self.constant
        xp, Pp = self._predict()
        self.k += 1

        sigma_S2 = c.sigma_S ** 2
        if np.isnan(measurement[0]) or np.isnan(measurement[1]):
            H = self.H_sonar
            zk = measurement[2:]
            R = np.diag([sigma_S2] * 3)
        else:
            H = self.H_full
            zk = measurement
            # State-dependent USBL noise, evaluated at the prior mean depth
            sigma_U2 = (c.sigma_base + c.alpha * xp[2]) ** 2
            R = np.diag([sigma_U2, sigma_U2, sigma_S2, sigma_S2, sigma_S2])

        PHt = Pp @ H.T
        S = H @ PHt + R
        K = np.linalg.solve(S, PHt.T).T
        xm = xp + K @ (zk - H @ xp)
        Pm = (np.eye(NUM_STATES) - K @ H) @ Pp
        Pm = 0.5 * (Pm + Pm.T)

        self.xm, self.Pm = xm, Pm
        return xm.copy(), Pm.copy()


class NonGaussianEstimator:
    """
    Estimator for the whale tracking problem under non-Gaussian noise.

    You are free to choose any recursive estimation strategy you like for this
    class. The public interface only exposes a single state estimate; the
    internal representation of your filter is up to you.

    Implementation: bootstrap particle filter. The particles are propagated
    through the full nonlinear dynamics (including depth clipping) with
    uniform process noise, and weighted with the exact non-Gaussian sensor
    likelihoods (uniform USBL noise, two-component Gaussian mixture sonar
    noise). Systematic resampling is performed when the effective sample size
    drops, followed by a small roughening step to avoid sample impoverishment.

    Args:
        estimator_constant : EstimatorConstant
            Constants known to the estimator.
    """

    NUM_PARTICLES = 10000
    RESAMPLE_FRAC = 0.5
    # Roughening: std = K * (range of particles) * N^(-1/d)
    ROUGHENING_K = 0.05
    # Floor on the uniform USBL likelihood (relative), keeps the filter alive
    # if all particles fall outside the uniform support
    USBL_FLOOR = 1e-3

    def __init__(
            self,
            estimator_constant: EstimatorConstant,
    ):
        self.constant = estimator_constant
        self.rng = np.random.default_rng()
        self.k = 0
        self.particles = None
        self.log_w = None

        c = self.constant
        beta = c.sonar_bimodality
        self.mix_w = np.array([c.sonar_mix_weight, 1.0 - c.sonar_mix_weight])
        self.mix_mu = np.array([beta * c.sigma_S, -9.0 * beta * c.sigma_S])
        self.mix_std = np.sqrt(1.0 - 9.0 * beta ** 2) * c.sigma_S
        self.noise_half = np.sqrt(3.0) * np.array(
            [c.sigma_r_psi, c.sigma_r_theta, c.sigma_a])

    def initialize(self) -> np.ndarray:
        """
        Initialize the estimator. Set up any internal state required by the
        filter and return the initial state estimate.

        Returns:
            xm : np.ndarray, dim: (num_states,)
                The initial state estimate. The order of states is
                x = [x, y, z, psi, theta, p, r_psi, r_theta, a].
        """
        c = self.constant
        n = self.NUM_PARTICLES
        rng = self.rng

        X = np.zeros((n, NUM_STATES))
        rad = c.R * np.sqrt(rng.random(n))
        ang = rng.uniform(-np.pi, np.pi, n)
        X[:, 0] = rad * np.cos(ang)
        X[:, 1] = rad * np.sin(ang)
        X[:, 2] = c.z0
        X[:, 3] = rng.uniform(-c.psi_bound, c.psi_bound, n)
        X[:, 4] = rng.uniform(c.theta0_min, c.theta0_max, n)
        X[:, 5] = rng.uniform(c.p0_min, c.p0_max, n)

        self.particles = X
        self.log_w = np.full(n, -np.log(n))
        self.k = 0

        xm, _ = _initial_moments(c)
        return xm

    def _propagate(self):
        c = self.constant
        Ts = c.Ts
        X = self.particles
        n = X.shape[0]
        x, y, z, psi, th, p, r_psi, r_th, a = X.T

        v = self.rng.uniform(-1.0, 1.0, (n, 3)) * self.noise_half
        pc = p * np.cos(th)

        Xn = np.empty_like(X)
        Xn[:, 0] = x + pc * np.cos(psi) * Ts
        Xn[:, 1] = y + pc * np.sin(psi) * Ts
        Xn[:, 2] = np.maximum(0.0, z + p * np.sin(th) * Ts)
        Xn[:, 3] = psi + r_psi * Ts
        Xn[:, 4] = th + r_th * Ts
        Xn[:, 5] = p + (a - c.cd * p) * Ts
        Xn[:, 6] = r_psi + c.K_yaw * (c.omega_yaw - r_psi) + v[:, 0]
        Xn[:, 7] = (r_th + c.K_theta_p * (_theta_star(c, self.k) - th)
                    - c.K_theta_d * r_th + v[:, 1])
        Xn[:, 8] = (a + c.K_p_p * (c.p_cruise - p) - c.K_p_d * a
                    + v[:, 2])
        self.particles = Xn

    def _sonar_log_lik(self, residual: np.ndarray) -> np.ndarray:
        """Log-likelihood (up to a constant) of sonar residuals, (n, 3)."""
        r = residual[..., None] - self.mix_mu
        comp = self.mix_w * np.exp(-0.5 * (r / self.mix_std) ** 2)
        return np.log(comp.sum(axis=-1) + 1e-300).sum(axis=1)

    def _usbl_log_lik(self, residual: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Log-likelihood of USBL residuals (n, 2) under uniform noise."""
        c = self.constant
        half = np.sqrt(3.0) * (c.sigma_base + c.alpha * z)
        inside = np.all(np.abs(residual) <= half[:, None], axis=1)
        dens = 1.0 / (2.0 * half) ** 2
        return np.log(np.where(inside, dens, self.USBL_FLOOR * dens))

    def _resample(self):
        n = self.particles.shape[0]
        w = np.exp(self.log_w)
        positions = (self.rng.random() + np.arange(n)) / n
        idx = np.searchsorted(np.cumsum(w), positions)
        np.minimum(idx, n - 1, out=idx)
        X = self.particles[idx]

        # Roughening
        spread = X.max(axis=0) - X.min(axis=0)
        sigma = self.ROUGHENING_K * spread * n ** (-1.0 / NUM_STATES)
        X = X + self.rng.standard_normal(X.shape) * sigma
        X[:, 2] = np.maximum(X[:, 2], 0.0)

        self.particles = X
        self.log_w = np.full(n, -np.log(n))

    def _point_estimate(self) -> np.ndarray:
        w = np.exp(self.log_w)
        X = self.particles
        xm = w @ X
        # Circular mean for the angles
        for i in (3, 4):
            xm[i] = np.arctan2(w @ np.sin(X[:, i]), w @ np.cos(X[:, i]))
        return xm

    def estimate(
            self,
            measurement: np.ndarray,
    ) -> np.ndarray:
        """
        Update the estimator with a new measurement and return the posterior
        state estimate at the current time step.

        Args:
            measurement : np.ndarray, dim: (num_measurements,)
                Sensor measurements from time step k, z(k). The order of
                measurements is z = [z_Ux, z_Uy, z_Sx, z_Sy, z_Sz], where
                z_Ux, z_Uy are the USBL measurements (np.nan during dropout)
                and z_Sx, z_Sy, z_Sz are the sonar measurements.

        Returns:
            xm : np.ndarray, dim: (num_states,)
                The posterior state estimate at time step k. The order of
                states is x = [x, y, z, psi, theta, p, r_psi, r_theta, a].
        """
        self._propagate()
        self.k += 1
        X = self.particles

        # Sonar measurement z_S = position + w_S
        log_lik = self._sonar_log_lik(measurement[2:] - X[:, :3])

        usbl_valid = not (np.isnan(measurement[0]) or np.isnan(measurement[1]))
        if usbl_valid:
            log_lik += self._usbl_log_lik(measurement[:2] - X[:, :2], X[:, 2])
            log_lik += np.where(X[:, 2] < self.constant.z_drop,
                                0.0, np.log(self.USBL_FLOOR))
        else:
            # Dropout tells us the whale is at or below z_drop
            log_lik += np.where(X[:, 2] >= self.constant.z_drop,
                                0.0, np.log(self.USBL_FLOOR))

        log_w = self.log_w + log_lik
        log_w -= log_w.max()
        log_w -= np.log(np.exp(log_w).sum())
        self.log_w = log_w

        n_eff = 1.0 / np.sum(np.exp(2.0 * log_w))
        xm = self._point_estimate()
        if n_eff < self.RESAMPLE_FRAC * X.shape[0]:
            self._resample()

        return xm
