import numpy as np


class EstimatorConstant:
    def __init__(self):
        # Initial state
        self.R = 500.0                   # initial horizontal position radius (m), range: [200, 1000]
        self.z0 = 10.0                   # initial depth (m), fixed during evaluation
        self.psi_bound = np.pi           # initial heading: uniform in [-psi_bound, psi_bound], fixed during evaluation
        self.theta0_min = 0.1            # initial pitch lower bound (rad), fixed during evaluation
        self.theta0_max = np.pi / 6      # initial pitch upper bound (rad), fixed during evaluation
        self.p0_min = 0.0                # initial forward speed lower bound (m/s), fixed during evaluation
        self.p0_max = 3.0                # initial forward speed upper bound (m/s), fixed during evaluation

        # Process noise standard deviations
        self.sigma_r_psi = 0.005         # yaw rate noise std (rad/s), fixed during evaluation
        self.sigma_r_theta = 0.003       # pitch rate noise std (rad/s), fixed during evaluation
        self.sigma_a = 0.005             # forward acceleration noise std (m/s^2), fixed during evaluation

        # USBL sensor
        self.sigma_base = 30.0           # USBL baseline noise std (m), range: [5, 50]
        self.alpha = 0.02                # USBL depth scaling coefficient, range: [0.005, 0.05]
        self.z_drop = 2500.0             # USBL dropout depth (m), fixed during evaluation

        # Sonar sensor
        self.sigma_S = 100.0             # sonar noise std (m), range: [70, 250]
        self.sonar_bimodality = 0.3      # beta, fixed during evaluation
        self.sonar_mix_weight = 0.9      # weight of normal distribution closest to 0, fixed during evaluation

        # Drag coefficient
        self.cd = 0.0001                 # linear drag coefficient (1/s), fixed during evaluation

        # Behavioral model coefficients
        self.K_yaw = 0.01                # fixed during evaluation
        self.omega_yaw = 0.005           # yaw oscillation frequency (rad/s), range: [-0.015, 0.015]
        self.K_theta_p = 0.005           # fixed during evaluation
        self.K_theta_d = 0.30            # fixed during evaluation
        self.theta_dive = 0.70           # fixed during evaluation
        self.K_p_p = 0.005               # fixed during evaluation
        self.K_p_d = 0.15                # fixed during evaluation
        self.p_cruise = 1.5              # fixed during evaluation
        self.k_switch_frac = 0.6         # timestep fraction after which theta_star flips sign, fixed during evaluation

        # Time
        self.N = 500                     # fixed during evaluation
        self.Ts = 10.0                   # time step (s), fixed during evaluation
