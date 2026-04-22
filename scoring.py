import numpy as np

# Timing budgets (per estimate() call)
EKF_BUDGET_PER_STEP = 0.005      # 5 ms
NGE_BUDGET_PER_STEP = 0.050      # 50 ms
LAMBDA  = 7.0                    # sigmoid transition sharpness


def angular_rmse_per_state(estimates, states):
    """
    RMSE per state, with circular wrap on psi and theta so a +pi vs -pi
    mismatch doesn't get punished as if it were a 2*pi error.
    """
    err = estimates - states
    err[3] = (err[3] + np.pi) % (2 * np.pi) - np.pi   # psi
    err[4] = (err[4] + np.pi) % (2 * np.pi) - np.pi   # theta
    return np.sqrt(np.mean(err ** 2, axis=1))


def timing_score(per_step_time, budget):
    return float(1.0 / (1.0 + np.exp(LAMBDA * (per_step_time - budget) / budget)))