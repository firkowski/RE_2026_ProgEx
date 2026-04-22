import sys
import time
import numpy as np

from const import EstimatorConstant
from simulator import Simulator
from estimator import EKF, NonGaussianEstimator

HEADLESS = '--headless' in sys.argv

# Simulate
est_const  = EstimatorConstant()
N          = est_const.N
num_states = 9

print("Starting simulation...")
sim_g = Simulator(noise="Gaussian") # Can change measurement noise scenario
sim_g.update_const(est_const)
sim_g.simulate()

sim_ng = Simulator(noise="Non-Gaussian") # Can change measurement noise scenario
sim_ng.update_const(est_const)
sim_ng.share_states_from(sim_g)


# Run estimators
def run_ekf(filter_obj, sim):
    """Returns (estimates_or_None, covariances_or_None, per_step_seconds_or_nan)."""
    init = filter_obj.initialize()
    if init is None or init[0] is None:
        return None, None, float('nan')
    xm0, Pm0 = init
    estimates = np.zeros((num_states, N))
    covariances = np.zeros((num_states, num_states, N))
    estimates[:, 0] = xm0
    covariances[:, :, 0] = Pm0
    t0 = time.perf_counter()
    for k in range(1, N):
        xm, Pm = filter_obj.estimate(sim.measurements[:, k].copy())
        estimates[:, k] = xm
        covariances[:, :, k] = Pm
    dt = (time.perf_counter() - t0) / (N - 1)
    return estimates, covariances, dt


def run_nge(filter_obj, sim):
    """Returns (estimates_or_None, per_step_seconds_or_nan)."""
    init = filter_obj.initialize()
    if init is None:
        return None, float('nan')
    estimates = np.zeros((num_states, N))
    estimates[:, 0] = init
    t0 = time.perf_counter()
    for k in range(1, N):
        estimates[:, k] = filter_obj.estimate(sim.measurements[:, k].copy())
    dt = (time.perf_counter() - t0) / (N - 1)
    return estimates, dt


print("Running EKF...")
estimates_ekf, covariances_ekf, ekf_dt = run_ekf(EKF(est_const), sim_g)

print("Running NonGaussianEstimator...")
estimates_nge, nge_dt = run_nge(NonGaussianEstimator(est_const), sim_ng)

ekf_rmse, ekf_timing = sim_g .score("EKF (Gaussian)",                      "ekf", estimates_ekf, ekf_dt)
nge_rmse, nge_timing = sim_ng.score("NonGaussianEstimator (Non-Gaussian)", "nge", estimates_nge, nge_dt)


print("\nSaving results.npz...")
np.savez('results.npz',
    estimates_ekf   = estimates_ekf if estimates_ekf is not None else np.zeros((0,)),
    estimates_nge   = estimates_nge if estimates_nge is not None else np.zeros((0,)),
    ekf_implemented = np.array(estimates_ekf is not None),
    nge_implemented = np.array(estimates_nge is not None),
    measurements_g  = sim_g.measurements,
    measurements_ng = sim_ng.measurements,
    N               = np.array(N),
    ekf_rmse        = ekf_rmse  if ekf_rmse  is not None else np.zeros((0,)),
    ekf_timing      = np.array(ekf_timing if not (ekf_timing  != ekf_timing)  else 0.0),
    ekf_dt          = np.array(ekf_dt     if not (ekf_dt      != ekf_dt)      else 0.0),
    nge_rmse        = nge_rmse  if nge_rmse  is not None else np.zeros((0,)),
    nge_timing      = np.array(nge_timing if not (nge_timing  != nge_timing)  else 0.0),
    nge_dt          = np.array(nge_dt     if not (nge_dt      != nge_dt)      else 0.0),
)
print("Saved results.npz")


if HEADLESS:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plotting import make_3d_figure, make_filter_figure

    make_3d_figure(estimates_ekf, estimates_nge).savefig(
        "plot_3d.png", dpi=350, bbox_inches='tight')
    make_filter_figure(sim_g,  estimates_ekf, "EKF",
                       "Gaussian",     est_color='crimson'
                       ).savefig("plot_ekf.png", dpi=350, bbox_inches='tight')
    make_filter_figure(sim_ng, estimates_nge, "Non-Gaussian estimator",
                       "Non-Gaussian", est_color='royalblue'
                       ).savefig("plot_nge.png", dpi=350, bbox_inches='tight')
    print("Saved: plot_3d.png  plot_ekf.png  plot_nge.png")
