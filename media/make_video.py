#!/usr/bin/env python3
"""
Render an animation of both estimators tracking a simulated dive.

The provided simulator hides the true state, so this script re-implements the
whale model and the non-Gaussian sensor models from the problem statement to
obtain ground truth for visualisation. Both the EKF and the particle filter
are run on the same non-Gaussian measurement sequence.

Usage (from the repository root):
    python media/make_video.py            # writes media/demo.mp4 and media/demo.gif
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from const import EstimatorConstant
from estimator import EKF, NonGaussianEstimator, _theta_star

OUT_DIR = os.path.join(ROOT, "media")


def simulate(c, rng):
    """Ground-truth trajectory and non-Gaussian measurements."""
    N, Ts = c.N, c.Ts
    X = np.zeros((9, N))
    rad, ang = c.R * np.sqrt(rng.random()), rng.uniform(-np.pi, np.pi)
    X[:, 0] = [rad * np.cos(ang), rad * np.sin(ang), c.z0,
               rng.uniform(-c.psi_bound, c.psi_bound),
               rng.uniform(c.theta0_min, c.theta0_max),
               rng.uniform(c.p0_min, c.p0_max), 0.0, 0.0, 0.0]
    s3 = np.sqrt(3.0)
    for k in range(1, N):
        x, y, z, psi, th, p, r_psi, r_th, a = X[:, k - 1]
        v = rng.uniform(-s3, s3, 3) * [c.sigma_r_psi, c.sigma_r_theta, c.sigma_a]
        X[:, k] = [
            x + p * np.cos(th) * np.cos(psi) * Ts,
            y + p * np.cos(th) * np.sin(psi) * Ts,
            max(0.0, z + p * np.sin(th) * Ts),
            psi + r_psi * Ts,
            th + r_th * Ts,
            p + (a - c.cd * p) * Ts,
            r_psi + c.K_yaw * (c.omega_yaw - r_psi) + v[0],
            r_th + c.K_theta_p * (_theta_star(c, k - 1) - th) - c.K_theta_d * r_th + v[1],
            a + c.K_p_p * (c.p_cruise - p) - c.K_p_d * a + v[2],
        ]

    Z = np.zeros((5, N))
    sig_u = c.sigma_base + c.alpha * X[2]
    Z[:2] = X[:2] + rng.uniform(-s3, s3, (2, N)) * sig_u
    Z[:2, X[2] >= c.z_drop] = np.nan

    beta = c.sonar_bimodality
    std = np.sqrt(1.0 - 9.0 * beta ** 2) * c.sigma_S
    comp = rng.random((3, N)) < c.sonar_mix_weight
    mu = np.where(comp, beta * c.sigma_S, -9.0 * beta * c.sigma_S)
    Z[2:] = X[:3] + mu + std * rng.standard_normal((3, N))
    return X, Z


def run_filters(c, Z):
    N = c.N
    ekf, pf = EKF(c), NonGaussianEstimator(c)
    Xe, Xp = np.zeros((9, N)), np.zeros((9, N))
    Xe[:, 0] = ekf.initialize()[0]
    Xp[:, 0] = pf.initialize()
    for k in range(1, N):
        Xe[:, k] = ekf.estimate(Z[:, k].copy())[0]
        Xp[:, k] = pf.estimate(Z[:, k].copy())
    return Xe, Xp


def main(seed=7, step=1, fps=30):
    c = EstimatorConstant()
    rng = np.random.default_rng(seed)
    X, Z = simulate(c, rng)
    Xe, Xp = run_filters(c, Z)
    N = c.N
    t = np.arange(N) * c.Ts / 60.0  # minutes

    err_e = np.linalg.norm(Xe[:3] - X[:3], axis=0)
    err_p = np.linalg.norm(Xp[:3] - X[:3], axis=0)

    C_TRUE, C_EKF, C_PF, C_SON, C_USBL = "#222222", "#d1495b", "#2e86ab", "#f4a259", "#5b8e7d"

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], hspace=0.35,
                          wspace=0.12, left=0.02, right=0.97, top=0.9, bottom=0.08)
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    axz = fig.add_subplot(gs[0, 1])
    axe = fig.add_subplot(gs[1, 1])
    fig.suptitle("Whale tracking: EKF vs. particle filter (non-Gaussian noise)",
                 fontsize=14, fontweight="bold")

    # 3D view
    pad = 150
    allxy = np.concatenate([X[:2], Xp[:2], Xe[:2]], axis=1)
    ax3.set_xlim(allxy[0].min() - pad, allxy[0].max() + pad)
    ax3.set_ylim(allxy[1].min() - pad, allxy[1].max() + pad)
    ax3.set_zlim(X[2].max() + 300, -100)
    ax3.set_xlabel("x (m)"); ax3.set_ylabel("y (m)"); ax3.set_zlabel("depth (m)")
    ax3.scatter(0, 0, 0, c="black", marker="^", s=80, label="ship")
    l_true, = ax3.plot([], [], [], c=C_TRUE, lw=2.0, label="true")
    l_ekf, = ax3.plot([], [], [], c=C_EKF, lw=1.3, label="EKF")
    l_pf, = ax3.plot([], [], [], c=C_PF, lw=1.3, label="particle filter")
    s_son = ax3.scatter([], [], [], c=C_SON, s=6, alpha=0.5, label="sonar")
    ax3.legend(loc="upper left", fontsize=9, frameon=False)

    # Depth vs time
    axz.axhspan(c.z_drop, X[2].max() + 400, color="0.92", zorder=0)
    axz.text(0.5, c.z_drop + 80, "USBL dropout", color="0.4", fontsize=9, va="top")
    axz.set_xlim(0, t[-1]); axz.set_ylim(X[2].max() + 400, -150)
    axz.set_ylabel("depth (m)"); axz.set_xlabel("time (min)")
    d_son = axz.scatter([], [], c=C_SON, s=4, alpha=0.45)
    d_true, = axz.plot([], [], c=C_TRUE, lw=2.0)
    d_ekf, = axz.plot([], [], c=C_EKF, lw=1.2)
    d_pf, = axz.plot([], [], c=C_PF, lw=1.2)

    # Position error vs time
    axe.set_xlim(0, t[-1])
    axe.set_ylim(0, np.percentile(np.r_[err_e, err_p], 99) * 1.15)
    axe.set_ylabel("3D position error (m)"); axe.set_xlabel("time (min)")
    e_ekf, = axe.plot([], [], c=C_EKF, lw=1.2, label="EKF")
    e_pf, = axe.plot([], [], c=C_PF, lw=1.2, label="particle filter")
    axe.legend(loc="upper right", fontsize=9, frameon=False)
    txt = axe.text(0.02, 0.95, "", transform=axe.transAxes, va="top", fontsize=9,
                   family="monospace")

    frames = list(range(1, N, step)) + [N - 1] * fps

    def update(k):
        s = slice(0, k + 1)
        for line, data in ((l_true, X), (l_ekf, Xe), (l_pf, Xp)):
            line.set_data(data[0, s], data[1, s])
            line.set_3d_properties(data[2, s])
        lo = max(0, k - 40)
        s_son._offsets3d = (Z[2, lo:k + 1], Z[3, lo:k + 1], Z[4, lo:k + 1])
        ax3.view_init(elev=22, azim=-60 + 50 * k / N)

        d_son.set_offsets(np.c_[t[s], Z[4, s]])
        d_true.set_data(t[s], X[2, s])
        d_ekf.set_data(t[s], Xe[2, s])
        d_pf.set_data(t[s], Xp[2, s])

        e_ekf.set_data(t[s], err_e[s])
        e_pf.set_data(t[s], err_p[s])
        rms_e = np.sqrt(np.mean(err_e[s] ** 2))
        rms_p = np.sqrt(np.mean(err_p[s] ** 2))
        txt.set_text(f"RMS so far   EKF {rms_e:6.1f} m\n"
                     f"             PF  {rms_p:6.1f} m")
        return ()

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    mp4 = os.path.join(OUT_DIR, "demo.mp4")
    anim.save(mp4, writer=FFMpegWriter(fps=fps, bitrate=2500,
                                       extra_args=["-pix_fmt", "yuv420p"]))
    print(f"Saved {mp4}")

    # Smaller GIF for inline display in the README
    fig.set_dpi(60)
    anim_gif = FuncAnimation(fig, update, frames=frames[::3], blit=False)
    gif = os.path.join(OUT_DIR, "demo.gif")
    anim_gif.save(gif, writer=PillowWriter(fps=fps // 3))
    print(f"Saved {gif}")


if __name__ == "__main__":
    main()
