import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

STATE_LABELS = [
    "$x$ (m)", "$y$ (m)", "$z$ (m)",
    "$\\psi$ (rad)", "$\\theta$ (rad)", "$p$ (m/s)",
    "$r_{\\psi}$ (rad/s)", "$r_{\\theta}$ (rad/s)", "$a$ (m/s$^2$)",
]
MEAS_LABELS = ["USBL $x$ (m)", "USBL $y$ (m)",
               "Sonar $x$ (m)", "Sonar $y$ (m)", "Sonar $z$ (m)"]
_ANGLE_IDX  = {3, 4}
USBL_COLOR  = 'steelblue'
SONAR_COLOR = 'darkorange'


def make_3d_figure(estimates_ekf=None, estimates_nge=None, rerun_callback=None):
    """3D trajectory figure.  Pass rerun_callback to add a Rerun button."""
    has_btn = rerun_callback is not None
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(bottom=0.11 if has_btn else 0.02, top=0.95)

    ax.scatter(0, 0, 0, c='black', marker='^', s=100, zorder=5,
               label="ship (origin)")
    if estimates_ekf is not None:
        ax.plot(estimates_ekf[0], estimates_ekf[1], estimates_ekf[2],
                c='crimson', linewidth=1, alpha=0.85, label="EKF")
    if estimates_nge is not None:
        ax.plot(estimates_nge[0], estimates_nge[1], estimates_nge[2],
                c='royalblue', linewidth=1, alpha=0.85,
                label="Non-Gaussian estimator")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("depth (m)")
    ax.set_title("Whale trajectory estimates", fontsize=11)
    ax.legend(fontsize=8)

    if has_btn:
        ax_btn = fig.add_axes([0.38, 0.02, 0.24, 0.055])
        btn = Button(ax_btn, 'Rerun Simulation',
                     color='#f5f0e8', hovercolor='#e8e0d0')
        btn.label.set_color('black')
        btn.label.set_fontsize(10)
        btn.label.set_fontweight('semibold')
        for spine in ax_btn.spines.values():
            spine.set_edgecolor('#c0b8a8')
        btn.on_clicked(rerun_callback)
        fig._rerun_btn = btn

    return fig


def make_filter_figure(sim, estimates, filter_label, noise_label, est_color,
                       suptitle=True):
    """3x3 state grid (left) + USBL/Sonar measurement scatter (right)."""
    N = sim.constant.N
    t = np.arange(N)

    fig = plt.figure(figsize=(20, 10))
    if suptitle:
        fig.suptitle(f"{filter_label}: {noise_label} noise",
                     fontsize=13, fontweight='bold')

    outer    = fig.add_gridspec(1, 2, width_ratios=[3, 3], wspace=0.35)
    state_gs = outer[0].subgridspec(3, 3, hspace=0.55, wspace=0.45)
    meas_gs  = outer[1].subgridspec(2, 3, hspace=0.8, wspace=0.75)

    for idx in range(9):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(state_gs[row, col])
        if estimates is not None:
            ax.plot(t, estimates[idx, :], alpha=0.85, c=est_color, linewidth=1,
                    label=filter_label if idx == 0 else "_")
            if idx == 0:
                ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "not implemented",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=8, color='gray', style='italic')
        if idx in _ANGLE_IDX:
            ax.set_ylim(-np.pi, np.pi)
        ax.set_ylabel(STATE_LABELS[idx], fontsize=8)
        ax.set_xlabel("step", fontsize=7)
        ax.tick_params(labelsize=7)

    for col_idx, j in enumerate([0, 1]):
        ax = fig.add_subplot(meas_gs[0, col_idx])
        meas_vals = sim.measurements[j, :]
        valid = ~np.isnan(meas_vals)
        ax.scatter(t[valid], meas_vals[valid],
                   c=USBL_COLOR, s=4, alpha=0.6)
        ax.set_ylabel(MEAS_LABELS[j], fontsize=8)
        ax.set_xlabel("step", fontsize=7)
        ax.tick_params(labelsize=7)

    for col_idx, j in enumerate([2, 3, 4]):
        ax = fig.add_subplot(meas_gs[1, col_idx])
        ax.scatter(t, sim.measurements[j, :],
                   c=SONAR_COLOR, s=4, alpha=0.6)
        ax.set_ylabel(MEAS_LABELS[j], fontsize=8)
        ax.set_xlabel("step", fontsize=7)
        ax.tick_params(labelsize=7)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
