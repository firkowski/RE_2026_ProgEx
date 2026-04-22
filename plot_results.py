#!/usr/bin/env python3
import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_DIR)
sys.path.insert(0, _DIR)

from plotting import make_3d_figure, make_filter_figure

_RESULTS = os.path.join(_DIR, 'results.npz')
_IMAGE   = 're-student'
_COMMON  = ['--rm', '--cpus=4', '--memory=2g', '--memory-swap=2g',
            '-v', f'{_DIR}:/work', '-w', '/work']


def _load():
    data = np.load(_RESULTS, allow_pickle=False)
    N         = int(data['N'])
    ekf_impl  = bool(data['ekf_implemented'])
    nge_impl  = bool(data['nge_implemented'])
    return dict(
        N             = N,
        estimates_ekf = data['estimates_ekf'] if ekf_impl else None,
        estimates_nge = data['estimates_nge'] if nge_impl else None,
        ekf_rmse      = data['ekf_rmse']      if ekf_impl else None,
        ekf_timing    = float(data['ekf_timing']),
        ekf_dt        = float(data['ekf_dt']),
        nge_rmse      = data['nge_rmse']      if nge_impl else None,
        nge_timing    = float(data['nge_timing']),
        nge_dt        = float(data['nge_dt']),
        sim_g  = SimpleNamespace(measurements=data['measurements_g'],
                                 constant=SimpleNamespace(N=N)),
        sim_ng = SimpleNamespace(measurements=data['measurements_ng'],
                                 constant=SimpleNamespace(N=N)),
    )


def _maximize(fig):
    import matplotlib as mpl
    backend = mpl.get_backend().lower()
    try:
        mng = fig.canvas.manager
        if 'tk' in backend:
            mng.window.state('zoomed')
        elif any(q in backend for q in ('qt5', 'qt6', 'qt')):
            mng.window.showMaximized()
        elif 'wx' in backend:
            mng.window.Maximize(True)
        elif 'macosx' in backend:
            mng.full_screen_toggle()
    except Exception:
        pass


def _show(d, rerun_callback):
    fig_3d  = make_3d_figure(d['estimates_ekf'], d['estimates_nge'],
                              rerun_callback=rerun_callback)
    fig_ekf = make_filter_figure(d['sim_g'],  d['estimates_ekf'],
                                 'EKF', 'Gaussian', 'crimson')
    fig_nge = make_filter_figure(d['sim_ng'], d['estimates_nge'],
                                 'Non-Gaussian Estimator', 'Non-Gaussian', 'royalblue')
    for fig in (fig_3d, fig_ekf, fig_nge):
        _maximize(fig)
    plt.show()


if __name__ == '__main__':
    if not os.path.exists(_RESULTS):
        print(f"ERROR: {_RESULTS} not found. Run run.bat / run.sh first.")
        sys.exit(1)

    def _rerun(_event):
        plt.close('all')
        print("Re-running simulation in Docker...")
        ret = subprocess.run(
            ['docker', 'run'] + _COMMON + [_IMAGE, 'python', 'run.py']
        ).returncode
        if ret == 0:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        else:
            print("ERROR: Docker run failed.")

    _show(_load(), rerun_callback=_rerun)
