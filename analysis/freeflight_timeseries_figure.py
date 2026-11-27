#!/usr/bin/env python3
"""Take-off time histories of one hardware case: tilt and horizontal
drift against time from lift-off, with and without the balanced-moment
feed-forward.

Same definitions as analysis/freeflight_metrics.py: t_lo is the instant
the rotor-speed thrust first carries the weight, tilt is the angle
between the body and world vertical axes (yaw-free), drift is
||p_xy(t) - p_xy(t_lo)|| from odometry, and the published peaks are
taken over [t_lo, t_70] (70 % of the commanded 0.20 m climb).  All
three repeats of each variant are drawn, aligned at t_lo; the shaded
band is the median metric window.

Usage: python analysis/freeflight_timeseries_figure.py [case] [ctrl] [out.png]
       e.g. 02 hgdo docs/exp_takeoff_timeseries.png
"""
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import freeflight_metrics as fm                       # noqa: E402

STYLE = {'wo_ff': dict(color='#1f3b66', lw=1.5, label='no compensation'),
         'ff_pivot_free': dict(color='#e8743b', lw=1.5, label=r'$M_{\mathrm{bal}}$ feed-forward (pivot-free, reported)'),
         'ff_pivot_based': dict(color='#9fc5e8', lw=1.1, ls='--', label=r'$M_{\mathrm{bal}}$ feed-forward (pivot-based)')}
T0, T1 = -0.3, 2.8                                     # s around lift-off


def series(path, case):
    d = np.load(path)
    t = d['odom/t']
    W = fm.MASS[case] * fm.G
    f = fm.C_T * np.sum(d['rpm/rpm'].astype(np.float64) ** 2, axis=1)
    t_lo = float(d['rpm/t'][np.flatnonzero(f >= W)[0]])
    pos = d['odom/position'].astype(np.float64)
    i_lo = int(np.searchsorted(t, t_lo))
    climbed = pos[:, 2] - pos[i_lo, 2]
    up = np.flatnonzero((climbed >= fm.CLIMB_FRAC * fm.Z_REF) & (np.arange(len(t)) > i_lo))
    t_70 = float(t[up[0]])
    q = d['odom/quaternion'].astype(np.float64)
    tilt = fm.tilt_from_vertical(q)
    drift = np.hypot(pos[:, 0] - pos[i_lo, 0], pos[:, 1] - pos[i_lo, 1])
    k = (t >= t_lo + T0) & (t <= t_lo + T1)
    return t[k] - t_lo, tilt[k], drift[k], t_70 - t_lo


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else '02'
    ctrl = sys.argv[2] if len(sys.argv) > 2 else 'hgdo'
    out = sys.argv[3] if len(sys.argv) > 3 else 'docs/exp_takeoff_timeseries.png'
    x, y = fm.OFFSET[case]
    fig, (A, B) = plt.subplots(1, 2, figsize=(10.0, 3.9))
    win = []
    peaks = {}
    for var in ('ff_pivot_based', 'ff_pivot_free', 'wo_ff'):
        files = sorted(glob.glob(f'DataSet/free_flight/{case}/{ctrl}/{var}*.npz'))
        st = dict(STYLE[var])
        lab = st.pop('label')
        pk_t, pk_d = [], []
        for k, f in enumerate(files):
            tau, tilt, drift, t70 = series(f, case)
            win.append(t70)
            m = tau <= t70
            pk_t.append(tilt[m].max())
            pk_d.append(drift[m].max())
            A.plot(tau, tilt, label=lab if k == 0 else None, **st)
            B.plot(tau, drift, label=lab if k == 0 else None, **st)
        peaks[var] = (np.median(pk_t), np.median(pk_d), len(files))
        print(f'{case}/{ctrl}/{var}: {len(files)} repeats, median peak tilt {np.median(pk_t):.1f} deg, '
              f'median drift {np.median(pk_d):.3f} m')
    tw = float(np.median(win))
    for ax in (A, B):
        ax.axvspan(0, tw, color='0.9', alpha=0.6, lw=0, zorder=0)
        ax.axvline(0, ls=':', lw=1.0, color='0.4')
        ax.grid(alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=8.5)
        ax.set_xlim(T0, T1)
        ax.set_xlabel(r'$t - t_{\mathrm{lo}}$ [s]  (lift-off at 0; shaded: metric window to $t_{70}$)', fontsize=9)
    A.set_ylabel('tilt from vertical [deg]', fontsize=9.5)
    B.set_ylabel(r'horizontal drift $\|p_{xy}(t) - p_{xy}(t_{\mathrm{lo}})\|$ [m]', fontsize=9.5)
    A.set_title(f'(a) tilt — E{int(case)} ({x:+.1f}, {y:+.1f}) mm, {ctrl.upper()}, 3 repeats each', fontsize=10)
    B.set_title('(b) horizontal drift', fontsize=10)
    hdl, lab = A.get_legend_handles_labels()
    order = [lab.index(STYLE[v]['label']) for v in ('wo_ff', 'ff_pivot_free', 'ff_pivot_based')]
    fig.legend([hdl[i] for i in order], [lab[i] for i in order], loc='lower center', ncol=3,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    u, c = peaks['wo_ff'], peaks['ff_pivot_free']
    A.text(0.02, 0.97, f'median peak: {u[0]:.1f}° → {c[0]:.1f}°', transform=A.transAxes, fontsize=9,
           va='top', bbox=dict(fc='white', ec='0.8', lw=0.6, pad=3))
    B.text(0.02, 0.97, f'median peak: {u[1]:.2f} m → {c[1]:.2f} m', transform=B.transAxes, fontsize=9,
           va='top', bbox=dict(fc='white', ec='0.8', lw=0.6, pad=3))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out, dpi=600)
    print('written', out)


if __name__ == '__main__':
    main()
