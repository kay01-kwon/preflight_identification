#!/usr/bin/env python3
"""
Chart: dynamics moment residual vs the theoretical rotor GE moment
===================================================================
With z_CoM and J_P pinned by the caller (defaults 0.267 m, 0.311 kg m^2)
and the pivot arm taken from the mocap circle fit, the contact-phase
balance has NO free parameter left:

    r(t) = J_P phi'' - M - f a + W [ a cos(dphi) - z_CoM sin(dphi) ]

r(t) is what the rotation dynamics says is missing; dM_GE(t) is what the
rotor ground-effect models predict on the same measured trajectory.  The
figure puts the two side by side over the whole dataset, in mN.m, as a
function of the tilt the vehicle has accumulated:

  (a) the two moments themselves — pooled median and IQR over all runs;
  (b) their difference r - dM_GE, as it stands (no fitting);
  (c) both with each run's own mean removed — the shape alone, which is
      the only part an arm error cannot account for.

The balance only holds once the vehicle is actually turning, so the
window starts at --dphi-min of accumulated tilt (default 1 deg) and runs
to the moment peak.  Every run of every case and both axes is pooled.

Usage
-----
python analysis/ge_error_chart.py DataSet/exp
python analysis/ge_error_chart.py DataSet/exp --z-com 0.267 --jp 0.311
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import contextlib
import csv
import io
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.extractor import load_excitation_dataset
from utils import math_tools
from critical_value_getter_piecewise import (
    prepare_signals, detect_excitation_window, estimate_pivot_from_mocap,
    MOMENT_CAP)
from ge_residual_vs_model import ge_moment, MODELS

G = 9.81
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#8f8e88'
GRID = '#e4e3dd'
SURF = '#fcfcfb'
COL = {'single': '#2a78d6', 'interf': '#eb6834', 'garofano': '#1baf7a'}
LBL = {'single': 'GE: single-rotor', 'interf': 'GE: interference',
       'garofano': 'GE: Garofano'}


def band(ax, x, Y, color, label, lw=2.0, ls='-'):
    """Pooled median + IQR of the (runs x grid) matrix Y."""
    n = np.sum(np.isfinite(Y), axis=0)
    ok = n >= 5
    med = np.nanmedian(Y[:, ok], axis=0)
    lo = np.nanpercentile(Y[:, ok], 25, axis=0)
    hi = np.nanpercentile(Y[:, ok], 75, axis=0)
    ax.fill_between(x[ok], lo, hi, color=color, alpha=0.16, lw=0)
    ax.plot(x[ok], med, color=color, lw=lw, ls=ls, label=label,
            solid_capstyle='round')
    return x[ok], med, n


def main():
    p = argparse.ArgumentParser(
        description="Chart the dynamics residual against the GE models.")
    p.add_argument('root')
    p.add_argument('--z-com', type=float, default=0.267)
    p.add_argument('--jp', type=float, default=0.311)
    p.add_argument('--mass', type=float, default=3.22)
    p.add_argument('--omega-source', choices=['odom', 'imu'], default='imu',
                   help="rate channel.  DEFAULT imu: the odometry's "
                        "twist.angular is an EKF output that integrates to "
                        "only 0.914 +- 0.060 of the attitude change over the "
                        "ramp (raw gyro: 0.993 +- 0.064) and to ~0.75 of it "
                        "during the fast fall, which biases phi'' here.")
    p.add_argument('--lpf-cutoff', type=float, default=15.0)
    p.add_argument('--savgol', type=int, default=15)
    p.add_argument('--grid-max', type=float, default=10.0,
                   help="tilt grid upper edge [deg]")
    p.add_argument('--grid-step', type=float, default=0.25)
    p.add_argument('--min-samples', type=int, default=20)
    p.add_argument('--dphi-min', type=float, default=1.0,
                   help="start the window at this accumulated tilt [deg]; "
                        "below it the vehicle is still static and the "
                        "rotational balance does not apply")
    p.add_argument('--arm', type=float, default=0.265)
    p.add_argument('--radius', type=float, default=0.127)
    p.add_argument('--hub-height', type=float, default=0.315)
    p.add_argument('--lp-roll', type=float, default=0.140)
    p.add_argument('--lp-pitch', type=float, default=0.110)
    p.add_argument('--frame-width', type=float, default=0.22)
    p.add_argument('--jk', type=float, default=2.2)
    p.add_argument('--c-t', type=float, default=1.3175e-7)
    p.add_argument('--output-dir', default='docs')
    args = p.parse_args()

    W = args.mass * G
    grid = np.arange(0.0, args.grid_max + 1e-9, args.grid_step)
    root = Path(args.root)
    res, th, names, sign_rows = [], {m: [] for m in MODELS}, [], []

    for d in sorted(root.glob('case_*/M[xy]')):
        axis = 'x' if d.name == 'Mx' else 'y'
        with contextlib.redirect_stdout(io.StringIO()):
            bags = load_excitation_dataset(d)
        for b in bags:
            sig = prepare_signals(
                b, axis, omega_source=args.omega_source,
                lpf_cutoff=(args.lpf_cutoff
                            if args.omega_source == 'imu'
                            and args.lpf_cutoff > 0 else None))
            t, om, M, f = sig['t'], sig['omega'], sig['moment'], sig['f_col']
            roll, pitch = math_tools.quaternion_to_euler_vectorized(
                b.odom.quaternion)
            # the attitude lives on the odom clock; interpolate rather than
            # truncate so it stays aligned with the chosen rate channel
            phi = np.interp(t, b.odom.t - b.odom.t[0],
                            roll if axis == 'x' else pitch)
            n = len(t)
            i0, i1 = detect_excitation_window(M,
                                              moment_cap=MOMENT_CAP.get(axis))
            if i1 - i0 < args.min_samples:
                continue
            pos = b.name.startswith('pos')
            sgn = +1.0 if pos else -1.0
            dphi = sgn * (phi - float(np.median(phi[:max(1, i0)])))
            piv = estimate_pivot_from_mocap(b, t[i0], axis)
            a = piv['pivot_abs'] * 1e-3
            if not np.isfinite(a):
                continue
            wl = args.savgol if args.savgol % 2 else args.savgol + 1
            if i1 - i0 < wl + 2:
                continue
            alpha = savgol_filter(om, wl, 2, deriv=1,
                                  delta=float(np.median(np.diff(t))))
            ge, okge, m = ge_moment(b, axis, pos, t, n, args)
            k1 = min(i1, m - 1)
            k0 = i0
            if k1 - k0 < args.min_samples or not okge[k0:k1 + 1].all():
                continue
            # the balance applies only once it is turning
            over = np.where(np.degrees(dphi[k0:k1 + 1])
                            >= args.dphi_min)[0]
            if len(over) == 0:
                continue
            k0 = k0 + int(over[0])
            if k1 - k0 < args.min_samples:
                continue
            s = slice(k0, k1 + 1)
            r = (args.jp * sgn * alpha[s] - sgn * M[s] - f[s] * a
                 + W * (a * np.cos(dphi[s]) - args.z_com * np.sin(dphi[s])))
            x = np.degrees(dphi[s])
            keep = np.concatenate([[True], np.diff(x) > 1e-6])
            if keep.sum() < 8:
                continue
            xi, ri = x[keep], r[keep]
            def onto(v):
                out = np.full_like(grid, np.nan)
                inside = (grid >= xi[0]) & (grid <= xi[-1])
                out[inside] = np.interp(grid[inside], xi, v)
                return out
            res.append(onto(ri))
            for name in MODELS:
                th[name].append(onto((sgn * ge[name][s])[keep]))
            sign_rows.append({name: (float(np.mean(sgn * ge[name][s] > 0)),
                                     float(np.mean(sgn * ge[name][s])),
                                     float(np.polyfit(x, sgn * ge[name][s],
                                                      1)[0]))
                              for name in MODELS})
            names.append(f"{d.parent.name}/{d.name}/{b.name}")
        print(f"  assessed {d}", flush=True)

    R = 1e3 * np.array(res)
    T = {m: 1e3 * np.array(th[m]) for m in MODELS}
    print(f"\n{len(R)} runs pooled;  z_CoM = {1e3 * args.z_com:.0f} mm, "
          f"J_P = {args.jp:.3f} kg m^2, arm from the mocap circle fit, "
          f"no free parameter")

    # ── figure ─────────────────────────────────────────────────────
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID,
                         'axes.labelcolor': INK2, 'xtick.color': INK2,
                         'ytick.color': INK2, 'figure.facecolor': SURF,
                         'axes.facecolor': SURF, 'savefig.facecolor': SURF})
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.3))
    for ax in axs:
        ax.grid(True, color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.axhline(0, color=MUTED, lw=0.9, zorder=1)
        ax.set_xlabel('tilt accumulated since rest,  δφ  [deg]')
        ax.set_xlim(args.dphi_min, args.grid_max)

    # (a) the two moments
    ax = axs[0]
    xr, medr, nrun = band(ax, grid, R, INK, 'dynamics residual', lw=2.4)
    for m in MODELS:
        band(ax, grid, T[m], COL[m], LBL[m])
    ax.set_ylabel('moment about the contact line  [mN·m]')
    ax.set_title('(a)  what the dynamics is missing,\nand what the GE models '
                 'predict', color=INK, loc='left', fontsize=10)
    for m in MODELS:
        yv = np.nanmedian(T[m][:, np.argmin(np.abs(grid - 8.0))])
        ax.annotate(LBL[m].replace('GE: ', ''), (8.0, yv), xytext=(2, 7),
                    textcoords='offset points', color=COL[m], fontsize=8,
                    va='bottom', ha='left')
    yv = np.nanmedian(R[:, np.argmin(np.abs(grid - 6.0))])
    ax.annotate('dynamics residual', (6.0, yv), xytext=(4, -12),
                textcoords='offset points', color=INK, fontsize=8)
    j2 = np.argmin(np.abs(grid - 2.0))
    j8a = np.argmin(np.abs(grid - 8.0))
    ax.annotate(f'runs pooled: {int(nrun[j2])} at 2°, {int(nrun[j8a])} at 8°'
                f'   ·   band = IQR', (0.02, 0.03), xycoords='axes fraction',
                color=MUTED, fontsize=7.5)

    # (b) raw difference
    ax = axs[1]
    for m in MODELS:
        band(ax, grid, R - T[m], COL[m], LBL[m])
    ax.set_ylabel('residual − theory  [mN·m]')
    ax.set_title('(b)  difference, as it stands\n(no fitting)', color=INK,
                 loc='left', fontsize=10)
    for k, m in enumerate(MODELS):
        xa = 8.6
        yv = np.nanmedian((R - T[m])[:, np.argmin(np.abs(grid - xa))])
        ax.annotate(LBL[m].replace('GE: ', ''), (xa, yv), xytext=(2, 7),
                    textcoords='offset points', color=COL[m], fontsize=8,
                    va='bottom', ha='left')

    # (c) shape only: the residual's shape against the models' shape
    ax = axs[2]
    D = {}
    Rd = R - np.nanmean(R, axis=1, keepdims=True)
    band(ax, grid, Rd, INK, 'dynamics residual', lw=2.4)
    for m in MODELS:
        Td = T[m] - np.nanmean(T[m], axis=1, keepdims=True)
        D[m] = (R - T[m]) - np.nanmean(R - T[m], axis=1, keepdims=True)
        band(ax, grid, Td, COL[m], LBL[m])
    ax.set_ylabel('same, per-run mean removed  [mN·m]')
    ax.set_title('(c)  shape alone — the part no arm\nerror can absorb',
                 color=INK, loc='left', fontsize=10)
    yv = np.nanmedian(Rd[:, np.argmin(np.abs(grid - 7.0))])
    ax.annotate('dynamics residual', (7.0, yv), xytext=(2, -13),
                textcoords='offset points', color=INK, fontsize=8)
    ax.annotate('all three GE models\n(within ±20 mN·m of zero)',
                (4.0, 0), xytext=(0, 26), textcoords='offset points',
                color=COL['interf'], fontsize=8, ha='center')

    fig.text(0.008, 0.965, "Rotation-dynamics moment residual vs the "
             "theoretical rotor ground-effect moment", color=INK,
             fontsize=11.5, ha='left', va='top')
    fig.text(0.008, 0.925, f"{len(R)} runs · 5 configurations · both axes   |   "
             f"z_CoM = {1e3 * args.z_com:.0f} mm, J_P = {args.jp:.3f} kg·m², "
             f"pivot arm from the mocap circle fit — no free parameter",
             color=INK2, fontsize=9, ha='left', va='top')
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'fig_ge_error.{ext}', dpi=200)
    print(f"Figure -> {out / 'fig_ge_error.png'} (+ .pdf)")

    # ── table view (the contrast-WARN relief) ──────────────────────
    with open(out / 'ge_error_by_tilt.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dphi_deg', 'n_runs', 'residual_mNm']
                   + [f'{m}_{k}' for m in MODELS
                      for k in ('theory_mNm', 'diff_mNm', 'diff_demeaned')])
        for i, x in enumerate(grid):
            if nrun[min(i, len(nrun) - 1)] < 5:
                continue
            row = [f'{x:.2f}', int(np.sum(np.isfinite(R[:, i]))),
                   f'{np.nanmedian(R[:, i]):.1f}']
            for m in MODELS:
                row += [f'{np.nanmedian(T[m][:, i]):.1f}',
                        f'{np.nanmedian(R[:, i] - T[m][:, i]):.1f}',
                        f'{np.nanmedian(D[m][:, i]):.1f}']
            w.writerow(row)
    print(f"Table  -> {out / 'ge_error_by_tilt.csv'}")

    print(f"\n  {'model':11}{'theory @8°':>12}{'diff @8°':>11}"
          f"{'diff level':>12}{'diff RMS':>11}{'demeaned RMS':>14}")
    j8 = np.argmin(np.abs(grid - 8.0))
    for m in MODELS:
        E = R - T[m]
        print(f"  {m:11}{np.nanmedian(T[m][:, j8]):12.0f}"
              f"{np.nanmedian(E[:, j8]):11.0f}"
              f"{np.nanmedian(np.nanmean(E, axis=1)):12.0f}"
              f"{np.nanmedian(np.sqrt(np.nanmean(E ** 2, axis=1))):11.0f}"
              f"{np.nanmedian(np.sqrt(np.nanmean(D[m] ** 2, axis=1))):14.0f}")
    print("  [mN·m]  level/RMS are per-run figures, median over runs")

    # ── which way does the GE moment push? ─────────────────────────
    stiff = 1e3 * W * args.z_com * np.pi / 180.0
    print(f"\n  DIRECTION.  sgn * dM_GE > 0 means the ground effect acts WITH "
          f"the tip-over.")
    print(f"  {'model':11}{'samples with sgn·GE>0':>24}{'level':>14}"
          f"{'slope [mN·m/deg]':>19}{'vs W·z_CoM':>13}")
    for m in MODELS:
        pf = 100 * np.mean([r[m][0] for r in sign_rows])
        lv = 1e3 * np.array([r[m][1] for r in sign_rows])
        sl = 1e3 * np.array([r[m][2] for r in sign_rows])
        print(f"  {m:11}{pf:21.1f} %{lv.mean():9.0f} ±{lv.std(ddof=1):3.0f}"
              f"{sl.mean():14.1f} ±{sl.std(ddof=1):4.1f}"
              f"{100 * sl.mean() / stiff:11.2f} %")
    rs = np.polyfit(grid[np.sum(np.isfinite(R), axis=0) >= 5],
                    np.nanmedian(R[:, np.sum(np.isfinite(R), axis=0) >= 5],
                                 axis=0), 1)[0]
    print(f"  gravitational anti-restoring term W z_CoM = {stiff:.0f} mN·m/deg "
          f"at z_CoM = {1e3 * args.z_com:.0f} mm")
    print(f"  -> the GE LEVEL always aids the tip; only its GRADIENT opposes "
          f"further\n     tilting, and that gradient is under "
          f"{abs(100 * min(np.mean([r[m][2] for r in sign_rows]) for m in MODELS) * 1e3 / stiff):.1f}% of the "
          f"gravitational term.")
    print(f"  The residual's own slope is {rs:.0f} mN·m/deg — same sign, "
          f"{abs(rs / (1e3 * np.mean([r['garofano'][2] for r in sign_rows]))):.0f}x "
          f"the largest model — i.e. a z_CoM error of "
          f"{abs(rs) / stiff * 1e3 * args.z_com:.0f} mm, not ground effect.")


if __name__ == '__main__':
    main()
