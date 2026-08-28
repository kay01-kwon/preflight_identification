#!/usr/bin/env python3
"""Summary of the simulated free-flight validation campaign.

Reads every packed flight under SimDataSet/free_flight and produces
the three artefacts the manuscript needs:

  tab_ff_sim.tex        per case and controller: the four take-off
                        metrics with and without the feedforward
  fig_ff_compass        offset bearing against the uncompensated
                        drift bearing, one point per case/controller
  fig_ff_scaling        uncompensated disturbance growing with the
                        offset magnitude, compensated floor flat

Metric definitions follow the hardware campaign exactly
(analysis/freeflight_metrics.py, section VIII-A): window from t_lo
(reconstructed thrust first carrying the weight) to t_70 (70% of the
0.20 m commanded climb); peak tilt from the quaternion's r33, peak
horizontal rate, farthest horizontal excursion, peak horizontal speed.

Usage
-----
  PYTHONPATH=<stubs> python analysis/freeflight_sim_summary.py \
      [--outdir DIR] [--dpi N]
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402

C_T = 1.3175e-7; G = 9.81
Z_REF, CLIMB_FRAC = 0.20, 0.70
# id -> (x_off, y_off [mm], mass [kg])
CASES = {
    'S1': (-6.0, 0.0, 3.066),   'S2': (0.0, 10.0, 3.066),
    'S3': (10.0, -5.0, 3.066),  'S4': (20.0, 20.0, 3.066),
    'S5': (20.0, -20.0, 3.066), 'S6': (-20.0, 20.0, 3.066),
    'S7': (-20.0, -20.0, 3.066),'S8': (25.0, 25.0, 3.066),
    'S9': (32.0, 32.0, 3.066),  'S11': (38.0, 14.0, 3.066),
    'S13': (25.0, 25.0, 3.220),
}
KEYS = ('tilt', 'rate', 'drift', 'speed')
COL = {'HGDO': '#0072B2', 'L1': '#D55E00'}
MRK = {'HGDO': 'o', 'L1': 's'}


def metrics(path, mass):
    d = np.load(path)
    t = d['odom/t']
    W = mass * G
    f = C_T * np.sum(d['rpm/rpm'].astype(np.float64) ** 2, axis=1)
    hit = np.flatnonzero(f >= W)
    if not hit.size:
        return None
    i_lo = int(np.searchsorted(t, float(d['rpm/t'][hit[0]])))
    pos = d['odom/position'].astype(np.float64)
    climbed = pos[:, 2] - pos[i_lo, 2]
    up = np.flatnonzero((climbed >= CLIMB_FRAC * Z_REF)
                        & (np.arange(len(t)) > i_lo))
    if not up.size:
        return None
    s = slice(i_lo, int(up[0]) + 1)
    q = d['odom/quaternion'].astype(np.float64)[s]
    w = d['odom/angular_vel'].astype(np.float64)[s]
    v = d['odom/linear_vel'].astype(np.float64)[s]
    dxy = pos[s, :2] - pos[i_lo, :2]
    drift = np.hypot(dxy[:, 0], dxy[:, 1])
    j = int(np.argmax(drift))
    r33 = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    return dict(
        tilt=float(np.degrees(np.arccos(np.clip(r33, -1, 1))).max()),
        rate=float(np.degrees(np.hypot(w[:, 0], w[:, 1])).max()),
        drift=float(drift.max()),
        speed=float(np.hypot(v[:, 0], v[:, 1]).max()),
        ddir=float(np.degrees(np.arctan2(dxy[j, 1], dxy[j, 0]))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('SimDataSet/free_flight'))
    p.add_argument('--outdir', type=Path,
                   default=Path(__file__).resolve().parents[1] / 'docs')
    p.add_argument('--dpi', type=int, default=600,
                   help='raster resolution of the PNG outputs')
    a = p.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    cells = defaultdict(list)          # (case, ctrl, variant) -> [metrics]
    for path in sorted(a.data.rglob('*.npz')):
        case, ctrl = path.parts[-3], path.parts[-2]
        if case not in CASES:
            continue
        var = 'w_ff' if path.stem.startswith('w_ff') else 'wo_ff'
        m = metrics(path, CASES[case][2])
        if m is None:
            print(f'  skipped {path}')
            continue
        cells[(case, ctrl, var)].append(m)

    mean = {}
    for k, v in cells.items():
        mean[k] = {kk: float(np.mean([x[kk] for x in v]))
                   for kk in ('tilt', 'rate', 'drift', 'speed')}
        # bearings live on the circle: the arithmetic mean of -180 and
        # +179 is not -180.5 but 0, so average unit vectors instead
        ang = np.radians([x['ddir'] for x in v])
        mean[k]['ddir'] = float(np.degrees(
            np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))))

    order = sorted(CASES, key=lambda c: int(c[1:]))
    ctrls = sorted({k[1] for k in mean})

    # ── per-run CSV and the LaTeX table body ─────────────────────────
    with open(a.outdir / 'ff_sim_summary.csv', 'w', newline='') as fh:
        wtr = csv.writer(fh)
        wtr.writerow(['case', 'controller', 'variant'] + list(KEYS) + ['ddir'])
        for k in sorted(mean):
            wtr.writerow(list(k) + [f'{mean[k][kk]:.4f}'
                                    for kk in KEYS + ('ddir',)])

    with open(a.outdir / 'tab_ff_sim.tex', 'w') as fh:
        for case in order:
            x, y, m0 = CASES[case]
            for ctrl in ctrls:
                wo, w_ = mean[(case, ctrl, 'wo_ff')], mean[(case, ctrl, 'w_ff')]
                lead = (f'{case} $({x:+.0f},{y:+.0f})$'
                        if ctrl == ctrls[0] else '')
                fh.write(
                    f'{lead} & {ctrl} & '
                    f'${wo["tilt"]:.1f}$ & ${w_["tilt"]:.1f}$ & '
                    f'${wo["rate"]:.0f}$ & ${w_["rate"]:.0f}$ & '
                    f'${100*wo["drift"]:.1f}$ & ${100*w_["drift"]:.1f}$ & '
                    f'${wo["speed"]:.2f}$ & ${w_["speed"]:.2f}$ \\\\\n')

    # improvement statistics across all cells
    imp = [100 * (mean[(c, ct, 'wo_ff')][k] - mean[(c, ct, 'w_ff')][k])
           / mean[(c, ct, 'wo_ff')][k]
           for c in order for ct in ctrls for k in KEYS]
    n_cells = len(order) * len(ctrls) * len(KEYS)
    print(f'{sum(1 for v in imp if v > 0)}/{n_cells} case-controller-metric '
          f'cells improve; improvement {min(imp):.0f}..{max(imp):.0f}%, '
          f'median {np.median(imp):.0f}%')

    # ── (a) the drift compass ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.plot([-180, 200], [-180, 200], color='0.55', lw=1.0, ls='--',
            zorder=1)
    for case in order:
        x, y, _ = CASES[case]
        want = np.degrees(np.arctan2(y, x))
        for ctrl in ctrls:
            got = mean[(case, ctrl, 'wo_ff')]['ddir']
            # both bearings live on the circle; plot on the branch
            # nearest the prediction so 180 and -180 do not split
            if got - want > 180:
                got -= 360
            if want - got > 180:
                got += 360
            # the two controllers drift along nearly the same bearing,
            # so concentric sizes keep both visible when they coincide
            if ctrl == 'HGDO':
                ax.plot(want, got, MRK[ctrl], ms=10, mfc='none',
                        mew=1.6, color=COL[ctrl], zorder=3,
                        label=ctrl if case == order[0] else None)
            else:
                ax.plot(want, got, MRK[ctrl], ms=4.2, mfc=COL[ctrl],
                        mew=0.0, color=COL[ctrl], zorder=4,
                        label=ctrl if case == order[0] else None)
    ax.set_xticks(np.arange(-135, 181, 45))
    ax.set_yticks(np.arange(-135, 181, 45))
    ax.set_xlabel('offset bearing [deg]', fontsize=9.5)
    ax.set_ylabel('uncompensated drift bearing [deg]', fontsize=9.5)
    ax.grid(alpha=0.45, lw=0.8, color='0.6')
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax.set_aspect('equal')
    fig.tight_layout()
    for ext, kw in (('pdf', {}), ('png', dict(dpi=a.dpi))):
        fig.savefig(a.outdir / f'fig_ff_compass.{ext}',
                    bbox_inches='tight', **kw)

    # ── (b) disturbance scaling ──────────────────────────────────────
    # crossbar style: at each distinct offset magnitude the median as
    # a short horizontal tick and the empirical 95% interval as a
    # capped vertical bar; cases sharing one magnitude pool their
    # flights (e.g. the four 28.3 mm cases contribute twelve flights)
    fig2, ax = plt.subplots(figsize=(5.4, 3.8))
    LIGHT = {'HGDO': '#74B4DC', 'L1': '#F0A868'}
    dodge = {'HGDO': -0.45, 'L1': +0.45}
    for ctrl in ctrls:
        for var, lab in (('wo_ff', 'uncompensated'),
                         ('w_ff', 'compensated')):
            col = COL[ctrl] if var == 'wo_ff' else LIGHT[ctrl]
            pool = defaultdict(list)
            for c in order:
                mag = round(float(np.hypot(*CASES[c][:2])), 2)
                pool[mag].extend(x['tilt']
                                 for x in cells[(c, ctrl, var)])
            mags = np.array(sorted(pool)) + dodge[ctrl]
            med = np.array([np.median(pool[m]) for m in sorted(pool)])
            lo = np.array([np.percentile(pool[m], 2.5)
                           for m in sorted(pool)])
            hi = np.array([np.percentile(pool[m], 97.5)
                           for m in sorted(pool)])
            ax.errorbar(mags, med, yerr=[med - lo, hi - med], fmt='_',
                        ms=10, mew=2.2, color=col, ecolor=col,
                        elinewidth=1.3, capsize=3.0, capthick=1.3,
                        ls='', label=f'{ctrl}, {lab}', zorder=3)
    ax.set_xlabel(r'$\|\mathbf{p}_{\mathrm{off}}\|$ [mm]', fontsize=9.5)
    ax.set_ylabel(r'$\vartheta_{\mathrm{peak}}$ [deg]', fontsize=9.5)
    ax.grid(alpha=0.45, lw=0.8, color='0.6')
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, framealpha=0.9)
    fig2.tight_layout()
    for ext, kw in (('pdf', {}), ('png', dict(dpi=a.dpi))):
        fig2.savefig(a.outdir / f'fig_ff_scaling.{ext}',
                     bbox_inches='tight', **kw)

    print(f'written to {a.outdir}/tab_ff_sim.tex, fig_ff_compass, '
          f'fig_ff_scaling, ff_sim_summary.csv')


if __name__ == '__main__':
    main()
