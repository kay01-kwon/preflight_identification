#!/usr/bin/env python3
"""Hardware free-flight campaign: the manuscript artefacts.

Everything here excludes the pivot-based feedforward variant; the
comparison is uncompensated (wo_ff) against the pivot-free
feedforward (ff_pivot_free), per disturbance observer (HGDO / L1).
Metric definitions are exactly analysis/freeflight_metrics.py
(section VIII-A): window [t_lo, t_70], peaks of tilt, horizontal
rate, drift and speed.

Artefacts
---------
  exp_ff_bearing.png   uncompensated drift bearing against the
                       load-cell offset bearing, one point per
                       case/controller (circular mean of the repeats)
  exp_ff_peaks.png     the four peak metrics per case and pooled
                       ('all'): median tick + empirical 95% bar,
                       HGDO/L1 x uncompensated/compensated
  tab_ff_hw.tex        the same numbers as a table: per-case mean of
                       the repeats, pooled median in the 'all' rows

It also prints the in-flight toggle count per run: the flight logic
declares in-flight when the reconstructed collective thrust
f = C_T sum(rpm^2) exceeds the nominal 3.0 kg weight; the count is
the number of indicator transitions from the start of the record to
t_70, grouped by controller and variant (1 = one clean rising edge).

Usage
-----
  python analysis/freeflight_hw_summary.py [--data DIR]
         [--outdir DIR] [--dpi N] [--source {odom,pose}]
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.freeflight_metrics import (             # noqa: E402
    metrics, C_T, OFFSET, MASS)

G = 9.81
W_NOM = 3.0 * G                     # in-flight threshold of the logic
KEYS = ('tilt', 'rate', 'drift', 'speed')
VARS = ('wo_ff', 'ff_pivot_free')   # pivot-based excluded throughout
CTRLS = ('hgdo', 'l1')
CLAB = {'hgdo': 'HGDO', 'l1': 'L1'}
VLAB = {'wo_ff': 'uncompensated', 'ff_pivot_free': 'compensated'}
COL = {'hgdo': '#0072B2', 'l1': '#D55E00'}
LIGHT = {'hgdo': '#74B4DC', 'l1': '#F0A868'}


def toggles(path, t_70):
    """Transitions of the in-flight indicator up to t_70."""
    d = np.load(path)
    tr = d['rpm/t']
    f = C_T * np.sum(d['rpm/rpm'].astype(np.float64) ** 2, axis=1)
    ind = (f >= W_NOM)[tr <= t_70]
    return int(np.sum(ind[1:] != ind[:-1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('DataSet/free_flight'))
    p.add_argument('--outdir', type=Path, default=Path('docs'))
    p.add_argument('--dpi', type=int, default=600)
    p.add_argument('--source', choices=('odom', 'pose'), default='odom')
    a = p.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    runs = defaultdict(list)        # (case, ctrl, var) -> [metrics]
    tog = defaultdict(list)         # (ctrl, var) -> [count]
    for f in sorted(a.data.glob('*/*/*.npz')):
        case, ctrl, fn = f.relative_to(a.data).parts
        var = re.sub(r'(_\d+)?\.npz$', '', fn)
        if var not in VARS:
            continue
        m, why = metrics(f, case, a.source)
        if m is None:
            print(f'  skipped {f}: {why}')
            continue
        runs[(case, ctrl, var)].append(m)
        tog[(ctrl, var)].append((f'{case}/{ctrl}/{fn}',
                                 toggles(f, m['t_70'])))

    cases = sorted({k[0] for k in runs})

    # ── in-flight toggle count ───────────────────────────────────────
    print('in-flight toggles to t_70 (f = C_T sum rpm^2 >= 3.0 kg):')
    for ctrl in CTRLS:
        for var in VARS:
            v = np.array([n for _, n in tog[(ctrl, var)]])
            extra = [(name, n) for name, n in tog[(ctrl, var)] if n > 1]
            print(f'  {CLAB[ctrl]:5} {VLAB[var]:14} n={len(v):2d}  '
                  f'transitions median {np.median(v):.0f} max {v.max()}'
                  f'  runs with re-toggling: {len(extra)}')
            for name, n in extra:
                print(f'      {name}: {n} transitions')

    # ── bearing figure: uncompensated drift vs offset ────────────────
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.plot([-200, 200], [-200, 200], color='0.55', lw=1.0, ls='--',
            zorder=1)
    for case in cases:
        ox, oy = OFFSET[case]
        want = float(np.degrees(np.arctan2(oy, ox)))
        for ctrl in CTRLS:
            ang = np.radians([m['drift_dir']
                              for m in runs[(case, ctrl, 'wo_ff')]])
            got = float(np.degrees(np.arctan2(np.mean(np.sin(ang)),
                                              np.mean(np.cos(ang)))))
            if got - want > 180:
                got -= 360
            if want - got > 180:
                got += 360
            if ctrl == 'hgdo':
                ax.plot(want, got, 'o', ms=10, mfc='none', mew=1.6,
                        color=COL[ctrl], zorder=3,
                        label=CLAB[ctrl] if case == cases[0] else None)
            else:
                ax.plot(want, got, 's', ms=4.2, mfc=COL[ctrl], mew=0.0,
                        color=COL[ctrl], zorder=4,
                        label=CLAB[ctrl] if case == cases[0] else None)
    ax.set_xticks(np.arange(-135, 181, 45))
    ax.set_yticks(np.arange(-135, 181, 45))
    ax.set_xlabel('offset bearing [deg]', fontsize=9.5)
    ax.set_ylabel('uncompensated drift bearing [deg]', fontsize=9.5)
    ax.grid(alpha=0.45, lw=0.8, color='0.6')
    ax.set_axisbelow(True)
    ax.set_aspect('equal')
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(a.outdir / 'exp_ff_bearing.png', dpi=a.dpi,
                bbox_inches='tight')
    plt.close(fig)

    # ── peaks: per case and pooled, table + crossbar figure ──────────
    def pool(case, ctrl, var, key):
        if case == 'all':
            return np.array([m[key] for c in cases
                             for m in runs[(c, ctrl, var)]])
        return np.array([m[key] for m in runs[(case, ctrl, var)]])

    UNIT = {'tilt': (1.0, r'$\vartheta_{\mathrm{peak}}$ [deg]', '{:.1f}'),
            'rate': (1.0, r'$\Vert\omega_{xy}\Vert_{\mathrm{peak}}$'
                          r' [deg/s]', '{:.0f}'),
            'drift': (100.0, r'$\Vert p_{xy}\Vert_{\mathrm{peak}}$'
                             r' [cm]', '{:.1f}'),
            'speed': (1.0, r'$\Vert v_{xy}\Vert_{\mathrm{peak}}$'
                           r' [m/s]', '{:.2f}')}

    slots = cases + ['all']
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2), sharex=True)
    dodge = dict(hgdo={'wo_ff': -0.27, 'ff_pivot_free': -0.09},
                 l1={'wo_ff': +0.09, 'ff_pivot_free': +0.27})
    for ax, key in zip(axes.flat, KEYS):
        sc, ylab, _ = UNIT[key]
        for ctrl in CTRLS:
            for var in VARS:
                col = COL[ctrl] if var == 'wo_ff' else LIGHT[ctrl]
                lab = (f'{CLAB[ctrl]} {VLAB[var]}'
                       if ax is axes.flat[0] else None)
                for i, slot in enumerate(slots):
                    v = sc * pool(slot, ctrl, var, key)
                    x = i + dodge[ctrl][var]
                    lo, med, hi = np.percentile(v, [2.5, 50, 97.5])
                    ax.errorbar(x, med, yerr=[[med - lo], [hi - med]],
                                fmt='none', ecolor=col, elinewidth=1.3,
                                capsize=2.4, capthick=1.3, zorder=3)
                    ax.plot(x, med, '_', ms=9, mew=2.0, color=col,
                            zorder=4, label=lab if i == 0 else None)
        for i in range(1, len(slots)):
            ax.axvline(i - 0.5, color='0.88', lw=0.8, zorder=0)
        ax.set_xticks(range(len(slots)))
        ax.set_xticklabels([f'E{int(c)}' for c in cases] + ['all'],
                           fontsize=9)
        ax.set_ylabel(ylab, fontsize=9.5)
        ax.grid(axis='y', alpha=0.4, lw=0.8, color='0.6')
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4,
               fontsize=8.2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(a.outdir / 'exp_ff_peaks.png', dpi=a.dpi,
                bbox_inches='tight')
    plt.close(fig)

    # ── table ────────────────────────────────────────────────────────
    out = a.outdir / 'tab_ff_hw.tex'
    with open(out, 'w') as fh:
        fh.write(
            '% Hardware free-flight take-off peaks, uncompensated vs\n'
            '% pivot-free feedforward (pivot-based variant excluded).\n'
            '% Generated by analysis/freeflight_hw_summary.py.\n'
            '\\begin{table}[t]\n'
            '\\caption{Hardware take-off peaks over $[t_{\\mathrm{lo}},'
            ' t_{70}]$ per case and observer: uncompensated (unc.) '
            'against the pivot-free feedforward (comp.). Per-case '
            'entries are the mean of the three repeats; the all rows '
            'are pooled medians.}\n'
            '\\label{tab:ff_hw}\n'
            '\\centering\\small\n'
            '\\setlength{\\tabcolsep}{3.2pt}\n'
            '\\begin{tabular}{@{}llcccccccc@{}}\n'
            '\\toprule\n'
            ' & & \\multicolumn{2}{c}{$\\vartheta_{\\mathrm{peak}}$ '
            '[deg]} & \\multicolumn{2}{c}{rate [deg/s]} & '
            '\\multicolumn{2}{c}{drift [cm]} & '
            '\\multicolumn{2}{c}{speed [m/s]}\\\\\n'
            '\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}'
            '\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\n'
            'Case & DOB & unc. & comp. & unc. & comp. & unc. & comp.'
            ' & unc. & comp.\\\\\n\\midrule\n')

        def cells(slot, ctrl, stat):
            parts = []
            for key in KEYS:
                sc, _, fmt = UNIT[key]
                for var in VARS:
                    v = sc * pool(slot, ctrl, var, key)
                    parts.append('$' + fmt.format(stat(v)) + '$')
            return ' & '.join(parts)

        for case in cases:
            for ctrl in CTRLS:
                lead = f'E{int(case)}' if ctrl == CTRLS[0] else ''
                fh.write(f'{lead} & {CLAB[ctrl]} & '
                         + cells(case, ctrl, np.mean) + '\\\\\n')
        fh.write('\\midrule\n')
        for ctrl in CTRLS:
            lead = 'all' if ctrl == CTRLS[0] else ''
            fh.write(f'{lead} & {CLAB[ctrl]} & '
                     + cells('all', ctrl, np.median) + '\\\\\n')
        fh.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n')

    # improvement over the pooled runs, printed for the text
    for ctrl in CTRLS:
        for key in KEYS:
            u = np.median(pool('all', ctrl, 'wo_ff', key))
            c = np.median(pool('all', ctrl, 'ff_pivot_free', key))
            print(f'{CLAB[ctrl]:5} {key:6} pooled median '
                  f'{u:8.3f} -> {c:8.3f}  ({100 * (u - c) / u:+.0f}%)')
    print(f'written exp_ff_bearing.png, exp_ff_peaks.png, {out}')


if __name__ == '__main__':
    main()
