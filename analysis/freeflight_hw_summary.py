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

It also prints the in-flight toggle count per run, per the
ground-to-air switching logic of Algorithm 4: airborne when the
reconstructed collective thrust f = C_T sum(rpm^2) exceeds the
nominal 3.0 kg weight, held in flight while the latched flag and
z > z_th = 0.010 m persist; the count is the number of in_flight
transitions from the start of the record to t_70, grouped by
controller and variant (1 = one clean rising edge).

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
Z_TH = 0.010                        # m, altitude hold-in of Algorithm 4
KEYS = ('tilt', 'rate', 'drift', 'speed')
VARS = ('wo_ff', 'ff_pivot_free')   # pivot-based excluded throughout
CTRLS = ('hgdo', 'l1')
CLAB = {'hgdo': 'HGDO', 'l1': 'L1'}
VLAB = {'wo_ff': 'uncompensated', 'ff_pivot_free': 'compensated'}
COL = {'hgdo': '#0072B2', 'l1': '#D55E00'}
LIGHT = {'hgdo': '#74B4DC', 'l1': '#F0A868'}


def toggles(path, t_70, source):
    """Transitions of the in_flight flag of Algorithm 4 up to t_70.

    Per tick: airborne = (f_act >= W_nom) latches was_airborne;
    in_flight = airborne or (was_airborne and z > z_th); the latch
    resets only when in_flight evaluates false.  A clean take-off
    toggles exactly once (off -> on)."""
    d = np.load(path)
    tr = d['rpm/t']
    f = C_T * np.sum(d['rpm/rpm'].astype(np.float64) ** 2, axis=1)
    sel = tr <= t_70
    tr, f = tr[sel], f[sel]
    # altitude on the rpm timeline, relative to the pre-take-off level
    tz = d['odom/t']
    z = d[f'{source}/position'].astype(np.float64)[:, 2]
    pre = z[tz <= tr[np.argmax(f >= W_NOM)]] if np.any(f >= W_NOM) \
        else z[:10]
    z0 = float(np.median(pre)) if pre.size else float(z[0])
    zr = np.interp(tr, tz, z) - z0
    was, n, prev = False, 0, False
    for fi, zi in zip(f, zr):
        airborne = fi >= W_NOM
        was = was or airborne
        in_flight = airborne or (was and zi > Z_TH)
        if not in_flight:
            was = False
        n += in_flight != prev
        prev = in_flight
    return n


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
                                 toggles(f, m['t_70'], a.source)))

    cases = sorted({k[0] for k in runs})

    # ── in-flight toggle count ───────────────────────────────────────
    print('in_flight toggles to t_70 (Algorithm 4: f >= 3.0 kg '
          'latched, held while z > 0.010 m):')
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
            'rate': (1.0, r'$\omega_{\mathrm{peak}}$ [deg/s]', '{:.0f}'),
            'drift': (100.0, r'$p_{\mathrm{peak}}$ [cm]', '{:.1f}'),
            'speed': (1.0, r'$v_{\mathrm{peak}}$ [m/s]', '{:.2f}')}

    # one slot per observer/variant, the whole campaign pooled: the
    # per-case detail lives in the table's aggregation and the runs
    # CSV, the figure carries only the campaign-level contrast
    series = [(ctrl, var) for ctrl in CTRLS for var in VARS]
    ticklab = [f'{CLAB[c]}\n{"unc." if v == "wo_ff" else "comp."}'
               for c, v in series]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.6), sharex=True)
    for ax, key in zip(axes.flat, KEYS):
        sc, ylab, _ = UNIT[key]
        for i, (ctrl, var) in enumerate(series):
            col = COL[ctrl] if var == 'wo_ff' else LIGHT[ctrl]
            v = sc * pool('all', ctrl, var, key)
            lo, med, hi = np.percentile(v, [2.5, 50, 97.5])
            ax.errorbar(i, med, yerr=[[med - lo], [hi - med]],
                        fmt='none', ecolor=col, elinewidth=1.5,
                        capsize=3.2, capthick=1.5, zorder=3)
            ax.plot(i, med, '_', ms=13, mew=2.4, color=col, zorder=4)
        ax.axvline(1.5, color='0.88', lw=0.8, zorder=0)
        ax.set_xticks(range(len(series)))
        ax.set_xticklabels(ticklab, fontsize=8.5)
        ax.set_xlim(-0.6, len(series) - 0.4)
        ax.set_ylim(bottom=0)
        ax.set_ylabel(ylab, fontsize=9.5)
        ax.grid(axis='y', alpha=0.4, lw=0.8, color='0.6')
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.outdir / 'exp_ff_peaks.png', dpi=a.dpi,
                bbox_inches='tight')
    plt.close(fig)

    # ── table: campaign aggregate, the tab_ff_sim_agg layout ─────────
    # per-case means of the repeats first, then median and empirical
    # 95% interval over the five cases; improvement is the per-case
    # percentage aggregated the same way.
    ROW = (('tilt', r'$\vartheta_{\mathrm{peak}}$ [deg]', '{:.1f}'),
           ('rate', r'$\omega_{\mathrm{peak}}$ [deg/s]', '{:.0f}'),
           ('drift', r'$p_{\mathrm{peak}}$ [m]', '{:.3f}'),
           ('speed', r'$v_{\mathrm{peak}}$ [m/s]', '{:.2f}'))

    def agg(v):
        lo, med, hi = np.percentile(v, [2.5, 50, 97.5])
        return med, lo, hi

    out = a.outdir / 'tab_ff_hw.tex'
    with open(out, 'w') as fh:
        fh.write(
            '% Campaign-level aggregate of the hardware free-flight\n'
            '% take-off metrics: median and empirical 95% interval\n'
            '% over the 5 offset cases (per-case means, three\n'
            '% flights each); improvement is the per-case percentage\n'
            '% 100 (uncomp - comp)/uncomp aggregated the same way.\n'
            '% Pivot-based variant excluded.\n'
            '% Generated by analysis/freeflight_hw_summary.py.\n'
            '\\begin{tabular}{@{}l ccc@{}}\n'
            '\\toprule\n'
            'metric & Uncomp & Comp & improvement [\\%]\\\\\n'
            '\\midrule\n')
        for ctrl, blk in (('hgdo', r'\textbf{HGDO}'),
                          ('l1', r'\textbf{$\mathcal{L}_1$}')):
            fh.write(f'\\multicolumn{{4}}{{@{{}}l}}{{{blk}}}\\\\\n')
            for key, name, fmt in ROW:
                u = np.array([np.mean(pool(c, ctrl, 'wo_ff', key))
                              for c in cases])
                w = np.array([np.mean(pool(c, ctrl, 'ff_pivot_free',
                                           key)) for c in cases])
                imp = 100 * (u - w) / u
                cell = []
                for v, f_ in ((u, fmt), (w, fmt), (imp, '{:.0f}')):
                    med, lo, hi = agg(v)
                    cell.append(f'${f_.format(med)}$ $[{f_.format(lo)}'
                                f',\\,{f_.format(hi)}]$')
                fh.write(f'{name} & ' + ' & '.join(cell) + '\\\\\n')
                print(f'{CLAB[ctrl]:5} {key:6} '
                      f'unc {agg(u)[0]:8.3f} [{agg(u)[1]:.3f},'
                      f'{agg(u)[2]:.3f}]  comp {agg(w)[0]:8.3f}  '
                      f'improvement {agg(imp)[0]:.0f}% '
                      f'[{agg(imp)[1]:.0f},{agg(imp)[2]:.0f}]')
            if ctrl == 'hgdo':
                fh.write('\\addlinespace\n')
        fh.write('\\bottomrule\n\\end{tabular}\n')

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
