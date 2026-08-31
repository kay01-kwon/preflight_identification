#!/usr/bin/env python3
"""Take-off switch: chattering and threshold sensitivity.

The ground-to-air switch -- balanced moment on the ground, DOB
compensation in flight -- is decided by the latched thrust-weight
comparison alone: airborne when the reconstructed collective thrust
f_act = C_T sum(rpm^2) reaches the nominal weight W_nom.  The
altitude hold-in z_th exists to disambiguate the LANDING criterion
and does not enter the take-off branch, so the only design constant
to sweep here is the thrust threshold.

Two things are measured over the hardware free-flight campaign,
counting switch transitions up to t_70:

  * the bare comparison with no latch -- the behaviour the latch
    exists to prevent;
  * the deployed latched switch over a range of thresholds, together
    with the peak thrust margin that sets where it breaks.

A clean take-off switches exactly once (off -> on).  More than once
is chattering; zero means the switch never fired at all.

The pivot-based feedforward variant is excluded throughout, as
everywhere else in the free-flight evaluation.

Usage
-----
  python analysis/switching_sensitivity.py [--data DIR] [--outdir DIR]
         [--source {odom,pose}]

Writes docs/tab_switching_sensitivity.tex and prints the sweep.
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.freeflight_metrics import metrics, C_T          # noqa: E402

G = 9.81
W_NOM = 3.0 * G
Z_TH = 0.010                      # m, deployed value
VARS = ('wo_ff', 'ff_pivot_free')
CTRLS = ('hgdo', 'l1')
# sweeps: altitude hold-in [m], and thrust threshold as a fraction of
# W_nom (1.00 is the deployed rule f_act >= W_nom)
Z_SWEEP = (0.0, 0.005, 0.010, 0.020, 0.030, 0.050, 0.100)
F_SWEEP = (0.90, 0.95, 1.00, 1.05, 1.10)


def load(data, source):
    """Per run: the rpm-timeline thrust, relative altitude and t_70."""
    runs = []
    for f in sorted(data.glob('*/*/*.npz')):
        case, ctrl, fn = f.relative_to(data).parts
        var = re.sub(r'(_\d+)?\.npz$', '', fn)
        if var not in VARS:
            continue
        m, why = metrics(f, case, source)
        if m is None:
            print(f'  skipped {f}: {why}')
            continue
        d = np.load(f)
        tr = d['rpm/t']
        fa = C_T * np.sum(d['rpm/rpm'].astype(np.float64) ** 2, axis=1)
        sel = tr <= m['t_70']
        tr, fa = tr[sel], fa[sel]
        tz = d['odom/t']
        z = d[f'{source}/position'].astype(np.float64)[:, 2]
        hit = np.flatnonzero(fa >= W_NOM)
        pre = z[tz <= tr[hit[0]]] if hit.size else z[:10]
        z0 = float(np.median(pre)) if pre.size else float(z[0])
        runs.append(dict(case=case, ctrl=ctrl, var=var, f=fa,
                         z=np.interp(tr, tz, z) - z0))
    return runs


def toggles(run, w_thr, z_th, latch=True):
    """in_flight transitions under (w_thr, z_th).

    latch=True is Algorithm 4; latch=False is the bare thrust
    indicator (f_act >= w_thr) with no latch and no altitude hold-in,
    i.e. the behaviour the hysteresis is there to prevent."""
    was, n, prev = False, 0, False
    for fi, zi in zip(run['f'], run['z']):
        airborne = fi >= w_thr
        if not latch:
            n += airborne != prev
            prev = airborne
            continue
        was = was or airborne
        in_flight = airborne or (was and zi > z_th)
        if not in_flight:
            was = False
        n += in_flight != prev
        prev = in_flight
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path,
                    default=Path('DataSet/free_flight'))
    ap.add_argument('--outdir', type=Path, default=Path('docs'))
    ap.add_argument('--source', choices=('odom', 'pose'), default='odom')
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    runs = load(a.data, a.source)
    print(f'{len(runs)} runs (pivot-based excluded)\n')

    def sweep(label, pairs, latch=True):
        rows = []
        print(f'{label:>26}   clean   never-triggered   chattering '
              f'(max)')
        for tag, w_thr, z_th in pairs:
            n = [toggles(r, w_thr, z_th, latch) for r in runs]
            clean = sum(1 for v in n if v == 1)
            never = sum(1 for v in n if v == 0)
            chat = [(r['case'], v) for r, v in zip(runs, n) if v > 1]
            rows.append((tag, clean, len(n), never, len(chat),
                         max(n) if n else 0))
            cs = sorted({c for c, _ in chat})
            print(f'{tag:>26}   {clean:3d}/{len(n):<3d}   {never:3d}'
                  f'               {len(chat):3d} '
                  f'({",".join(cs) if cs else "-"}; max {max(n)})')
        print()
        return rows

    # what the take-off switch would do WITHOUT the latch -- the
    # behaviour the latch exists to prevent
    brows = sweep('bare f >= W_nom, no latch',
                  [('no latch', W_NOM, 0.0)], latch=False)
    # the deployed switch: thrust-weight comparison, latched. z_th does
    # NOT enter the take-off branch, so it is not swept here.
    drows = sweep('deployed (latched) / W_nom',
                  [(f'{c:.2f}', c * W_NOM, 0.0) for c in F_SWEEP])
    # thrust margin at the crossing, the quantity the threshold choice
    # actually trades against
    marg = []
    for r in runs:
        hit = np.flatnonzero(r['f'] >= W_NOM)
        if hit.size:
            marg.append(float(r['f'][hit[0]:].max() / W_NOM))
    marg = np.array(marg)
    print(f'peak thrust / W_nom after the crossing: min {marg.min():.3f}'
          f'  median {np.median(marg):.3f}  max {marg.max():.3f}\n')
    zrows, frows = [], drows

    out = a.outdir / 'tab_switching_sensitivity.tex'
    with open(out, 'w') as fh:
        fh.write(
            '% Take-off switch: chattering and threshold sensitivity.\n'
            '% Generated by analysis/switching_sensitivity.py.\n'
            '\\begin{table}[t]\n'
            '\\caption{Ground-to-air switch over the free-flight '
            f'campaign ({len(runs)} take-offs). The switch is the '
            'latched thrust--weight comparison alone; the altitude '
            'hold-in $z_{\\mathrm{th}}$ does not enter the take-off '
            'branch. Columns count the runs that switch exactly once '
            '(clean), never switch, and switch more than once '
            '(chattering, worst count in parentheses). Without the '
            'latch the same comparison chatters; with it the switch '
            'is clean for every threshold up to '
            '$1.05\\,W_{\\mathrm{nom}}$.}\n'
            '\\label{tab:switching_sensitivity}\n'
            '\\centering\\small\n'
            '\\begin{tabular}{@{}lccc@{}}\n\\toprule\n'
            'setting & clean & never & chattering (max)\\\\\n'
            '\\midrule\n'
            '\\multicolumn{4}{@{}l}{bare $f_{\\mathrm{act}} \\ge '
            'W_{\\mathrm{nom}}$, no latch}\\\\\n')
        for tag, clean, tot, never, nchat, mx in brows:
            fh.write(f'\\quad {tag} & {clean}/{tot} & {never} & '
                     f'{nchat} ({mx})\\\\\n')
        fh.write('\\addlinespace\n'
                 '\\multicolumn{4}{@{}l}{deployed (latched), '
                 'threshold / $W_{\\mathrm{nom}}$}\\\\\n')
        for tag, clean, tot, never, nchat, mx in drows:
            star = '$^{\\ast}$' if tag == '1.00' else ''
            fh.write(f'\\quad {tag}{star} & {clean}/{tot} & {never} & '
                     f'{nchat} ({mx})\\\\\n')
        fh.write('\\bottomrule\n\\end{tabular}\n'
                 '\\\\[2pt]\\footnotesize $^{\\ast}$deployed value.\n'
                 '\\end{table}\n')
    print(f'written to {out}')


if __name__ == '__main__':
    main()
