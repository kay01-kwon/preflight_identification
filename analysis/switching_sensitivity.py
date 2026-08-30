#!/usr/bin/env python3
"""Anti-chattering sensitivity of the ground-to-air switching logic.

Algorithm 4 declares the vehicle airborne when the reconstructed
collective thrust f_act = C_T sum(rpm^2) reaches the nominal weight
W_nom, latches that flag, and holds in_flight while z > z_th.  Two
design constants therefore govern the switch: the thrust threshold
(here expressed as a fraction of W_nom) and the altitude hold-in
z_th.  This script sweeps each of them over the hardware free-flight
campaign and counts in_flight transitions up to t_70, so the
anti-chattering claim is supported by a parameter sensitivity study
rather than by the single deployed setting.

A clean take-off gives exactly one transition (off -> on).  More than
one means the flag dropped and re-armed, i.e. chattering.

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
        print(f'{label:>26}   clean/total   max toggles   chattering runs')
        for tag, w_thr, z_th in pairs:
            n = [toggles(r, w_thr, z_th, latch) for r in runs]
            bad = sorted({r['case'] for r, v in zip(runs, n) if v > 1})
            clean = sum(1 for v in n if v == 1)
            rows.append((tag, clean, len(n), max(n)))
            print(f'{tag:>26}   {clean:3d}/{len(n):<3d}       '
                  f'{max(n):3d}          '
                  f'{len(n) - clean} ({",".join(bad) if bad else "-"})')
        print()
        return rows

    # the baseline the hysteresis exists to prevent
    brows = sweep('bare indicator (no latch)',
                  [('f >= W_nom', W_NOM, 0.0)], latch=False)
    zrows = sweep('altitude hold-in z_th [m]',
                  [(f'{z:.3f}', W_NOM, z) for z in Z_SWEEP])
    frows = sweep('thrust threshold / W_nom',
                  [(f'{c:.2f}', c * W_NOM, Z_TH) for c in F_SWEEP])

    out = a.outdir / 'tab_switching_sensitivity.tex'
    with open(out, 'w') as fh:
        fh.write(
            '% Anti-chattering sensitivity of the Algorithm-4 switch.\n'
            '% Generated by analysis/switching_sensitivity.py.\n'
            '\\begin{table}[t]\n'
            '\\caption{Sensitivity of the ground-to-air switch of '
            'Algorithm~4 over the free-flight campaign '
            f'({len(runs)} take-offs). Entries count the runs whose '
            '\\texttt{in\\_flight} flag makes exactly one transition '
            '(a clean take-off) and the worst-case transition count '
            'over all runs. The first block is the bare thrust '
            'indicator, which chatters; the latch of Algorithm~4 '
            'removes it, and does so over a wide range of both design '
            'constants. The deployed setting is '
            '$z_{\\mathrm{th}} = 0.010$~m with the threshold at '
            '$W_{\\mathrm{nom}}$.}\n'
            '\\label{tab:switching_sensitivity}\n'
            '\\centering\\small\n'
            '\\begin{tabular}{@{}lcc@{}}\n\\toprule\n'
            'setting & clean take-offs & worst count\\\\\n\\midrule\n'
            '\\multicolumn{3}{@{}l}{bare thrust indicator '
            '$f_{\\mathrm{act}} \\ge W_{\\mathrm{nom}}$ '
            '(no latch, no hold-in)}\\\\\n')
        for tag, clean, tot, mx in brows:
            fh.write(f'\\quad {tag} & {clean}/{tot} & {mx}\\\\\n')
        fh.write(
            '\\addlinespace\n'
            f'\\multicolumn{{3}}{{@{{}}l}}{{Algorithm~4, altitude '
            f'hold-in $z_{{\\mathrm{{th}}}}$ [m]}}\\\\\n')
        for tag, clean, tot, mx in zrows:
            star = '$^{\\ast}$' if tag == '0.010' else ''
            fh.write(f'\\quad {tag}{star} & {clean}/{tot} & {mx}\\\\\n')
        fh.write('\\addlinespace\n'
                 '\\multicolumn{3}{@{}l}{thrust threshold / '
                 '$W_{\\mathrm{nom}}$, $z_{\\mathrm{th}} = 0.010$~m}'
                 '\\\\\n')
        for tag, clean, tot, mx in frows:
            star = '$^{\\ast}$' if tag == '1.00' else ''
            fh.write(f'\\quad {tag}{star} & {clean}/{tot} & {mx}\\\\\n')
        fh.write('\\bottomrule\n\\end{tabular}\n'
                 '\\\\[2pt]\\footnotesize $^{\\ast}$deployed value.\n'
                 '\\end{table}\n')
    print(f'written to {out}')


if __name__ == '__main__':
    main()
