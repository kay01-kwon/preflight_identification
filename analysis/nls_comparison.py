#!/usr/bin/env python3
"""Dataset-calibrated COSH vs per-run nonlinear least squares (TRF).

Both estimators see identical inputs — same signals, excitation window,
allocator caps and onset-free gates (the gated run set is method-independent
because the gates use the moment trace only).  They differ only in what is
fitted per run:

  A. COSH (reported method): (C2, K) calibrated once per dataset by
     ramp-rate invariance; per run the amplitude is pinned (C1 = K Mdot),
     the baseline is the pre-segment median, and the single unknown — the
     onset index — is swept exhaustively.
  B. Nonlinear LS: per-run scipy least_squares (TRF) fit of (C1, C2, C)
     with C2 bounded to the physical band [3, 8] rad/s, onset swept locally
     around the quadratic-model seed (the pipeline's historical mode).

Reported per case/axis and method: directional means of M_crit, the
pivot-free offset M_ff = (mean+ + mean-)/2, and the CoM offset
lambda = M_ff / W.

Usage: PYTHONPATH=<stubs> python analysis/nls_comparison.py [outdir]
"""
import contextlib
import csv
import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset

ROOT = Path(__file__).resolve().parents[1] / 'DataSet' / 'exp'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
W_N = 31.59  # vehicle weight [N]

rows = []
for d in sorted(ROOT.glob('case_*/M[xy]')):
    axis = 'x' if d.name == 'Mx' else 'y'
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
        crits_a, _ = cvp.extract_piecewise_batch(bags, axis)   # A: COSH
    gated = {c.bag_name for c in crits_a}
    by_bag = {b.name: b for b in bags}
    res = {'cosh': {c.bag_name: c.onset_moment for c in crits_a}, 'nls': {}}
    for name in sorted(gated):                                  # B: TRF
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                crit, _ = cvp.extract_piecewise(
                    by_bag[name], axis, model='cosh',
                    cosh_c2=None, ramp_gain=None)
            res['nls'][name] = crit.onset_moment
        except Exception as e:
            print(f"  NLS failed on {name}: {e}", flush=True)
    for name in sorted(gated):
        rows.append(dict(case=d.parent.name, axis=d.name, bag=name,
                         rate=cvp.commanded_ramp_rate(name),
                         dir='pos' if name.startswith('pos') else 'neg',
                         mcrit_cosh=f"{res['cosh'][name]:.4f}",
                         mcrit_nls=(f"{res['nls'][name]:.4f}"
                                    if name in res['nls'] else '')))
    print(f"done {d.parent.name}/{d.name}", flush=True)

with open(OUT / 'nls_comparison_runs.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"\n{'case':8} {'ax':3} {'method':6} {'mean_neg':>9} {'mean_pos':>9} "
      f"{'CV_neg':>7} {'CV_pos':>7} {'M_ff':>8} {'offset_mm':>9}")
agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for r in rows:
    for m in ('cosh', 'nls'):
        if r[f'mcrit_{m}']:
            agg[(r['case'], r['axis'])][m][r['dir']].append(
                float(r[f'mcrit_{m}']))
summary = []
for key in sorted(agg):
    for m in ('cosh', 'nls'):
        g = agg[key][m]
        if 'pos' not in g or 'neg' not in g:
            continue
        mn, mp = np.mean(g['neg']), np.mean(g['pos'])
        cvn = np.std(g['neg'], ddof=1) / abs(mn)
        cvp_ = np.std(g['pos'], ddof=1) / abs(mp)
        mff = 0.5 * (mp + mn)
        off = 1e3 * mff / W_N
        summary.append(dict(case=key[0], axis=key[1], method=m,
                            mean_neg=f"{mn:.4f}", mean_pos=f"{mp:.4f}",
                            cv_neg=f"{cvn:.4f}", cv_pos=f"{cvp_:.4f}",
                            M_ff=f"{mff:+.4f}", offset_mm=f"{off:+.3f}"))
        print(f"{key[0]:8} {key[1]:3} {m:6} {mn:>9.4f} {mp:>9.4f} "
              f"{cvn:>7.4f} {cvp_:>7.4f} {mff:>+8.4f} {off:>+9.3f}")

with open(OUT / 'nls_comparison_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)

# paired per-run deltas and per-direction CV comparison
both = [r for r in rows if r['mcrit_nls']]
dm = np.array([1e3 * (float(r['mcrit_nls']) - float(r['mcrit_cosh']))
               for r in both])
print(f"\nper-run |M_crit(NLS) - M_crit(COSH)| [mN·m]: "
      f"median {np.median(np.abs(dm)):.1f}, p90 "
      f"{np.percentile(np.abs(dm), 90):.1f}, max {np.max(np.abs(dm)):.1f} "
      f"(n={len(both)})")
cv_pairs = defaultdict(dict)
for s in summary:
    cv_pairs[(s['case'], s['axis'], 'neg')][s['method']] = float(s['cv_neg'])
    cv_pairs[(s['case'], s['axis'], 'pos')][s['method']] = float(s['cv_pos'])
wins = sum(1 for v in cv_pairs.values()
           if 'cosh' in v and 'nls' in v and v['cosh'] < v['nls'])
tot = sum(1 for v in cv_pairs.values() if 'cosh' in v and 'nls' in v)
cc = [v['cosh'] for v in cv_pairs.values() if 'cosh' in v and 'nls' in v]
cn = [v['nls'] for v in cv_pairs.values() if 'cosh' in v and 'nls' in v]
print(f"directional CV: COSH lower in {wins}/{tot}; "
      f"median CV COSH {np.median(cc):.4f} vs NLS {np.median(cn):.4f}")
