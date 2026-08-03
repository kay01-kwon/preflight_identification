#!/usr/bin/env python3
"""Ablation: median vs mean pre-onset baseline in the COSH onset sweep.

The onset sweep sets the pre-onset baseline C to the median of the
pre-segment.  A reviewer may ask (a) whether replacing the LS-optimal mean
by the median is legitimate and (b) whether it matters on the data.  This
script answers (b) directly: it reruns the full pipeline — including the
per-dataset (C2, K) calibration, which also uses the baseline — with
BASELINE_STAT = 'median' and 'mean', and reports the per-run onset-index
and M_crit differences plus the effect on the directional means, M_ff and
the ramp-invariance CV.

Usage:  PYTHONPATH=<stubs> python analysis/baseline_stat_ablation.py [outdir]
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

rows = []
for d in sorted(ROOT.glob('case_*/M[xy]')):
    axis = 'x' if d.name == 'Mx' else 'y'
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
    res = {}
    for stat in ('median', 'mean'):
        cvp.BASELINE_STAT = stat
        with contextlib.redirect_stdout(io.StringIO()):
            crits, _ = cvp.extract_piecewise_batch(bags, axis)
        res[stat] = {c.bag_name: c for c in crits}
    cvp.BASELINE_STAT = 'median'
    for name in sorted(set(res['median']) | set(res['mean'])):
        cm, cn = res['median'].get(name), res['mean'].get(name)
        rows.append(dict(
            case=d.parent.name, axis=d.name, bag=name,
            rate=cvp.commanded_ramp_rate(name),
            gated_median=cm is not None, gated_mean=cn is not None,
            onset_median=(cm.onset_idx if cm else ''),
            onset_mean=(cn.onset_idx if cn else ''),
            mcrit_median=(f"{cm.onset_moment:.4f}" if cm else ''),
            mcrit_mean=(f"{cn.onset_moment:.4f}" if cn else ''),
            d_onset=(cn.onset_idx - cm.onset_idx if cm and cn else ''),
            d_mcrit_mNm=(f"{1e3*(cn.onset_moment - cm.onset_moment):.2f}"
                         if cm and cn else ''),
        ))
    print(f"done {d.parent.name}/{d.name}", flush=True)

with open(OUT / 'baseline_stat_ablation.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

both = [r for r in rows if r['d_onset'] != '']
d_on = np.array([r['d_onset'] for r in both], float)
d_mc = np.array([float(r['d_mcrit_mNm']) for r in both], float)
print(f"\nruns compared: {len(both)} "
      f"(median-only {sum(1 for r in rows if r['gated_median'] and not r['gated_mean'])}, "
      f"mean-only {sum(1 for r in rows if r['gated_mean'] and not r['gated_median'])})")
print(f"onset-index shift  [samples]: identical {int(np.sum(d_on == 0))}"
      f"/{len(both)}, median |d| {np.median(np.abs(d_on)):.1f}, "
      f"max |d| {np.max(np.abs(d_on)):.0f}")
print(f"M_crit shift [mN·m]: median |d| {np.median(np.abs(d_mc)):.2f}, "
      f"p90 |d| {np.percentile(np.abs(d_mc), 90):.2f}, "
      f"max |d| {np.max(np.abs(d_mc)):.2f}")

print(f"\n{'case':8} {'ax':3} {'dir':4} {'mean_med':>9} {'mean_mean':>10} "
      f"{'d[mNm]':>7} {'CV_med':>7} {'CV_mean':>8}")
agg = defaultdict(lambda: defaultdict(dict))
for r in both:
    dirn = 'pos' if r['bag'].startswith('pos') else 'neg'
    agg[(r['case'], r['axis'])][dirn][r['bag']] = (
        float(r['mcrit_median']), float(r['mcrit_mean']))
mff_shift = []
for key in sorted(agg):
    means = {}
    for dirn in ('neg', 'pos'):
        v = np.array(list(agg[key][dirn].values()))
        if not len(v):
            continue
        m_med, m_mean = v[:, 0].mean(), v[:, 1].mean()
        cv_med = v[:, 0].std(ddof=1) / abs(m_med) if len(v) > 1 else 0.0
        cv_mean = v[:, 1].std(ddof=1) / abs(m_mean) if len(v) > 1 else 0.0
        means[dirn] = (m_med, m_mean)
        print(f"{key[0]:8} {key[1]:3} {dirn:4} {m_med:>9.4f} {m_mean:>10.4f} "
              f"{1e3*(m_mean-m_med):>7.2f} {cv_med:>7.4f} {cv_mean:>8.4f}")
    if len(means) == 2:
        ff_med = 0.5 * (means['pos'][0] + means['neg'][0])
        ff_mean = 0.5 * (means['pos'][1] + means['neg'][1])
        mff_shift.append(1e3 * (ff_mean - ff_med))
mff_shift = np.array(mff_shift)
print(f"\nM_ff shift [mN·m]: median |d| {np.median(np.abs(mff_shift)):.2f}, "
      f"max |d| {np.max(np.abs(mff_shift)):.2f}")
