#!/usr/bin/env python3
"""Ablation: pre-onset baseline statistic in the COSH onset sweep.

The onset sweep sets the pre-onset baseline C to the median of the
pre-segment.  Two natural challenges: (a) the mean is the LS-optimal
constant — does the median change anything?  (b) the onset conditions say
the true angular rate is zero before the onset — why estimate C at all
instead of pinning C = 0?  This script answers both empirically: it reruns
the full pipeline — including the per-dataset (C2, K) calibration, which
also uses the baseline — with BASELINE_STAT = 'median' and an alternative
('mean' or 'zero'), reports per-run onset-index and M_crit differences and
the effect on directional means, M_ff and the ramp-invariance CV, and
quantifies how far the measured pre-onset baseline actually sits from zero
(gyro bias + residual pre-excitation motion).

Usage:
  PYTHONPATH=<stubs> python analysis/baseline_stat_ablation.py [outdir] [alt]
with alt in {mean, zero} (default mean).
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
ALT = sys.argv[2] if len(sys.argv) > 2 else 'mean'
assert ALT in ('mean', 'zero')

rows = []
base_mrads = []
for d in sorted(ROOT.glob('case_*/M[xy]')):
    axis = 'x' if d.name == 'Mx' else 'y'
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
    res = {}
    for stat in ('median', ALT):
        cvp.BASELINE_STAT = stat
        with contextlib.redirect_stdout(io.StringIO()):
            crits, _ = cvp.extract_piecewise_batch(bags, axis)
        res[stat] = {c.bag_name: c for c in crits}
    cvp.BASELINE_STAT = 'median'
    for name in sorted(set(res['median']) | set(res[ALT])):
        cm, cn = res['median'].get(name), res[ALT].get(name)
        # measured pre-onset baseline (median-run onset, window-start onward)
        if cm is not None:
            i0, _ = cvp.detect_excitation_window(
                cm.moment, moment_cap=cvp.MOMENT_CAP.get(axis))
            if cm.onset_idx > i0 + 1:
                base_mrads.append(
                    1e3 * float(np.median(cm.omega[i0:cm.onset_idx])))
        rows.append(dict(
            case=d.parent.name, axis=d.name, bag=name,
            rate=cvp.commanded_ramp_rate(name),
            gated_median=cm is not None, **{f'gated_{ALT}': cn is not None},
            onset_median=(cm.onset_idx if cm else ''),
            **{f'onset_{ALT}': cn.onset_idx if cn else ''},
            mcrit_median=(f"{cm.onset_moment:.4f}" if cm else ''),
            **{f'mcrit_{ALT}': f"{cn.onset_moment:.4f}" if cn else ''},
            d_onset=(cn.onset_idx - cm.onset_idx if cm and cn else ''),
            d_mcrit_mNm=(f"{1e3*(cn.onset_moment - cm.onset_moment):.2f}"
                         if cm and cn else ''),
        ))
    print(f"done {d.parent.name}/{d.name}", flush=True)

with open(OUT / f'baseline_stat_ablation_{ALT}.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

b = np.abs(np.array(base_mrads))
print(f"\nmeasured pre-onset baseline |C| [mrad/s]: "
      f"median {np.median(b):.1f}, p90 {np.percentile(b, 90):.1f}, "
      f"max {np.max(b):.1f}  (n={len(b)})")

both = [r for r in rows if r['d_onset'] != '']
d_on = np.array([r['d_onset'] for r in both], float)
d_mc = np.array([float(r['d_mcrit_mNm']) for r in both], float)
print(f"\nalt = {ALT};  runs compared: {len(both)} "
      f"(median-only {sum(1 for r in rows if r['gated_median'] and not r[f'gated_{ALT}'])}, "
      f"{ALT}-only {sum(1 for r in rows if r[f'gated_{ALT}'] and not r['gated_median'])})")
print(f"onset-index shift  [samples]: identical {int(np.sum(d_on == 0))}"
      f"/{len(both)}, median |d| {np.median(np.abs(d_on)):.1f}, "
      f"max |d| {np.max(np.abs(d_on)):.0f}")
print(f"M_crit shift [mN·m]: median |d| {np.median(np.abs(d_mc)):.2f}, "
      f"p90 |d| {np.percentile(np.abs(d_mc), 90):.2f}, "
      f"max |d| {np.max(np.abs(d_mc)):.2f}")

print(f"\n{'case':8} {'ax':3} {'dir':4} {'mean_med':>9} {'mean_'+ALT:>10} "
      f"{'d[mNm]':>7} {'CV_med':>7} {'CV_'+ALT:>8}")
agg = defaultdict(lambda: defaultdict(dict))
for r in both:
    dirn = 'pos' if r['bag'].startswith('pos') else 'neg'
    agg[(r['case'], r['axis'])][dirn][r['bag']] = (
        float(r['mcrit_median']), float(r[f'mcrit_{ALT}']))
mff_shift = []
for key in sorted(agg):
    means = {}
    for dirn in ('neg', 'pos'):
        v = np.array(list(agg[key][dirn].values()))
        if not len(v):
            continue
        m_med, m_alt = v[:, 0].mean(), v[:, 1].mean()
        cv_med = v[:, 0].std(ddof=1) / abs(m_med) if len(v) > 1 else 0.0
        cv_alt = v[:, 1].std(ddof=1) / abs(m_alt) if len(v) > 1 else 0.0
        means[dirn] = (m_med, m_alt)
        print(f"{key[0]:8} {key[1]:3} {dirn:4} {m_med:>9.4f} {m_alt:>10.4f} "
              f"{1e3*(m_alt-m_med):>7.2f} {cv_med:>7.4f} {cv_alt:>8.4f}")
    if len(means) == 2:
        ff_med = 0.5 * (means['pos'][0] + means['neg'][0])
        ff_alt = 0.5 * (means['pos'][1] + means['neg'][1])
        mff_shift.append(1e3 * (ff_alt - ff_med))
mff_shift = np.array(mff_shift)
print(f"\nM_ff shift [mN·m]: median |d| {np.median(np.abs(mff_shift)):.2f}, "
      f"max |d| {np.max(np.abs(mff_shift)):.2f}")
