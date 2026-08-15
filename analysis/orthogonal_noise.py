#!/usr/bin/env python3
"""Measure the tip's broadband noise without using the cosh model at all.

The endpoint budget of (114) needs a noise term, and the obvious one --
the scatter of the measurement about the fitted curve -- is the residual
itself, so using it to explain the residual proves nothing.  The
pre-onset floor avoids that but is measured before the vehicle moves,
and the tip is exactly when the contact loads up and the rotors work
against a tilted airframe.

The orthogonal axis settles it.  During an Mx excitation the tip-over
response lives in omega_x; omega_y carries no cosh term, yet it is
recorded at the same instant, through the same rotors, the same contact
and the same estimator.  So its scatter measures the disturbance
environment with the model playing no part.

One caveat and its handling.  omega_y during an Mx run is not identically
zero -- the contact line is not exactly aligned with the axis, so some
real cross-axis motion appears.  That motion is smooth, whereas rotor
vibration and contact chatter are broadband, so two statistics are
reported and neither is the raw standard deviation:

  HF        std of the first difference over sqrt(2), which a smooth
            signal barely touches;
  detrended std about a cubic fit over the segment, which absorbs
            smooth cross-axis motion but keeps the wander.

Both are computed pre- and post-onset on the orthogonal axis, and the
excited axis's pre-onset values are printed alongside for scale.  The
comparison that matters is post- against pre-onset on the SAME channel:
that ratio is what the pre-onset floor understates by.

Inertia is not corrected for.  The same disturbance torque makes a
different rate on the two axes, so the orthogonal reading is used as an
estimate rather than a transfer -- and where it is used in a budget,
over-estimating is the safe direction.

Usage: python analysis/orthogonal_noise.py [out.csv]
"""
import contextlib
import collections
import csv
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import critical_value_getter_piecewise as cvp
from pnls_constants import PNLS_CONSTANTS
from utils.extractor import load_excitation_dataset
from constrained_calibration import ROOT

OTHER = {'x': 'y', 'y': 'x'}


def hf(v):
    """Broadband content, insensitive to any smooth component."""
    return float(np.std(np.diff(v)) / np.sqrt(2)) if len(v) > 4 else np.nan


def detrended(t, v, order=3):
    """Scatter about a cubic: smooth cross-axis motion is absorbed."""
    if len(v) < order + 4:
        return np.nan
    c = np.polyfit(t - t[0], v, order)
    return float(np.std(v - np.polyval(c, t - t[0])))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'orthogonal_noise.csv'
    rows = []
    for (case, ad), (c2, k) in sorted(PNLS_CONSTANTS.items()):
        axis = 'x' if ad == 'Mx' else 'y'
        with contextlib.redirect_stdout(io.StringIO()):
            bags = load_excitation_dataset(ROOT / case / ad)
        for bag in bags:
            rate = cvp.commanded_ramp_rate(bag.name)
            if rate is None:
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                sig = cvp.prepare_signals(bag, axis)
                sig_o = cvp.prepare_signals(bag, OTHER[axis])
            i0, i1 = cvp.detect_excitation_window(
                sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
            w = slice(i0, i1 + 1)
            t, om, M = sig['t'][w], sig['omega'][w], sig['moment'][w]
            om_o = sig_o['omega'][w]
            if len(om_o) != len(om):
                continue
            md = float(np.polyfit(t, M, 1)[0])
            pw = cvp.cosh_onset_fit(t, om, np.zeros_like(t), onset_guess=None,
                                    c2_fixed=c2, moment_floor=0.0,
                                    ramp_gain=k, ramp_rate=md)
            j = pw['onset_idx']
            if j < 12 or len(om) - j < 12:
                continue
            rows.append(dict(
                case=case, axis=ad, bag=bag.name, rate=rate,
                # excited axis, pre-onset: the floor the budget has used
                exc_pre_std=float(np.std(om[:j] - np.median(om[:j]))),
                exc_pre_hf=hf(om[:j]),
                # orthogonal axis: no cosh term anywhere in it
                orth_pre_hf=hf(om_o[:j]),
                orth_post_hf=hf(om_o[j:]),
                orth_pre_det=detrended(t[:j], om_o[:j]),
                orth_post_det=detrended(t[j:], om_o[j:]),
                orth_post_std=float(np.std(om_o[j:] - np.median(om_o[j:]))),
            ))
        print(f"  {case}/{ad}: {len(rows)} runs so far")

    with open(out, 'w', newline='') as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    g = collections.defaultdict(list)
    for r in rows:
        g[r['rate']].append(r)

    def med(rate, key):
        return np.median([r[key] for r in g[rate]])

    print(f"\n  {len(rows)} runs.  All figures rad/s.\n")
    print("  orthogonal axis -- the tip-over response is not in this channel\n")
    print(f"  {'rate':>6}{'n':>4}" + "".join(
        f"{h:>13}" for h in ('pre HF', 'post HF', 'ratio',
                             'pre detr', 'post detr', 'ratio')))
    rh, rd = [], []
    for rate in sorted(g):
        v = g[rate]
        a, b = med(rate, 'orth_pre_hf'), med(rate, 'orth_post_hf')
        c, d = med(rate, 'orth_pre_det'), med(rate, 'orth_post_det')
        rh.append(np.median([r['orth_post_hf'] / r['orth_pre_hf'] for r in v]))
        rd.append(np.median([r['orth_post_det'] / r['orth_pre_det']
                             for r in v]))
        print(f"  {rate:6.2f}{len(v):4d}{a:13.5f}{b:13.5f}{rh[-1]:13.2f}"
              f"{c:13.5f}{d:13.5f}{rd[-1]:13.2f}")
    print(f"\n  the tip raises the orthogonal channel by {np.median(rh):.2f}x"
          f" broadband and {np.median(rd):.2f}x detrended --")
    print(f"  measured with no cosh model involved.\n")

    print("  for scale, the excited axis before the onset"
          " (what the budget used)\n")
    print(f"  {'rate':>6}{'exc pre std':>14}{'exc pre HF':>13}"
          f"{'orth post detr':>16}{'ratio to exc pre std':>22}")
    for rate in sorted(g):
        a = med(rate, 'exc_pre_std')
        d = med(rate, 'orth_post_det')
        print(f"  {rate:6.2f}{a:14.5f}{med(rate, 'exc_pre_hf'):13.5f}"
              f"{d:16.5f}{d / a:22.2f}")
    print(f"\n  wrote {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
