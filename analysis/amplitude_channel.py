#!/usr/bin/env python3
"""The amplitude channel: what releasing C1 would have absorbed.

A cosh and a sinh of the same argument add to a cosh:

    a cosh(C2 tau) + b sinh(C2 tau) = R cosh(C2 tau + psi),
        R = sqrt(a^2 - b^2),   tanh psi = b / a.

Expanding the Duhamel integral of the deviation dynamics,

    e_omega(tau) = (1/J_P)[ cosh(C2 tau) P(tau) - sinh(C2 tau) Q(tau) ],
        P = int_0^tau cosh(C2 s) rho ds,   Q = int_0^tau sinh(C2 s) rho ds,

so wherever P and Q vary slowly the measured signal is again exactly a
cosh -- with the amplitude moved from C1 to C1 + P/J_P and the origin
shifted by psi/C2.  A fit with BOTH free would therefore absorb the
deviation entirely and leave no residual at all.  Pinning C1 = K Mdot is
what makes anything observable, which is a stronger reason for the
constrained fit than the amplitude-onset trade it was introduced for.

It also opens a channel.  Release C1, fit it per run, and the departure
from K Mdot measures the cosh-component P/(J_P C1) that the pinned fit
refuses to absorb.  Two limits on reading it:

  * the sinh-component -- the onset, hence Delta M_crit -- is absorbed
    either way, so this says nothing about the quantity (109) bounds;
  * the departure also carries the calibration error in K and the
    measurement error in Mdot, so it is an upper bound on P/(J_P C1),
    not a measurement of it.

A prediction worth checking against the unexplained residual: an
amplitude error leaves dC1 (cosh C2 tau - 1) behind, which GROWS across
the window.  The observed residual is flat, so this channel should turn
out small.

Usage: python analysis/amplitude_channel.py [out.csv]
"""
import contextlib
import collections
import csv
import io
import os
import sys

import numpy as np
from scipy import stats
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import critical_value_getter_piecewise as cvp
from pnls_constants import PNLS_CONSTANTS
from utils.extractor import load_excitation_dataset
from constrained_calibration import ROOT


def free_amplitude(t, om, c2, j0, c0):
    """Refit the amplitude alone, onset and C2 held where the pinned fit put
    them, so the comparison isolates the cosh direction."""
    tau = t[j0:] - t[j0]
    basis = np.cosh(np.clip(c2 * tau, 0, 30)) - 1.0
    y = om[j0:] - c0
    denom = float(basis @ basis)
    return float(basis @ y) / denom if denom > 0 else np.nan


def free_both(t, om, c2, j0, c0, c1_guess):
    """Amplitude and onset together, which is what a per-run NLS would do."""
    def resid(p):
        a, dt = p
        tau = np.clip(t[j0:] - t[j0] - dt, 0.0, None)
        return om[j0:] - (c0 + a * (np.cosh(np.clip(c2 * tau, 0, 30)) - 1.0))
    try:
        s = least_squares(resid, [c1_guess, 0.0],
                          bounds=([-np.inf, -0.15], [np.inf, 0.15]))
        return float(s.x[0]), float(s.x[1])
    except Exception:
        return np.nan, np.nan


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'amplitude_channel.csv'
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
            i0, i1 = cvp.detect_excitation_window(
                sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
            w = slice(i0, i1 + 1)
            t, om, M = sig['t'][w], sig['omega'][w], sig['moment'][w]
            md = float(np.polyfit(t, M, 1)[0])
            pw = cvp.cosh_onset_fit(t, om, np.zeros_like(t), onset_guess=None,
                                    c2_fixed=c2, moment_floor=0.0,
                                    ramp_gain=k, ramp_rate=md)
            j = pw['onset_idx']
            if j < 12 or len(om) - j < 12:
                continue
            c1_pin = k * md
            c0 = float(pw['c'])
            sgn = np.sign(c1_pin) if c1_pin != 0 else 1.0
            a_free = free_amplitude(t, om, c2, j, c0)
            a_both, dt_both = free_both(t, om, c2, j, c0, c1_pin)
            rows.append(dict(
                case=case, axis=ad, bag=bag.name, rate=rate,
                c1_pinned=c1_pin, c1_free=a_free, c1_both=a_both,
                dt_both=dt_both,
                rel_free=(a_free - c1_pin) / abs(c1_pin),
                rel_both=(a_both - c1_pin) / abs(c1_pin),
                x=float(c2 * (t[-1] - t[j]))))
        print(f"  {case}/{ad} done ({len(rows)} runs)")

    with open(out, 'w', newline='') as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    g = collections.defaultdict(list)
    for r in rows:
        g[r['rate']].append(r)

    print(f"\n  {len(rows)} runs.  Amplitude released, onset held where the"
          f"\n  pinned fit put it: (C1_free - K Mdot) / |K Mdot|.\n")
    print(f"  {'rate':>6}{'n':>4}{'median':>10}{'p10':>9}{'p90':>9}"
          f"{'|median|':>11}   and with the onset released too")
    print(f"  {'':6}{'':4}{'':10}{'':9}{'':9}{'':11}{'median':>10}{'p90 |dt| [ms]':>16}")
    for rate in sorted(g):
        v = g[rate]
        rf = np.array([r['rel_free'] for r in v])
        rb = np.array([r['rel_both'] for r in v])
        dt = np.array([abs(r['dt_both']) for r in v])
        print(f"  {rate:6.2f}{len(v):4d}{100 * np.median(rf):9.2f}%"
              f"{100 * np.percentile(rf, 10):8.2f}%"
              f"{100 * np.percentile(rf, 90):8.2f}%"
              f"{100 * np.median(np.abs(rf)):10.2f}%"
              f"{100 * np.median(rb):9.2f}%{1e3 * np.percentile(dt, 90):15.1f}")

    rf = np.array([r['rel_free'] for r in rows])
    rate = np.array([r['rate'] for r in rows])
    sp = stats.spearmanr(rate, np.abs(rf))
    print(f"\n  |departure| over all runs: median {100 * np.median(np.abs(rf)):.2f}%,"
          f" p90 {100 * np.percentile(np.abs(rf), 90):.2f}%,"
          f" max {100 * np.max(np.abs(rf)):.2f}%")
    print(f"  Spearman(ramp rate, |departure|) = {sp[0]:+.3f} (p = {sp[1]:.4f})")
    print(f"\n  Read as an upper bound on the cosh-component of the accumulated")
    print(f"  disturbance, since the calibration error in K and the measurement")
    print(f"  error in Mdot are folded into the same number.  The ramp-rate")
    print(f"  gate alone admits 3%.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
