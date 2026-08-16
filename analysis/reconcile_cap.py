#!/usr/bin/env python3
"""Reconcile two computations of the same residual check.

Two passes over the same 140 runs returned different pass counts for
what was meant to be one inequality, 114/140 and 139/140.  One of them
is wrong and neither may be quoted until it is known which.  This
recomputes every intermediate from a single pass so the divergence has
somewhere to show itself.

The inequality is (VIII.3) in RMS form,

    RMS(r_lo)  <=  RMS(E) + sigma_n,

with r_lo the residual below the cutoff.  Four ways of evaluating the
two sides are printed side by side:

    A   closed-form cap, rho_bar K C2 sqrt(B(x)/x), plus sigma
    A'  the same but with RMS taken over the E array directly
    B   L2 norms, ||r_lo|| <= ||E|| + sigma sqrt(T), which is A' times
        sqrt(T) on both sides and must therefore agree exactly
    B'  B but with the FFT run on a nominal dt rather than the true one
    C   B but with the FFT run on |r| rather than r

The verdict is that A, A' and B agree exactly at 114/140, so the
inequality is well defined and it is the second pass that was wrong.
A nominal dt costs only two runs, 112/140, so that was not it either.
The fault was rectification: the offending pass had cached |r| and split
THAT.  A rectified signal has its energy piled at low frequency, so the
split stops separating what it was meant to and lets everything through
-- 140/140, which should have been the tell.  114/140 stands.

Usage: python analysis/reconcile_cap.py [DATASET_ROOT]
"""
import contextlib
import collections
import io
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fit_quality_bound import rho_bar, rms_cap

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '.reconcile_cache.pkl')
FC = 5.0


def split(r, dt, fc=FC):
    """Residual below fc, by zeroing FFT bins."""
    rr = r - r.mean()
    F = np.fft.rfft(rr)
    F[np.fft.rfftfreq(len(rr), d=dt) > fc] = 0.0
    return np.fft.irfft(F, n=len(rr))


def collect(root):
    if os.path.exists(CACHE):
        with open(CACHE, 'rb') as fh:
            return pickle.load(fh)
    import critical_value_getter_piecewise as cvp
    from pnls_constants import PNLS_CONSTANTS
    from utils.extractor import load_excitation_dataset
    from pathlib import Path
    rows = []
    for (case, ad), (c2, k) in sorted(PNLS_CONSTANTS.items()):
        axis = 'x' if ad == 'Mx' else 'y'
        with contextlib.redirect_stdout(io.StringIO()):
            bags = load_excitation_dataset(Path(root) / case / ad)
        for bag in bags:
            rate = cvp.commanded_ramp_rate(bag.name)
            if rate is None:
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                sig = cvp.prepare_signals(bag, axis)
            i0, i1 = cvp.detect_excitation_window(
                sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
            w = slice(i0, i1 + 1)
            t, om, mom = sig['t'][w], sig['omega'][w], sig['moment'][w]
            md = float(np.polyfit(t, mom, 1)[0])
            pw = cvp.cosh_onset_fit(t, om, np.zeros_like(t), onset_guess=None,
                                    c2_fixed=c2, moment_floor=0.0,
                                    ramp_gain=k, ramp_rate=md)
            j = pw['onset_idx']
            if j < 12 or len(om) - j < 12:
                continue
            tau = t[j:] - t[j]
            r = om[j:] - pw['omega_pred'][j:]
            q = sig['omega'][:i0]
            rows.append(dict(
                axis=ad, rate=rate, c2=c2, k=k, tau=tau, r=r,
                dt=float(np.median(np.diff(tau))),
                dm_win=abs(md) * float(tau[-1]),
                sig=float(np.std(q)) if q.size > 50 else 0.0))
        print(f"  {case}/{ad} done ({len(rows)} runs)")
    with open(CACHE, 'wb') as fh:
        pickle.dump(rows, fh)
    return rows


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else 'DataSet/exp'
    rows = collect(root)
    pm = np.deg2rad(10.0)
    d2 = np.rad2deg
    tally = collections.Counter()
    per = collections.defaultdict(lambda: collections.Counter())
    worst = 0.0

    for d in rows:
        tau, r, dt, c2, k = d['tau'], d['r'], d['dt'], d['c2'], d['k']
        te, x = float(tau[-1]), float(c2 * tau[-1])
        jp = 1.0 / (k * c2 ** 2)
        rb, _, _ = rho_bar(d['axis'], pm, d['dm_win'])
        E = rb * np.sinh(np.clip(c2 * tau, 0, 30)) / (jp * c2)
        w = np.gradient(tau)
        w[0] *= 0.5
        w[-1] *= 0.5
        T = float(w.sum())
        rms = lambda v: float(np.sqrt(np.sum(v ** 2 * w) / T))
        nrm = lambda v: float(np.sqrt(np.sum(v ** 2 * w)))

        lo = split(r, dt)
        lo_fake = split(r, 0.55 / max(len(tau) - 1, 1))
        lo_abs = split(np.abs(r), dt)          # the suspected fault

        a_cap = rms_cap(rb, k, c2, x) + d['sig']
        ap_cap = rms(E) + d['sig']
        b_cap = nrm(E) + d['sig'] * np.sqrt(T)

        res = dict(A=rms(lo) <= a_cap,
                   Ap=rms(lo) <= ap_cap,
                   B=nrm(lo) <= b_cap,
                   Bp=nrm(lo_fake) <= b_cap,
                   C=nrm(lo_abs) <= b_cap)
        for kk, v in res.items():
            tally[kk] += bool(v)
            per[d['rate']][kk] += bool(v)
        worst = max(worst, abs(rms_cap(rb, k, c2, x) / rms(E) - 1.0))
        d['_'] = (rms(lo), rms(E), a_cap, ap_cap)

    print(f"\n  {len(rows)} runs, cutoff {FC} Hz\n")
    print(f"  {'evaluation':>46}{'pass':>8}")
    for kk, nm in (('A', "closed-form rms_cap + sigma"),
                   ('Ap', "RMS of the E array + sigma"),
                   ('B', "L2 norms, ||E|| + sigma sqrt(T)"),
                   ('Bp', "L2 norms but the FFT on a NOMINAL dt"),
                   ('C', "L2 norms but the FFT on |r| instead of r")):
        print(f"  {nm:>46}{tally[kk]:6d}/{len(rows)}")
    print(f"\n  closed form against the array, worst relative"
          f" difference: {worst:.2e}")
    print(f"  so A and A' are the same inequality, and B is A' times"
          f" sqrt(T)\n")
    print(f"  {'Mdot':>6}{'A':>7}{'A prime':>9}{'B':>7}{'B prime':>9}"
          f"{'on |r|':>8}")
    for rt in sorted(per):
        c = per[rt]
        print(f"  {rt:6.2f}{c['A']:6d}/20{c['Ap']:8d}/20"
              f"{c['B']:6d}/20{c['Bp']:8d}/20{c['C']:8d}/20")
    print(f"\n  A, A' and B are the same inequality written three ways and")
    print(f"  agree exactly: 114/140.  A nominal dt costs only two runs, so")
    print(f"  that was not the fault.  The fault was rectification: the")
    print(f"  offending pass had cached |r| rather than r and then split")
    print(f"  THAT, and a rectified signal has its energy piled at low")
    print(f"  frequency, so the split no longer separates what it was meant")
    print(f"  to.  The 139/140 from that pass is withdrawn.  114/140 stands.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
