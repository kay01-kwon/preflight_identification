#!/usr/bin/env python3
"""The error of the cosh method, measured rather than bounded.

Every bound in Sec. VI-E is a priori: it is built from a scale, a CAD
inertia and the excitation protocol, and it says what the identified
threshold CANNOT be off by.  A reviewer is entitled to ask what it
actually IS off by, and this experiment can answer that without any
model at all, because of one fact:

    M_crit is a STATIC property of the rig.  It is the moment at which
    the vehicle is on the verge of tipping over a landing-gear edge,
    W x arm, and it does not depend on how fast the moment was ramped
    to get there.

So the identified M_crit, plotted against the ramp rate over the seven
rates of the protocol, should be FLAT.  Any slope is a rate-dependent
error and any scatter about the line is a rate-independent one.  Nothing
has to be assumed about where either comes from.  That decomposition is
what this measures, per configuration and per direction:

    intercept   the rate-free estimate, the best available truth proxy
    slope       systematic, rate-dependent error -- rho and the onset
                grid both live here, both being proportional to Mdot
    scatter     everything else, run to run

and then the same for the quantity actually reported, the half-sum
M_off = (M_+ + M_-)/2, whose error is what becomes the CoM offset.

What comes out, over 140 runs and ten configurations:

    per direction   slope over the rate range   3.9 to 86.6 mN.m
                    scatter about the line      16 to 43 mN.m
    half-sum        1 sigma across the rates    0.28 to 1.13 mm
                    worst peak-to-peak          3.28 mm

against an a priori rho_bar of 12.04 mN.m for roll and 9.95 for pitch,
and the 0.400 mm bound of (108).  So the realised error is 2 to 7 times
what the small-angle and GE approximations can account for, and the same
factor appears in the fit residual (analysis/fit_quality_bound.py),
which exceeds its own rho cap by about 7 at the realised tilt.  One
disturbance is doing both.

That is not a failure of the bound and it should not be presented as
one.  (108) certifies ONE channel: it says the modelling approximations
contribute at most 0.400 mm, so whatever limits the method, it is not
the neglected sin(phi).  The performance claim is a separate,
model-free number -- the repeatability above -- and the two must be
quoted as what they are rather than merged.  Median half-sum spread is
0.50 mm, consistent with the 1.64 mm validation RMS once the offset
comparison's own error is included.

Usage: python analysis/mcrit_repeatability.py [DATASET_ROOT]
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

W_MIN = 30.08                          # N, the weight the offset divides by
TS = 0.010                             # s, the onset grid
RHO_BAR = {'Mx': 0.01204, 'My': 0.00995}   # N m, the a priori bound (97b)


def collect(root, cache):
    if os.path.exists(cache):
        with open(cache, 'rb') as fh:
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
            rows.append(dict(case=case, axis=ad, rate=rate,
                             sign=int(np.sign(md)), mdot=abs(md),
                             mcrit=float(mom[j])))
        print(f"  {case}/{ad} done ({len(rows)} runs)")
    with open(cache, 'wb') as fh:
        pickle.dump(rows, fh)
    return rows


def line(x, y):
    """Least-squares slope and intercept, plus the scatter about them."""
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    a, b = np.polyfit(x, y, 1)
    return a, b, float(np.std(y - (a * np.array(x) + b), ddof=2))


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else 'DataSet/exp'
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '.mcrit_cache.pkl')
    rows = collect(root, cache)
    print(f"\n  {len(rows)} runs.  M_crit is static, so the ideal plot of")
    print(f"  M_crit against Mdot is FLAT; slope and scatter are the error.\n")

    g = collections.defaultdict(list)
    for r in rows:
        g[(r['case'], r['axis'], r['sign'])].append(r)

    print(f"  {'config':>14}{'dir':>4}{'n':>3}{'intercept':>11}{'slope':>10}"
          f"{'over range':>12}{'scatter':>10}{'total':>9}")
    print(f"  {'':14}{'':4}{'':3}{'N m':>11}{'N m per':>10}{'mN.m':>12}"
          f"{'mN.m':>10}{'mN.m':>9}")
    print(f"  {'':14}{'':4}{'':3}{'':11}{'N m/s':>10}{'':12}{'1 sigma':>10}"
          f"{'rss':>9}")
    per = {}
    for k in sorted(g):
        v = g[k]
        md = [r['mdot'] for r in v]
        mc = [abs(r['mcrit']) for r in v]
        a, b, s = line(md, mc)
        span = 1e3 * abs(a) * (max(md) - min(md))
        tot = np.sqrt(span ** 2 + (1e3 * s) ** 2)
        per[k] = (b, a, s)
        print(f"  {k[0] + '/' + k[1]:>14}{'+' if k[2] > 0 else '-':>4}"
              f"{len(v):3d}{b:11.4f}{a:10.4f}{span:12.2f}{1e3*s:10.2f}"
              f"{tot:9.2f}")

    print(f"\n  against the a priori budget\n")
    for ad in ('Mx', 'My'):
        sl = [abs(per[k][1]) for k in per if k[1] == ad]
        sc = [1e3 * per[k][2] for k in per if k[1] == ad]
        print(f"  {ad}: |slope| median {np.median(sl):.4f} N m per N m/s"
              f"  ->  {1e3*np.median(sl)*1.10:.2f} mN.m over the rate range")
        print(f"      scatter median {np.median(sc):.2f} mN.m,"
              f"  a priori rho_bar {1e3*RHO_BAR[ad]:.2f} mN.m")
        print(f"      onset grid alone would give"
              f" {1e3*0.5*TS*1.20:.2f} mN.m at the fastest ramp")

    # the reported quantity: the half-sum, per configuration and rate
    print(f"\n  the half-sum M_off = (M_+ + M_-)/2, which becomes the"
          f" offset\n")
    print(f"  {'config':>14}{'n rates':>9}{'M_off':>10}{'spread':>10}"
          f"{'as offset':>12}{'p-p':>9}")
    print(f"  {'':14}{'':9}{'mN.m':>10}{'1 sigma':>10}{'mm':>12}{'mm':>9}")
    hs = collections.defaultdict(dict)
    for r in rows:
        hs[(r['case'], r['axis'])].setdefault(r['rate'], {})[r['sign']] = \
            r['mcrit']
    worst = 0.0
    for k in sorted(hs):
        vals = [0.5 * (d[1] + d[-1]) for d in hs[k].values()
                if 1 in d and -1 in d]
        if len(vals) < 3:
            continue
        v = np.array(vals)
        sd, pp = float(np.std(v, ddof=1)), float(v.max() - v.min())
        worst = max(worst, 1e3 * pp / W_MIN)
        print(f"  {k[0] + '/' + k[1]:>14}{len(v):9d}{1e3*np.mean(v):10.3f}"
              f"{1e3*sd:10.3f}{1e3*sd/W_MIN:12.4f}{1e3*pp/W_MIN:9.4f}")
    print(f"\n  worst peak-to-peak offset spread across ramp rates:"
          f" {worst:.4f} mm")
    print(f"  against the 0.400 mm a priori bound and the 1.64 mm"
          f" validation RMS.")
    print(f"\n  Read the half-sum table as the headline number: it is the")
    print(f"  reported quantity, measured seven times per configuration")
    print(f"  under conditions that should give identical answers, with no")
    print(f"  model in the comparison.")
    decompose(rows)
    return 0





def decompose(rows):
    """Is the dominant error in the MOMENT domain or the TIME domain?

    Two questions, and the second only makes sense after the first.

    (1) Are the per-configuration slopes real?  With seven rates spanning
        0.10 to 1.20 the regressor sum of squares is Sxx = 0.952, so the
        slope standard error is sigma/sqrt(Sxx) ~= sigma.  A scatter of
        29 mN.m therefore puts a 1-sigma slope at 0.030 N m per N m/s --
        the same size as most of the slopes observed.  Reported here with
        t-statistics so the significant ones can be told from the rest.

    (2) The residual scatter about each line is the dominant term.  Its
        domain says what it is.  A failure to locate the onset is an
        error in TIME, so it would appear as a moment error proportional
        to Mdot: sigma_M ~ Mdot, sigma_t flat.  A genuine run-to-run
        change in where the vehicle tips -- gear settling, floor
        friction, repositioning -- is an error in MOMENT, so sigma_M is
        flat and sigma_t falls as 1/Mdot.  The two are distinguishable
        with no model at all.
    """
    import collections as _c
    print("\n\n  (1) are the slopes real?  SE ~= sigma/sqrt(Sxx),"
          " Sxx = 0.952\n")
    print(f"  {'config':>14}{'dir':>4}{'slope':>10}{'SE':>9}{'t':>7}"
          f"{'|t|>2':>7}")
    g = _c.defaultdict(list)
    for r in rows:
        g[(r['case'], r['axis'], r['sign'])].append(r)
    resid = _c.defaultdict(list)
    nsig = 0
    for k in sorted(g):
        v = g[k]
        md = np.array([r['mdot'] for r in v])
        mc = np.array([abs(r['mcrit']) for r in v])
        a, b = np.polyfit(md, mc, 1)
        e = mc - (a * md + b)
        s = float(np.std(e, ddof=2))
        sxx = float(np.sum((md - md.mean()) ** 2))
        se = s / np.sqrt(sxx)
        t = a / se if se > 0 else 0.0
        nsig += abs(t) > 2
        for r, ei in zip(v, e):
            resid[r['rate']].append(ei)
        print(f"  {k[0] + '/' + k[1]:>14}{'+' if k[2] > 0 else '-':>4}"
              f"{a:10.4f}{se:9.4f}{t:7.2f}{'yes' if abs(t) > 2 else '':>7}")
    print(f"\n  {nsig}/{len(g)} slopes significant at |t| > 2."
          f"  The rate-dependent term is mostly NOT resolved;")
    print(f"  the scatter is what dominates, and (2) says what it is.\n")

    print(f"  (2) the scatter, in both domains\n")
    print(f"  {'Mdot':>6}{'n':>4}{'sigma_M':>11}{'sigma_t':>11}"
          f"{'as offset':>12}")
    print(f"  {'N m/s':>6}{'':4}{'mN.m':>11}{'ms':>11}{'mm':>12}")
    ms, ts, rt = [], [], []
    for rate in sorted(resid):
        e = np.array(resid[rate])
        sm = float(np.std(e, ddof=1))
        st = sm / rate
        ms.append(1e3 * sm)
        ts.append(1e3 * st)
        rt.append(rate)
        print(f"  {rate:6.2f}{len(e):4d}{1e3*sm:11.2f}{1e3*st:11.2f}"
              f"{1e3*sm/W_MIN:12.4f}")
    ms, ts, rt = np.array(ms), np.array(ts), np.array(rt)
    # which one is flat?  compare the spread of each across the rates
    cv_m = ms.std(ddof=1) / ms.mean()
    cv_t = ts.std(ddof=1) / ts.mean()
    sl_m = np.polyfit(rt, ms, 1)[0]
    sl_t = np.polyfit(rt, ts, 1)[0]
    print(f"\n  sigma_M across rates: mean {ms.mean():.2f} mN.m,"
          f" CV {cv_m:.1%}, slope {sl_m:+.2f} per N m/s")
    print(f"  sigma_t across rates: mean {ts.mean():.2f} ms,"
          f" CV {cv_t:.1%}, slope {sl_t:+.2f} per N m/s")
    if cv_m < cv_t:
        print(f"\n  sigma_M is the flatter of the two, so the dominant error"
              f" lives in the\n  MOMENT domain: the vehicle does not tip at"
              f" exactly the same moment twice.\n  That is the rig, not the"
              f" onset detector -- gear settling, floor friction,\n"
              f"  repositioning between runs.")
    else:
        print(f"\n  sigma_t is the flatter of the two, so the dominant error"
              f" lives in the\n  TIME domain: it is a failure to locate the"
              f" onset, and it is the\n  algorithm's to own.")
    return ms, ts, rt


if __name__ == '__main__':
    sys.exit(main())
