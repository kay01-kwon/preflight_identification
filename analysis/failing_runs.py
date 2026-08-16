#!/usr/bin/env python3
"""The 26 runs that fail the residual check, named and accounted for.

The check of Sec. VIII is (VIII.3) in RMS form,

    RMS(r_lo)  <=  RMS(E) + sigma_n,

and 114 of 140 runs satisfy it.  The 26 that do not are not scattered:
every one of them sits at Mdot >= 0.65, and none of them misses by much.
Reporting "114/140" and moving on would leave a reviewer to decide for
themselves whether the shortfall is the rho analysis failing or something
else, so this decides it instead.

The bound has one hypothesis that can be tested directly.  It reads

    ||y - g_hat|| = min_g ||y - g|| <= ||y - f|| = ||e + n||,

and the first equality is only true if g_hat really is the minimiser over
the model family.  In the campaign it is not: C2 and K are pinned per
configuration by the stage-1 PNLS fit and frozen for every run of that
configuration, so the per-run fit is a constrained minimiser and can only
be worse.  Freeing (C1, C2, c) on the same window, with the onset held,
gives the minimiser the bound is actually about.  If the excess is the
pinning, it disappears; if it survives, the true nominal is not in the
family on those windows and rho is not the whole story.

The second hypothesis is the disturbance.  sigma_n is measured before the
onset, where the vehicle rests on every landing gear; after the onset it
pivots on one edge and the in-window content above 5 Hz is some 7 times
the pre-onset floor.  A disturbance that is louder in the window is also
louder BELOW the cutoff, and that part is charged to the model.  Its size
can be estimated without assuming a level: take the pre-onset record over
a window of the same length, split it at the same 5 Hz, and use its
lo/hi ratio -- a property of the disturbance SHAPE, not its size -- to
carry the measured in-window hi content down into the low band.

Both tests are run here, separately and together, and what each one is
worth is printed per rate.  What comes out:

  the 26 are all at Mdot >= 0.65 (3, 9, 14 of 20 at the three fastest
  rates, none at the four slowest), from 8 of the 10 configurations, and
  the worst misses by 1.80.  A shortfall that respects the rate and
  ignores the configuration is not one bad vehicle.

  test 1 settles it.  Freeing the AMPLITUDE alone, C2 held at its
  physical value and the onset held where the estimator put it, brings
  all 140 inside.  The freed amplitude sits 11.3% below the pinned K
  Mdot, and dC1 (cosh C2 tau - 1) band-limited the same way predicts the
  measured excess to within 0 to 13%, median 3%.  Only 1 to 2.6% of the
  11% is the ramp rounding off near the allocation cap; the rest is the
  stage-one gain, the same 5 to 12.5% the calibration stage is shown
  elsewhere to inherit from e_omega in its own fitting data.

  so the family provably contains a curve inside the cap for every run,
  and rho is not what fails -- but the pinned gain is a real bias.
  Re-running the onset sweep with the gain each window prefers moves the
  per-direction threshold 7.2 mN.m in the median, 24.3 at the fastest
  ramp, which is 0.81 mm.  In the half-sum it moves 1.30 mN.m, 0.043 mm,
  because the gain is common to both directions while Mdot is not.

  test 2 accounts for the rest of the margin on its own: 114 -> 124.

Usage: python analysis/failing_runs.py [DATASET_ROOT]
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

from fit_quality_bound import rho_bar

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '.failing_cache.pkl')
FC = 5.0
W_MIN = 30.08
PHI_BOX = np.deg2rad(10.0)


def split(v, dt, fc=FC):
    """Low-frequency part of v, by zeroing FFT bins above fc."""
    vv = v - v.mean()
    F = np.fft.rfft(vv)
    F[np.fft.rfftfreq(len(vv), d=dt) > fc] = 0.0
    lo = np.fft.irfft(F, n=len(vv))
    return lo, vv - lo


def amplitude_best(tau, om, c2):
    """min over (C1, c) with C2 held: the amplitude channel alone."""
    u = np.cosh(np.clip(c2 * tau, 0.0, 30.0)) - 1.0
    A = np.column_stack([u, np.ones_like(u)])
    co, *_ = np.linalg.lstsq(A, om, rcond=None)
    return float(co[0]), om - A @ co


def family_best(tau, om, c2_lo=1.0, c2_hi=25.0, n=600):
    """min over (C1, C2, c) of ||om - C1(cosh C2 tau - 1) - c||.

    C1 and c are linear once C2 is fixed, so this is a one-dimensional
    search with a closed-form inner solve.  The grid is followed by a
    parabolic refinement on the three points around the minimum.
    """
    grid = np.linspace(c2_lo, c2_hi, n)

    def cost(c2):
        u = np.cosh(np.clip(c2 * tau, 0.0, 30.0)) - 1.0
        A = np.column_stack([u, np.ones_like(u)])
        co, *_ = np.linalg.lstsq(A, om, rcond=None)
        return float(np.sum((om - A @ co) ** 2)), co

    vals = np.array([cost(c)[0] for c in grid])
    j = int(np.argmin(vals))
    c2 = grid[j]
    if 0 < j < n - 1:
        a, b, c_ = vals[j - 1], vals[j], vals[j + 1]
        den = a - 2 * b + c_
        if den > 0:
            c2 += np.clip(0.5 * (a - c_) / den, -0.5, 0.5) * (grid[1] - grid[0])
    v, co = cost(c2)
    u = np.cosh(np.clip(c2 * tau, 0.0, 30.0)) - 1.0
    return float(c2), float(co[0]), om - (co[0] * u + co[1])


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
            npost = len(tau)
            r = om[j:] - pw['omega_pred'][j:]
            # A quiet stretch of the SAME length, ending where the
            # excitation window opens, so the band split it feeds has the
            # same frequency resolution as the in-window one.
            q = sig['omega'][:i0]
            quiet = q[-npost:] if q.size >= npost else np.array([])
            oc = om[j:] - float(pw['c'])
            # The amplitude the model is given is C1 = K Mdot with Mdot
            # the slope over the WHOLE excitation window.  What drives
            # the tip-over is the slope after the onset, and the two are
            # not the same if the commanded ramp rounds off near the cap.
            mdp = float(np.polyfit(tau, mom[j:], 1)[0])
            # Does any of this reach the reported answer?  Re-run the
            # onset sweep with the gain the window itself prefers and
            # see where the threshold lands.
            c1a, _ = amplitude_best(tau, om[j:], c2)
            k_fix = abs(c1a / md) if abs(md) > 1e-9 else k
            pw2 = cvp.cosh_onset_fit(t, om, np.zeros_like(t),
                                     onset_guess=None, c2_fixed=c2,
                                     moment_floor=0.0, ramp_gain=k_fix,
                                     ramp_rate=md)
            rows.append(dict(
                mcrit=float(mom[j]),
                mcrit_fix=float(mom[pw2['onset_idx']]),
                k_fix=k_fix,
                case=case, axis=ad, rate=rate, c2=c2, k=k,
                md_full=abs(md), md_post=abs(mdp),
                sign=int(np.sign(md)), tau=tau, om=om[j:], r=r, quiet=quiet,
                dt=float(np.median(np.diff(tau))),
                dm_win=abs(md) * float(tau[-1]),
                sig=float(np.std(q)) if q.size > 50 else 0.0,
                phi_end=float(abs(np.trapz(oc, tau)))))
        print(f"  {case}/{ad} done ({len(rows)} runs)")
    with open(CACHE, 'wb') as fh:
        pickle.dump(rows, fh)
    return rows


def evaluate(rows):
    """Per run: the cap, the pinned residual, and the two tests."""
    d2 = np.rad2deg
    for d in rows:
        tau, dt, c2, k = d['tau'], d['dt'], d['c2'], d['k']
        jp = 1.0 / (k * c2 ** 2)
        rb, _, _ = rho_bar(d['axis'], PHI_BOX, d['dm_win'])
        E = rb * np.sinh(np.clip(c2 * tau, 0, 30)) / (jp * c2)
        w = np.gradient(tau)
        w[0] *= 0.5
        w[-1] *= 0.5
        T = float(w.sum())
        rms = lambda v: float(np.sqrt(np.sum(v ** 2 * w) / T))

        lo, hi = split(d['r'], dt)
        c1a, ra = amplitude_best(tau, d['om'], c2)
        loa, _ = split(ra, dt)
        c2f, c1f, rf = family_best(tau, d['om'])
        lof, hif = split(rf, dt)

        # The disturbance shape, from a quiet stretch of equal length.
        kap = np.nan
        if d['quiet'].size == len(tau):
            ql, qh = split(np.asarray(d['quiet'], float), dt)
            qh_r = float(np.sqrt(np.mean(qh ** 2)))
            if qh_r > 0:
                kap = float(np.sqrt(np.mean(ql ** 2))) / qh_r

        d['cap'] = rms(E) + d['sig']
        d['cap_dist'] = rms(E) + (kap * rms(hi) if np.isfinite(kap)
                                  else d['sig'])
        d['lo'] = rms(lo)
        d['hi'] = rms(hi)
        # What an amplitude error of exactly that size would leave
        # behind: the shape it multiplies, band-limited the same way.
        u = np.cosh(np.clip(c2 * tau, 0.0, 30.0)) - 1.0
        ulo, _ = split(u, dt)
        d['u_lo'] = rms(ulo)
        d['lo_a'] = rms(loa)
        d['lo_f'] = rms(lof)
        d['hi_f'] = rms(hif)
        d['c1_pin'] = k * d['dm_win'] / max(float(tau[-1]), 1e-9)
        d['c1_amp'] = abs(c1a)
        d['c2_free'] = c2f
        d['ratio_a'] = d['lo_a'] / d['cap']
        d['kappa'] = kap
        d['ratio'] = d['lo'] / d['cap']
        d['ratio_f'] = d['lo_f'] / d['cap']
        d['ratio_d'] = d['lo'] / d['cap_dist']
        d['ratio_fd'] = d['lo_f'] / d['cap_dist']
    return rows


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else 'DataSet/exp'
    rows = evaluate(collect(root))
    d2 = np.rad2deg
    n = len(rows)
    bad = sorted([d for d in rows if d['ratio'] > 1.0],
                 key=lambda d: -d['ratio'])

    print(f"\n  {n} runs, cutoff {FC} Hz, cap = RMS(E) + sigma_n at the"
          f" 10 deg box")
    print(f"  {n - len(bad)} inside, {len(bad)} outside.\n")

    print(f"  --- the {len(bad)} that fail, worst first ---\n")
    print(f"  {'configuration':>16}{'axis':>6}{'Mdot':>7}{'dir':>5}"
          f"{'resid <5Hz':>12}{'cap':>9}{'ratio':>8}{'excess':>9}")
    print(f"  {'':16}{'':6}{'N m/s':>7}{'':5}{'deg/s':>12}{'deg/s':>9}"
          f"{'':8}{'deg/s':>9}")
    for d in bad:
        print(f"  {d['case'].replace('case_', ''):>16}{d['axis']:>6}"
              f"{d['rate']:7.2f}{'+' if d['sign'] > 0 else '-':>5}"
              f"{d2(d['lo']):12.3f}{d2(d['cap']):9.3f}{d['ratio']:8.3f}"
              f"{d2(d['lo'] - d['cap']):9.3f}")

    print(f"\n  --- where they are ---\n")
    by_rate = collections.defaultdict(list)
    for d in rows:
        by_rate[d['rate']].append(d)
    print(f"  {'Mdot':>6}{'fail':>7}{'x = C2 tau':>12}{'median ratio':>14}"
          f"{'worst':>8}")
    for rt in sorted(by_rate):
        v = by_rate[rt]
        rr = np.array([d['ratio'] for d in v])
        xs = np.median([d['c2'] * d['tau'][-1] for d in v])
        print(f"  {rt:6.2f}{sum(rr > 1):5d}/{len(v)}{xs:12.2f}"
              f"{np.median(rr):14.3f}{rr.max():8.3f}")

    by_cfg = collections.defaultdict(list)
    for d in rows:
        by_cfg[(d['case'], d['axis'])].append(d)
    print(f"\n  {'configuration':>16}{'axis':>6}{'fail':>8}{'of':>5}"
          f"{'median ratio':>14}")
    for kk in sorted(by_cfg):
        v = by_cfg[kk]
        rr = np.array([d['ratio'] for d in v])
        print(f"  {kk[0].replace('case_', ''):>16}{kk[1]:>6}"
              f"{int((rr > 1).sum()):8d}{len(v):5d}{np.median(rr):14.3f}")
    nz = sum(1 for kk, v in by_cfg.items()
             if any(d['ratio'] > 1 for d in v))
    print(f"\n  {nz} of {len(by_cfg)} configurations contribute at least one"
          f" failure,")
    print(f"  so the shortfall is not one bad configuration.")

    print(f"\n  --- test 1: free the pinned constants, onset held ---\n")
    print(f"  the residual below 5 Hz, three fits of the same window:"
          f" both pinned,")
    print(f"  amplitude freed with C2 held, and the full family"
          f" minimiser.\n")
    print(f"  {'Mdot':>6}{'pinned':>9}{'C1 free':>9}{'both free':>11}"
          f"{'C1 free':>10}{'both':>7}{'C1 fit /':>11}{'C2 free /':>11}")
    print(f"  {'N m/s':>6}{'deg/s':>9}{'deg/s':>9}{'deg/s':>11}"
          f"{'inside':>10}{'inside':>7}{'C1 pinned':>11}{'C2 pinned':>11}")
    for rt in sorted(by_rate):
        v = by_rate[rt]
        a = np.median([d['lo'] for d in v])
        b = np.median([d['lo_a'] for d in v])
        c = np.median([d['lo_f'] for d in v])
        print(f"  {rt:6.2f}{d2(a):9.3f}{d2(b):9.3f}{d2(c):11.3f}"
              f"{sum(1 for d in v if d['ratio_a'] <= 1):7d}/{len(v)}"
              f"{sum(1 for d in v if d['ratio_f'] <= 1):5d}/{len(v)}"
              f"{np.median([d['c1_amp'] / d['c1_pin'] for d in v]):11.3f}"
              f"{np.median([d['c2_free'] / d['c2'] for d in v]):11.3f}")
    print(f"\n  inside with the amplitude freed:"
          f" {sum(1 for d in rows if d['ratio_a'] <= 1)}/{n}")
    print(f"  inside with the family minimiser:"
          f" {sum(1 for d in rows if d['ratio_f'] <= 1)}/{n}")
    ca = np.array([d['c1_amp'] / d['c1_pin'] for d in rows
                   if abs(d['c1_pin']) > 1e-9])
    print(f"  the freed amplitude is {100 * (np.median(ca) - 1):+.1f}% of the"
          f" pinned one (median),")
    print(f"  p10 {100 * (np.percentile(ca, 10) - 1):+.1f}%,"
          f" p90 {100 * (np.percentile(ca, 90) - 1):+.1f}%")

    print(f"\n  where the amplitude deficit comes from: C1 = K Mdot uses the")
    print(f"  slope of the WHOLE excitation window, but only the slope after")
    print(f"  the onset drives the tip-over.\n")
    print(f"  {'Mdot':>6}{'slope after onset':>20}{'amplitude fitted':>19}"
          f"{'unexplained':>13}")
    print(f"  {'N m/s':>6}{'over slope of all':>20}{'over C1 pinned':>19}"
          f"{'':>13}")
    for rt in sorted(by_rate):
        v = by_rate[rt]
        s = np.median([d['md_post'] / d['md_full'] for d in v])
        a = np.median([d['c1_amp'] / d['c1_pin'] for d in v])
        print(f"  {rt:6.2f}{s:20.3f}{a:19.3f}{a / s:13.3f}")
    print(f"\n  and it accounts for the excess in size, not only in sign:"
          f" an\n  amplitude error dC1 leaves dC1 (cosh C2 tau - 1) behind.\n")
    print(f"  {'Mdot':>6}{'|dC1| x shape':>16}{'measured excess':>18}"
          f"{'ratio':>9}{'cap':>9}")
    print(f"  {'N m/s':>6}{'deg/s':>16}{'deg/s':>18}{'':9}{'deg/s':>9}")
    for rt in sorted(by_rate):
        v = by_rate[rt]
        p = np.median([abs(d['c1_pin'] - d['c1_amp']) * d['u_lo'] for d in v])
        m = np.median([np.sqrt(max(d['lo'] ** 2 - d['lo_a'] ** 2, 0.0))
                       for d in v])
        print(f"  {rt:6.2f}{d2(p):16.3f}{d2(m):18.3f}{p / m:9.3f}"
              f"{d2(np.median([d['cap'] for d in v])):9.3f}")

    print(f"\n  --- test 2: the in-window disturbance ---\n")
    ks = np.array([d['kappa'] for d in rows if np.isfinite(d['kappa'])])
    print(f"  kappa = RMS(<5Hz)/RMS(>5Hz) on a quiet window of the same")
    print(f"  length: median {np.median(ks):.3f}, "
          f"p10 {np.percentile(ks, 10):.3f}, p90 {np.percentile(ks, 90):.3f}"
          f"  ({len(ks)} runs)\n")
    print(f"  {'Mdot':>6}{'sigma_pre':>11}{'in-win >5Hz':>13}"
          f"{'kappa x hi':>12}{'cap':>9}{'cap+dist':>10}{'inside':>9}")
    print(f"  {'N m/s':>6}{'deg/s':>11}{'deg/s':>13}{'deg/s':>12}"
          f"{'deg/s':>9}{'deg/s':>10}{'':9}")
    for rt in sorted(by_rate):
        v = by_rate[rt]
        print(f"  {rt:6.2f}{d2(np.median([d['sig'] for d in v])):11.3f}"
              f"{d2(np.median([d['hi'] for d in v])):13.3f}"
              f"{d2(np.nanmedian([d['kappa'] * d['hi'] for d in v])):12.3f}"
              f"{d2(np.median([d['cap'] for d in v])):9.3f}"
              f"{d2(np.median([d['cap_dist'] for d in v])):10.3f}"
              f"{sum(1 for d in v if d['ratio_d'] <= 1):6d}/{len(v)}")
    print(f"\n  total inside with the in-window disturbance:"
          f" {sum(1 for d in rows if d['ratio_d'] <= 1)}/{n}")
    print(f"  total inside with both:"
          f" {sum(1 for d in rows if d['ratio_fd'] <= 1)}/{n}")

    print(f"\n  --- does it reach the reported answer? ---\n")
    print(f"  Re-run the onset sweep with the gain each window prefers,"
          f" and\n  compare the threshold and the half-sum it feeds."
          f"  W = {W_MIN} N.\n")
    print(f"  {'Mdot':>6}{'K change':>11}{'M_crit shift':>14}"
          f"{'as offset':>11}{'half-sum shift':>16}{'as offset':>11}")
    print(f"  {'N m/s':>6}{'':11}{'mN.m':>14}{'mm':>11}{'mN.m':>16}"
          f"{'mm':>11}")
    hs = collections.defaultdict(dict)
    for d in rows:
        hs[(d['case'], d['axis'], round(d['rate'], 2))][d['sign']] = d
    for rt in sorted(by_rate):
        v = by_rate[rt]
        dk = np.median([d['k_fix'] / d['k'] for d in v])
        dm = np.median([abs(d['mcrit_fix'] - d['mcrit']) for d in v])
        pairs = [p for kk, p in hs.items()
                 if kk[2] == round(rt, 2) and 1 in p and -1 in p]
        dh = [abs(0.5 * (p[1]['mcrit_fix'] + p[-1]['mcrit_fix'])
                  - 0.5 * (p[1]['mcrit'] + p[-1]['mcrit'])) for p in pairs]
        dh = np.median(dh) if dh else np.nan
        print(f"  {rt:6.2f}{dk:11.3f}{1e3 * dm:14.2f}{1e3 * dm / W_MIN:11.3f}"
              f"{1e3 * dh:16.2f}{1e3 * dh / W_MIN:11.3f}")
    allp = [p for p in hs.values() if 1 in p and -1 in p]
    d1 = np.median([abs(d['mcrit_fix'] - d['mcrit']) for d in rows])
    dh = np.median([abs(0.5 * (p[1]['mcrit_fix'] + p[-1]['mcrit_fix'])
                        - 0.5 * (p[1]['mcrit'] + p[-1]['mcrit']))
                    for p in allp])
    print(f"\n  per direction the threshold moves {1e3 * d1:.2f} mN.m"
          f" ({1e3 * d1 / W_MIN:.3f} mm),")
    print(f"  but in the half-sum only {1e3 * dh:.2f} mN.m"
          f" ({1e3 * dh / W_MIN:.3f} mm) -- a factor of"
          f" {d1 / dh:.1f} --")
    print(f"  because Mdot flips sign with the direction and the shift"
          f" with it.")

    print(f"\n  --- what is left ---\n")
    left = [d for d in rows if d['ratio_fd'] > 1.0]
    print(f"  {len(left)} runs survive both corrections"
          f"{',' if left else '.'}")
    if left:
        print(f"  worst ratio {max(d['ratio_fd'] for d in left):.3f},"
              f" at Mdot"
              f" {sorted({d['rate'] for d in left})}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
