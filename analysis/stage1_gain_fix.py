#!/usr/bin/env python3
"""Correct the stage-one gain bias, and reprocess all 140 runs with it.

analysis/failing_runs.py showed that the amplitude each window prefers
sits about 11% below the pinned C1 = K Mdot, and that this -- not the
small-angle or ground-effect remainder -- is what puts 26 runs outside
the residual cap.  The obvious next step is to take the gain the data
prefers and run the campaign again on it.  That step is not free, and
the reason is in estimate_rig_constants' own docstring:

    with the onset swept over the whole window, a smaller amplitude with
    an earlier onset trades off against a larger amplitude with a later
    one, so the residual is degenerate along a ridge in (C2, K)

Stage two therefore does NOT pick K by residual.  It picks the K that
makes M_crit most repeatable across ramp rates, on the physical grounds
that a static tip-over threshold cannot depend on how fast the moment was
ramped.  Moving K to the residual optimum slides along that ridge, and
the onset slides with it.  So the question is not "is K biased" but
"which of the two criteria should set it", and that is a measurement:

    scale the frozen K by s, hold C2, reprocess every run, and score
    each s on BOTH criteria at once -- the fit residual and the
    ramp-invariance of the identified threshold -- plus the quantity
    actually reported, the two-sided half-sum and its spread.

If the two optima coincide, the gain was simply biased and correcting it
costs nothing.  If they do not, the 11% is the price stage two pays for
repeatability, and it should be paid knowingly rather than removed.

They do not coincide, and the reason is that only one of them is a
criterion at all:

  the residual has NO interior minimum.  It falls monotonically from
  s = 1 down to s ~ 0.43, a 57% cut in K, and 8 of the 10 configurations
  put their optimum at whatever the low edge of the grid happens to be.
  A 57% smaller K asserts W z_CoM is 2.3 times what the scale says.  The
  residual cannot set K; the ridge is real and this is what it looks
  like from the outside.

  the runs inside the cap follow the residual, 114/140 at s = 1 and
  139/140 at s = 0.43.  So the pass count is not a criterion for setting
  K either -- it is bought by making the gain physically wrong.

  the invariance score DOES have an interior minimum, at s = 0.97 in the
  median, with the per-configuration optima scattered 0.76 to 1.21 on
  fourteen runs each.  That is scatter about 1, not a bias away from it:
  stage two's choice is right and the correction available is 3%.

Applying it changes nothing that is reported.  At the global s = 0.97
the offset moves 0.019 mm in the median and 0.068 at worst, and the
half-sum spread goes 0.522 -> 0.533 mm; letting each configuration take
its own optimum gives 0.022 mm and 0.522 -> 0.506 mm.

The robustness this exposes is worth more than the correction.  Over the
whole ridge -- K scaled from 0.40 to 1.30, a factor of three -- the
reported offset moves at most 0.10 mm in the median and the spread stays
between 0.507 and 0.574 mm.  The gain is common to both tip directions
while Mdot is not, so the half-sum removes it, and the answer is
insensitive to a constant the fit residual cannot pin down at all.

Usage: python analysis/stage1_gain_fix.py [DATASET_ROOT]
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
                     '.gainfix_cache.pkl')
FC = 5.0
W_MIN = 30.08
PHI_BOX = np.deg2rad(10.0)
SCALES = np.round(np.arange(0.40, 1.3001, 0.03), 3)


def split_lo(v, dt, fc=FC):
    vv = v - v.mean()
    F = np.fft.rfft(vv)
    F[np.fft.rfftfreq(len(vv), d=dt) > fc] = 0.0
    return np.fft.irfft(F, n=len(vv))


def collect(root):
    """The prepared window of every run, once, so the sweep is cheap."""
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
            q = sig['omega'][:i0]
            rows.append(dict(
                case=case, axis=ad, rate=rate, c2=c2, k=k,
                t=np.asarray(t, float), om=np.asarray(om, float),
                mom=np.asarray(mom, float),
                md=float(np.polyfit(t, mom, 1)[0]),
                sig=float(np.std(q)) if q.size > 50 else 0.0))
        print(f"  {case}/{ad} prepared ({len(rows)} runs)")
    with open(CACHE, 'wb') as fh:
        pickle.dump(rows, fh)
    return rows


def refit(d, k, cvp):
    """One run at one gain: threshold, and the residual it leaves."""
    pw = cvp.cosh_onset_fit(d['t'], d['om'], np.zeros_like(d['t']),
                            onset_guess=None, c2_fixed=d['c2'],
                            moment_floor=0.0, ramp_gain=float(k),
                            ramp_rate=d['md'])
    j = pw['onset_idx']
    if j < 12 or len(d['om']) - j < 12:
        return None
    tau = d['t'][j:] - d['t'][j]
    r = d['om'][j:] - pw['omega_pred'][j:]
    dt = float(np.median(np.diff(tau)))
    jp = 1.0 / (d['k'] * d['c2'] ** 2)
    rb, _, _ = rho_bar(d['axis'], PHI_BOX, abs(d['md']) * float(tau[-1]))
    E = rb * np.sinh(np.clip(d['c2'] * tau, 0, 30)) / (jp * d['c2'])
    w = np.gradient(tau)
    w[0] *= 0.5
    w[-1] *= 0.5
    T = float(w.sum())
    rms = lambda v: float(np.sqrt(np.sum(v ** 2 * w) / T))
    lo = rms(split_lo(r, dt))
    return dict(mcrit=float(d['mom'][j]), sign=int(np.sign(d['md'])),
                rate=d['rate'], lo=lo, cap=rms(E) + d['sig'],
                resid=float(pw['total_residual']))


def score(fits):
    """Both criteria on one configuration's 14 runs, plus the report."""
    by_dir = collections.defaultdict(list)
    for f in fits:
        by_dir[f['sign']].append(f)
    cv = []
    for v in by_dir.values():
        m = np.abs([f['mcrit'] for f in v])
        if len(m) >= 3 and m.mean() > 0:
            cv.append(float(np.std(m, ddof=1) / m.mean()))
    hs = {}
    for f in fits:
        hs.setdefault(round(f['rate'], 2), {})[f['sign']] = f['mcrit']
    half = [0.5 * (v[1] + v[-1]) for v in hs.values() if 1 in v and -1 in v]
    return dict(
        resid=float(np.sum([f['resid'] for f in fits])),
        cv=float(np.sum(cv)) if cv else np.nan,
        half=float(np.mean(half)) if half else np.nan,
        spread=1e3 * float(np.std(half, ddof=1)) / W_MIN if len(half) > 2
        else np.nan,
        inside=int(sum(1 for f in fits if f['lo'] <= f['cap'])),
        n=len(fits))


def sweep(rows, cvp):
    """Every configuration over the gain scale."""
    by_cfg = collections.defaultdict(list)
    for d in rows:
        by_cfg[(d['case'], d['axis'])].append(d)
    out = {}
    for cfg in sorted(by_cfg):
        v = by_cfg[cfg]
        k0 = v[0]['k']
        tab = {}
        for s in SCALES:
            fits = [f for f in (refit(d, s * k0, cvp) for d in v)
                    if f is not None]
            tab[float(s)] = (score(fits), fits)
        out[cfg] = (k0, tab)
        print(f"  {cfg[0]}/{cfg[1]} swept")
    return out


def best(tab, key, lo=True):
    ks = sorted(tab)
    vals = np.array([tab[s][0][key] for s in ks])
    ok = np.isfinite(vals)
    if not ok.any():
        return np.nan
    j = int(np.argmin(vals[ok])) if lo else int(np.argmax(vals[ok]))
    return float(np.array(ks)[ok][j])


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else 'DataSet/exp'
    rows = collect(root)
    import critical_value_getter_piecewise as cvp
    res = sweep(rows, cvp)

    print(f"\n  --- the two criteria, per configuration ---\n")
    print(f"  the gain scale s that each criterion prefers, K -> s K,"
          f" C2 held\n")
    print(f"  {'configuration':>16}{'K':>8}{'residual':>10}{'invariance':>12}"
          f"{'half-sum':>10}{'inside at':>11}{'inside at':>11}")
    print(f"  {'':16}{'':8}{'best s':>10}{'best s':>12}{'best s':>10}"
          f"{'s = 1':>11}{'best s':>11}")
    rows_out = []
    for cfg, (k0, tab) in sorted(res.items()):
        sr = best(tab, 'resid')
        si = best(tab, 'cv')
        sh = best(tab, 'spread')
        rows_out.append((cfg, k0, sr, si, sh, tab))
        print(f"  {cfg[0].replace('case_', '') + '/' + cfg[1]:>16}"
              f"{k0:8.4f}{sr:10.2f}{si:12.2f}{sh:10.2f}"
              f"{tab[1.0][0]['inside']:8d}/{tab[1.0][0]['n']}"
              f"{tab[sr][0]['inside']:8d}/{tab[sr][0]['n']}")
    sr_all = np.array([r[2] for r in rows_out])
    si_all = np.array([r[3] for r in rows_out])
    sh_all = np.array([r[4] for r in rows_out])
    print(f"\n  median: residual {np.median(sr_all):.2f},"
          f"  invariance {np.median(si_all):.2f},"
          f"  half-sum spread {np.median(sh_all):.2f}")

    print(f"\n  --- what moving to the residual optimum costs ---\n")
    print(f"  {'configuration':>16}{'invariance':>12}{'invariance':>12}"
          f"{'spread':>10}{'spread':>10}{'half-sum':>11}")
    print(f"  {'':16}{'at s = 1':>12}{'at s resid':>12}{'s = 1':>10}"
          f"{'s resid':>10}{'shift, mm':>11}")
    for cfg, k0, sr, si, sh, tab in rows_out:
        a, b = tab[1.0][0], tab[sr][0]
        print(f"  {cfg[0].replace('case_', '') + '/' + cfg[1]:>16}"
              f"{a['cv']:12.4f}{b['cv']:12.4f}{a['spread']:10.3f}"
              f"{b['spread']:10.3f}"
              f"{1e3 * (b['half'] - a['half']) / W_MIN:11.3f}")
    cva = np.array([tab[1.0][0]['cv'] for _, _, _, _, _, tab in rows_out])
    cvb = np.array([tab[sr][0]['cv']
                    for _, _, sr, _, _, tab in rows_out])
    spa = np.array([tab[1.0][0]['spread'] for _, _, _, _, _, tab in rows_out])
    spb = np.array([tab[sr][0]['spread']
                    for _, _, sr, _, _, tab in rows_out])
    ins_a = sum(tab[1.0][0]['inside'] for _, _, _, _, _, tab in rows_out)
    ins_b = sum(tab[sr][0]['inside'] for _, _, sr, _, _, tab in rows_out)
    ntot = sum(tab[1.0][0]['n'] for _, _, _, _, _, tab in rows_out)
    print(f"\n  invariance score, median  {np.median(cva):.4f} ->"
          f" {np.median(cvb):.4f}"
          f"  ({100 * (np.median(cvb) / np.median(cva) - 1):+.1f}%)")
    print(f"  half-sum spread, median   {np.median(spa):.3f} ->"
          f" {np.median(spb):.3f} mm"
          f"  ({100 * (np.median(spb) / np.median(spa) - 1):+.1f}%)")
    print(f"  runs inside the cap       {ins_a}/{ntot} -> {ins_b}/{ntot}")

    print(f"\n  --- along the ridge ---\n")
    print(f"  every configuration at the same scale, pooled.  If the"
          f" residual\n  has no interior minimum the criterion is"
          f" degenerate and cannot\n  set K; what matters then is how the"
          f" reported answer behaves.\n")
    print(f"  {'s':>6}{'residual':>11}{'invariance':>12}{'inside':>10}"
          f"{'spread':>9}{'offset shift':>14}{'worst config':>14}")
    print(f"  {'':6}{'sum, norm':>11}{'median':>12}{'of 140':>10}"
          f"{'mm':>9}{'mm':>14}{'mm':>14}")
    base = sum(tab[1.0][0]['resid'] for _, _, _, _, _, tab in rows_out)
    h1 = {cfg: tab[1.0][0]['half'] for cfg, _, _, _, _, tab in rows_out}
    for s in SCALES:
        s = float(s)
        rr = sum(tab[s][0]['resid'] for _, _, _, _, _, tab in rows_out)
        cv = np.median([tab[s][0]['cv'] for _, _, _, _, _, tab in rows_out])
        ins = sum(tab[s][0]['inside'] for _, _, _, _, _, tab in rows_out)
        sp = np.median([tab[s][0]['spread']
                        for _, _, _, _, _, tab in rows_out])
        sh = np.array([1e3 * (tab[s][0]['half'] - h1[cfg]) / W_MIN
                       for cfg, _, _, _, _, tab in rows_out])
        mark = '  <- stage 2' if abs(s - 1.0) < 1e-9 else ''
        print(f"  {s:6.2f}{rr / base:11.4f}{cv:12.4f}{ins:8d}/140"
              f"{sp:9.3f}{np.median(sh):14.3f}"
              f"{sh[np.argmax(np.abs(sh))]:14.3f}{mark}")

    print(f"\n  --- the reported answer, before and after ---\n")
    s_fix = float(np.median(si_all))
    s_fix = min(SCALES, key=lambda z: abs(float(z) - s_fix))
    print(f"  Correcting K by the invariance criterion means s ="
          f" {s_fix:.2f}.\n")
    print(f"  {'configuration':>16}{'offset at':>12}{'offset at':>12}"
          f"{'shift':>9}{'spread':>9}{'spread':>9}")
    print(f"  {'':16}{'s = 1, mm':>12}{'s fixed, mm':>12}{'mm':>9}"
          f"{'s = 1':>9}{'s fixed':>9}")
    for cfg, k0, sr, si, sh_, tab in rows_out:
        a, b = tab[1.0][0], tab[float(s_fix)][0]
        print(f"  {cfg[0].replace('case_', '') + '/' + cfg[1]:>16}"
              f"{1e3 * a['half'] / W_MIN:12.3f}{1e3 * b['half'] / W_MIN:12.3f}"
              f"{1e3 * (b['half'] - a['half']) / W_MIN:9.3f}"
              f"{a['spread']:9.3f}{b['spread']:9.3f}")
    aa = np.array([tab[1.0][0]['spread'] for _, _, _, _, _, tab in rows_out])
    bb = np.array([tab[float(s_fix)][0]['spread']
                   for _, _, _, _, _, tab in rows_out])
    dd = np.array([1e3 * (tab[float(s_fix)][0]['half'] - tab[1.0][0]['half'])
                   / W_MIN for _, _, _, _, _, tab in rows_out])
    print(f"\n  median spread {np.median(aa):.3f} -> {np.median(bb):.3f} mm,"
          f"  offset moves {np.median(np.abs(dd)):.3f} mm in the median"
          f" and {np.abs(dd).max():.3f} at worst")
    print(f"  runs inside the cap {ins_a}/140 ->"
          f" {sum(tab[float(s_fix)][0]['inside'] for _, _, _, _, _, tab in rows_out)}/140")

    print(f"\n  --- and with each configuration's own invariance optimum"
          f" ---\n")
    print(f"  The most that can be defended: K is a per-configuration"
          f" constant,\n  so let each one take the scale its own"
          f" repeatability prefers.\n")
    print(f"  {'configuration':>16}{'s':>7}{'K':>9}{'K fixed':>10}"
          f"{'offset':>10}{'shift':>9}{'spread':>9}{'spread':>9}")
    print(f"  {'':16}{'':7}{'':9}{'':10}{'mm':>10}{'mm':>9}{'s = 1':>9}"
          f"{'own s':>9}")
    off_a, off_b, sp_a, sp_b, ins_c = [], [], [], [], 0
    for cfg, k0, sr, si, sh_, tab in rows_out:
        a, b = tab[1.0][0], tab[float(si)][0]
        off_a.append(1e3 * a['half'] / W_MIN)
        off_b.append(1e3 * b['half'] / W_MIN)
        sp_a.append(a['spread'])
        sp_b.append(b['spread'])
        ins_c += b['inside']
        print(f"  {cfg[0].replace('case_', '') + '/' + cfg[1]:>16}"
              f"{si:7.2f}{k0:9.4f}{si * k0:10.4f}{off_b[-1]:10.3f}"
              f"{off_b[-1] - off_a[-1]:9.3f}{sp_a[-1]:9.3f}{sp_b[-1]:9.3f}")
    dd = np.abs(np.array(off_b) - np.array(off_a))
    print(f"\n  median spread {np.median(sp_a):.3f} ->"
          f" {np.median(sp_b):.3f} mm,  offset moves"
          f" {np.median(dd):.3f} mm in the median and {dd.max():.3f} at worst")
    print(f"  runs inside the cap {ins_a}/140 -> {ins_c}/140")
    print(f"\n  The per-configuration optima run"
          f" {min(r[3] for r in rows_out):.2f} to"
          f" {max(r[3] for r in rows_out):.2f} on fourteen runs each, which"
          f" is\n  scatter about 1, not a bias away from it.  Freezing them"
          f" would fit\n  the noise of a one-dimensional search and buy"
          f" {np.median(dd):.3f} mm.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
