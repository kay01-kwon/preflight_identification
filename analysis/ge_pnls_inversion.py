#!/usr/bin/env python3
"""The dynamic GE inversion with the PNLS-fitted parameters -- no
differentiator at all.

Instead of numerically differentiating the gyro, omega_dot is taken
ANALYTICALLY from the fitted branch: with the dataset-calibrated C2
and the per-run amplitude refit (C1, baseline c, exactly the
manuscript's family (2)),

    omega_dot(tau) = C1 C2 sinh(C2 tau),

so the inversion has no filter window, no half-width exclusion, no
noise gain, and no 5.5 Hz band -- everything measured (phi, M, f)
enters raw, and only the smooth acceleration comes from the fit.

TWO INERTIAS, AND THE EMPIRICAL VERDICT (the reverse of the naive
expectation):

* CAD (physical): J_P from the parallel axis with the CoM offset.
  THIS is the variant that closes on the interference model: median
  d = +32.5 mN.m (IQR -49..+81, RMS 111) of a 157 mN.m level,
  |d| < 100 on 95/140 -- the same quality as the deployed SG check,
  but with no filter, no exclusion, and the full window.  The
  residual slope census is median -17.7 mN.m/deg (IQR -51..+22,
  83/140 negative): a 2.3x milder reading than the SG route's -41.5,
  with the model's -1.2 inside the IQR.

* SELF-CONSISTENT: J_P = W z / C2_cal^2, the inertia the calibrated
  exponent implies.  It looked like the tautologically clean choice,
  but empirically it is the WORSE one (RMS 229, 42/140): the
  1/C2^2 factor imports the dataset-level C2-calibration variability
  (3.5-8.0 rad/s for a geometry that predicts 4.95-5.17) straight
  into the inertia, and the per-run C1 refit cannot cancel it.
  Kept as the contrast that localises the anomaly of
  ge_dynamics_check.py in the exponent calibration, not the balance.

This is a CONSISTENCY test of the identified dynamics plus the
interference GE model -- omega_dot comes from the fitted branch, so
it is not an independent inertia measurement and is reported as
such.

Usage: PYTHONPATH=<stubs> python analysis/ge_pnls_inversion.py [out.png]
"""
import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import critical_value_getter_piecewise as cvp             # noqa: E402
from utils.extractor import load_excitation_dataset       # noqa: E402
from utils import math_tools                              # noqa: E402
from error_budget import ge_moment, LP                    # noqa: E402
from analysis.ge_dynamics_check import (                  # noqa: E402
    MASS_KG, G, OFF_SIGN, OFF_MM, J_COM)

Z = 0.261


def collect():
    root = Path(_HERE).resolve().parents[0] / 'DataSet' / 'exp'
    rows = []
    for d in sorted(root.glob('case_*/M[xy]')):
        axis = 'x' if d.name == 'Mx' else 'y'
        case, axname = d.parent.name, d.name
        mass = MASS_KG[case]
        W = mass * G
        off_truth = OFF_SIGN[axname] * OFF_MM[(case, axname)] * 1e-3
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                bags = load_excitation_dataset(d)
                c2, kg = cvp.estimate_rig_constants(bags, axis)
                crits, _ = cvp.extract_piecewise_batch(
                    bags, axis, cosh_c2=c2, ramp_gain=kg)
        except Exception as exc:                           # noqa: BLE001
            print(f"  {case}/{axname}: skipped ({type(exc).__name__})")
            continue
        by = {b.name: b for b in bags}
        cache = {}
        for crit in crits:
            bag = by.get(crit.bag_name)
            if bag is None:
                continue
            if crit.bag_name not in cache:
                with contextlib.redirect_stdout(io.StringIO()):
                    sig = cvp.prepare_signals(bag, axis)
                roll, pitch = math_tools.quaternion_to_euler_vectorized(
                    bag.odom.quaternion)
                i0w, _ = cvp.detect_excitation_window(
                    sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
                qr = np.median(bag.odom.quaternion[:max(20, i0w)], axis=0)
                qr = qr / np.linalg.norm(qr)
                cache[crit.bag_name] = (
                    sig, roll if axis == 'x' else pitch, qr)
            sig, phi_all, qr = cache[crit.bag_name]
            n = min(len(phi_all), len(sig['t']))
            _, i1 = cvp.detect_excitation_window(
                sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
            j = crit.onset_idx
            i1 = min(i1, n - 1)
            if i1 - j < 20:
                continue
            sl = slice(j, i1 + 1)
            pos = crit.bag_name.startswith('pos')
            s = 1.0 if pos else -1.0
            piv = cvp.estimate_pivot_from_mocap(bag, crit.onset_time, axis)
            lp = (piv['pivot_abs'] * 1e-3
                  if not np.isnan(piv['pivot_abs']) else LP[axis])
            a = lp + s * off_truth
            tau = sig['t'][sl] - sig['t'][j]
            phi = s * (phi_all[sl] - phi_all[j])
            m = s * sig['moment'][sl]
            f = sig['f_col'][sl]
            om = s * np.asarray(sig['omega'][:n], float)[sl]
            # per-run amplitude refit, C2 pinned: family (2)
            u = np.cosh(np.clip(c2 * tau, 0, 30)) - 1.0
            A = np.column_stack([u, np.ones_like(u)])
            (c1_fit, _c0), *_ = np.linalg.lstsq(A, om, rcond=None)
            if c1_fit <= 0:
                continue
            om_dot = c1_fit * c2 * np.sinh(np.clip(c2 * tau, 0, 30))
            grav = W * a * np.cos(phi) - W * Z * np.sin(phi)
            base = -m - f * lp + grav
            jp_eff = W * Z / c2 ** 2
            jp_cad = J_COM[axis] + mass * (Z ** 2 + a ** 2)
            raw = ge_moment(bag, sig, axis, n, pos, q_rest=qr, window=sl)
            if raw is None:
                continue
            gm = s * raw[sl]
            rows.append(dict(
                case=case, axisname=axname, bag=crit.bag_name,
                mism=float(jp_cad * c2 ** 2 / (W * Z) - 1.0),
                phi=np.rad2deg(phi),
                eff=1e3 * (jp_eff * om_dot + base),
                cad=1e3 * (jp_cad * om_dot + base),
                mod=1e3 * gm))
        print(f"  {case}/{axname}: done")
    return rows


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'ge_pnls_inversion.png'
    rows = collect()
    n = len(rows)
    d_eff = np.array([float(np.mean(r['eff'] - r['mod'])) for r in rows])
    d_cad = np.array([float(np.mean(r['cad'] - r['mod'])) for r in rows])
    # onset-anchored dynamic increment: subtract each trace's own onset
    # level (mean of the first 5 samples) from BOTH sides, so the static
    # repeatability floor -- which the static check tests separately --
    # cancels and only the tilt-driven increment is compared
    d_anc = np.array([float(np.mean(
        (r['cad'] - np.mean(r['cad'][:5])) -
        (r['mod'] - np.mean(r['mod'][:5])))) for r in rows])
    s_eff = np.array([np.polyfit(r['phi'], r['eff'] - r['mod'], 1)[0]
                      for r in rows])
    s_cad = np.array([np.polyfit(r['phi'], r['cad'] - r['mod'], 1)[0]
                      for r in rows])
    lvl = np.median([float(np.mean(r['mod'])) for r in rows])

    def stats(v, lab):
        print(f"  {lab:<22} median {np.median(v):+7.1f}  IQR "
              f"[{np.percentile(v,25):+7.1f},{np.percentile(v,75):+7.1f}]"
              f"  RMS {np.sqrt(np.mean(v**2)):6.1f}  "
              f"|d|<100: {int(np.sum(np.abs(v)<100))}/{len(v)}")

    print(f"\n  {n} runs, analytic omega_dot from the PNLS fit "
          f"(no filter, no trim)\n")
    stats(d_eff, 'self-consistent J_P')
    stats(d_cad, 'CAD parallel-axis J_P')
    stats(d_anc, 'CAD, onset-anchored')
    print(f"  onset-anchored |d|<50: "
          f"{int(np.sum(np.abs(d_anc)<50))}/{len(d_anc)}")
    print(f"  model level: median {lvl:.1f} mN.m")
    print(f"  residual slope (self-consistent): median "
          f"{np.median(s_eff):+.1f} mN.m/deg, IQR "
          f"[{np.percentile(s_eff,25):+.1f},{np.percentile(s_eff,75):+.1f}]")
    print(f"  residual slope (CAD J_P)        : median "
          f"{np.median(s_cad):+.1f} mN.m/deg, IQR "
          f"[{np.percentile(s_cad,25):+.1f},{np.percentile(s_cad,75):+.1f}],"
          f" {int(np.sum(s_cad < 0))}/{len(s_cad)} negative")
    # where the run-to-run scatter comes from
    mism = np.array([r['mism'] for r in rows])
    cc = float(np.corrcoef(mism, d_cad)[0, 1])
    print(f"\n  scatter attribution: corr(d, J_P C2^2/Wz - 1) = {cc:+.2f}")
    print(f"  {'dataset':<14}{'C2 mism':>9}{'median d':>10}")
    seen = []
    for r in rows:
        key = (r['case'], r['axisname'])
        if key in seen:
            continue
        seen.append(key)
        ds = [q for q in rows
              if (q['case'], q['axisname']) == key]
        dd = np.median([float(np.mean(q['cad'] - q['mod'])) for q in ds])
        print(f"  {key[0]+'/'+key[1]:<14}{ds[0]['mism']:+9.2f}{dd:+10.1f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.9))
    fig.subplots_adjust(left=0.065, right=0.99, top=0.79, bottom=0.115,
                        wspace=0.24)

    PH = np.concatenate([r['phi'] for r in rows])
    GE = np.concatenate([r['cad'] for r in rows])
    GM = np.concatenate([r['mod'] for r in rows])
    bins = np.arange(0.0, PH.max() + 0.25, 0.25)
    cx, md, q1, q3, p10, p90, mm, m1, m3 = ([] for _ in range(9))
    for b0, b1 in zip(bins[:-1], bins[1:]):
        k = (PH >= b0) & (PH < b1)
        if k.sum() < 50:
            continue
        cx.append(0.5 * (b0 + b1))
        md.append(np.median(GE[k])); q1.append(np.percentile(GE[k], 25))
        q3.append(np.percentile(GE[k], 75))
        p10.append(np.percentile(GE[k], 10)); p90.append(np.percentile(GE[k], 90))
        mm.append(np.median(GM[k])); m1.append(np.percentile(GM[k], 10))
        m3.append(np.percentile(GM[k], 90))
    a1.fill_between(cx, p10, p90, color='#7b3294', alpha=0.12, lw=0)
    a1.fill_between(cx, q1, q3, color='#7b3294', alpha=0.25, lw=0)
    a1.plot(cx, md, '-', color='#7b3294', lw=2.0,
            label=f'PNLS-$\\dot\\omega$ inversion, CAD $J_P$ '
                  f'({n} runs): median, IQR, 10-90%')
    a1.fill_between(cx, m1, m3, color='#e08214', alpha=0.35, lw=0)
    a1.plot(cx, mm, '-', color='#e08214', lw=2.0,
            label='rotor-interference model: median, 10-90%')
    a1.axhline(0, color='0.5', lw=0.8)
    a1.set_xlim(0.0, 5.2)
    a1.set_xlabel(r'excursion $\delta\varphi$ [deg]', fontsize=10)
    a1.set_ylabel(r'$\Delta M_{GE}$ [mN$\cdot$m]', fontsize=10)
    a1.set_title('(a) analytic $\\dot\\omega$ from the fitted branch:\n'
                 'no filter, no exclusion, full window', fontsize=11)
    a1.legend(fontsize=8.5, loc='upper left')
    a1.grid(alpha=0.22, lw=0.4)

    hb = np.linspace(-400, 400, 33)
    a2.hist(np.clip(d_cad, hb[0], hb[-1]), bins=hb, color='0.75', alpha=0.9,
            label=f'CAD $J_P$  (RMS {np.sqrt(np.mean(d_cad**2)):.0f})')
    a2.hist(np.clip(d_anc, hb[0], hb[-1]), bins=hb, color='#148f77',
            alpha=0.75,
            label=f'CAD, onset-anchored  '
                  f'(RMS {np.sqrt(np.mean(d_anc**2)):.0f})')
    a2.axvline(0, color='#e08214', lw=2.0, label='model')
    a2.axvline(float(np.median(d_cad)), color='0.35', lw=1.2, ls='--',
               label=f'CAD median {np.median(d_cad):+.0f}')
    a2.set_xlabel(r'per-run mean of $\Delta M_{GE}^{dyn} - \Delta M_{GE}'
                  r'^{model}$ [mN$\cdot$m]', fontsize=10)
    a2.set_ylabel('runs', fontsize=10)
    a2.set_title('(b) per-run agreement: raw level (grey) carries the\n'
                 'static floor; the onset-anchored increment (green) '
                 'halves it', fontsize=11)
    a2.legend(fontsize=8.5, loc='upper left')
    a2.grid(alpha=0.22, lw=0.4, axis='y')

    fig.suptitle('Dynamic GE inversion with the PNLS-fitted parameters '
                 r'($\dot\omega = C_1C_2\sinh C_2\tau$; '
                 r'$z_{CoM} = 0.261$ m)', fontsize=12, y=0.965)
    fig.savefig(out, dpi=150)
    print(f"\n  wrote {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
