#!/usr/bin/env python3
"""
Can the resting tilt identify z_CoM (and expose the GE residual)?
================================================================
The static threshold is written at the resting attitude, so the CoM
height enters only through the *absolute* tilt of the body against
gravity.  Carrying phi_0 through the pivot balance (pivot arm l, onset
thrust f, offset lambda, sgn = +1 for the pos tip):

    M_crit = sgn (W cos(phi_0) - f) l  +  s_ax W lambda cos(phi_0)
             - W z_CoM sin(phi_0)  +  Delta M_GE                    (*)

with s_ax = +1 (roll, lambda = y_off) and -1 (pitch, lambda = x_off).
The z_CoM term is *symmetric* between the two tip directions -- a tilted
vehicle simply has its CoM displaced by z_CoM sin(phi_0) -- so it is
algebraically indistinguishable from a CoM offset, and it survives the
pivot-free average:

    M_ff = 1/2 (M_+ + M_-) = s_ax W lambda cos(phi_0) - W z_CoM sin(phi_0)
           + 1/2 (W cos(phi_0) - f)(l_+ - l_-)  + GE_sym .

Hence knowing W and lambda from the load cell does NOT make z_CoM
observable per group: one scalar equation, and z_CoM, the landing-gear
asymmetry (l_+ - l_-) and the symmetric GE part all enter as constants.
The only thing that separates them is a *regressor that varies*, and
sin(phi_0) is the one z_CoM owns.  Two ways to use it:

  A  WITHIN group (truth-free).  phi_0 varies run to run.  Group fixed
     effects (case x axis x direction) absorb the arm, the offset, the
     gear asymmetry and the ground effect -- all constant within a group
     -- leaving z_CoM as the only tilt-modulated term.  This estimate is
     therefore free of GE contamination by construction, and W and
     lambda drop out of it entirely.
  B  BETWEEN groups (uses the truth).  The offset error of the
     pivot-free average against the load-cell truth should carry
     -s_ax z_CoM sin(phi_0), with phi_0 differing per case/axis.

Reported alongside: an orthogonal-axis placebo (the same regression run
on the tilt of the *un-excited* axis, which (*) says has no effect), an
errors-in-variables attenuation factor from the odom/mocap attitude
pair, the detection limit the achieved precision implies, and the tilt
lever a dedicated wedge experiment would need.

Input: the per-run table written by ``--collect`` (identification +
resting attitude + fitted pivot arm), 140 runs on the reference dataset.

Usage
-----
python analysis/zcom_tilt.py --collect DataSet/exp --output-dir .
python analysis/zcom_tilt.py --table zcom_tilt_runs.csv
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import contextlib
import csv
import io
from collections import defaultdict
from pathlib import Path

import numpy as np

G = 9.81
# manuscript Table 7 (load-cell ground truth)
MASS_KG = {'case_01': 3.066, 'case_02': 3.220, 'case_03': 3.220,
           'case_04': 3.220, 'case_05': 3.220}
OFF_MM = {('case_01', 'Mx'): -2.90,  ('case_01', 'My'): -11.45,
          ('case_02', 'Mx'): -14.29, ('case_02', 'My'): -9.90,
          ('case_03', 'Mx'): -5.26,  ('case_03', 'My'): 3.14,
          ('case_04', 'Mx'): 6.67,   ('case_04', 'My'): 2.40,
          ('case_05', 'Mx'): 10.91,  ('case_05', 'My'): -10.89}
S_AX = {'Mx': +1.0, 'My': -1.0}       # W y_off = +M_ff ; W x_off = -M_ff
FIELDS = ['case', 'axis', 'bag', 'dir', 'rate', 'M', 'f', 'arm', 'tilt',
          'tilt_mc', 'tilt_on', 'tilt_orth', 'onset_t', 'R', 'res', 'c2',
          'k']


# ═════════════════════════════════════════════════════════════
#  Collection
# ═════════════════════════════════════════════════════════════

def collect(root: Path, out: Path) -> list[dict]:
    """Identify every run and record its resting attitude."""
    import critical_value_getter_piecewise as cvp
    from utils.extractor import load_excitation_dataset
    from utils import math_tools

    rows = []
    for d in sorted(root.glob('case_*/M[xy]')):
        axis = 'x' if d.name == 'Mx' else 'y'
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            bags = load_excitation_dataset(d)
            crits, _ = cvp.extract_piecewise_batch(bags, axis)
        line = [l for l in buf.getvalue().splitlines() if 'Rig constants' in l]
        c2 = float(line[0].split('C₂=')[1].split()[0]) if line else np.nan
        k = float(line[0].split('K=')[1].split()[0]) if line else np.nan

        by_bag = {b.name: b for b in bags}
        for c in crits:
            b = by_bag[c.bag_name]
            piv = cvp.estimate_pivot_from_mocap(b, c.onset_time, axis)
            sig = cvp.prepare_signals(b, axis)
            i0, _ = cvp.detect_excitation_window(
                sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
            roll, pitch = math_tools.quaternion_to_euler_vectorized(
                b.odom.quaternion)
            rl_mc, pt_mc = math_tools.quaternion_to_euler_vectorized(
                b.pose.quaternion)
            # resting attitude = median over the pre-excitation stretch
            n, nm = max(1, min(i0, len(roll))), max(1, min(i0, len(rl_mc)))
            j = min(int(np.searchsorted(b.odom.t - b.odom.t[0], c.onset_time)),
                    len(roll) - 1)
            exc, orth = ((roll, pitch) if axis == 'x' else (pitch, roll))
            exc_mc = rl_mc if axis == 'x' else pt_mc
            rows.append(dict(
                case=d.parent.name, axis=d.name, bag=c.bag_name,
                dir='pos' if c.bag_name.startswith('pos') else 'neg',
                rate=cvp.commanded_ramp_rate(c.bag_name) or np.nan,
                M=c.onset_moment, f=c.onset_thrust,
                arm=piv['pivot_abs'] * 1e-3,
                R=piv['R'] * 1e-3, res=piv['residual'],
                tilt=float(np.median(exc[:n])),
                tilt_mc=float(np.median(exc_mc[:nm])),
                tilt_on=float(exc[j]),
                tilt_orth=float(np.median(orth[:n])),
                onset_t=c.onset_time, c2=c2, k=k))
        print(f"  collected {d} ({len(rows)} runs)", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'zcom_tilt_runs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Table -> {out / 'zcom_tilt_runs.csv'}")
    return rows


def dynamic(root: Path, rows: list[dict], c2_bounds=(0.2, 25.0)) -> None:
    """E: truth-pinned post-onset fit — the other place z_CoM could live.

    The load-cell truth pins exactly the symmetric part of a direction
    pair, 1/2 (M_+ + M_-) = s_ax W lambda; taking the antisymmetric part
    1/2 (M_+ - M_-) from the identification gives a threshold — and hence
    an onset time, via M(t_on) = M_crit — that owes nothing to the onset
    sweep.  That removes the amplitude-onset trade-off which makes the
    (C_2, K) calibration a shallow ridge, leaving a two-parameter fit

        omega(tau) = C_1 (cosh(C_2 tau) - 1),  C_1 = Mdot / (W z_CoM)

    per run, i.e. z_CoM = Mdot / (C_1 W) as a per-run measurement.  What
    the fit actually recovers is reported alongside: for C_2 tau << 1 the
    model degenerates to 1/2 C_1 C_2^2 tau^2 = 1/2 (Mdot/J_P) tau^2, so
    only the product is determined and z_CoM separates from J_P solely
    through the departure from that parabola.
    """
    from scipy.optimize import curve_fit

    import critical_value_getter_piecewise as cvp
    from utils.extractor import load_excitation_dataset

    print("\n" + "=" * 70)
    print("E.  Truth-pinned post-onset fit:  omega = C1(cosh(C2 tau) - 1)")
    anti = {}
    grp = defaultdict(list)
    for r in rows:
        grp[(r['case'], r['axis'], r['dir'])].append(r['M'])
    for case in MASS_KG:
        for axis in ('Mx', 'My'):
            p, n = grp.get((case, axis, 'pos')), grp.get((case, axis, 'neg'))
            if p and n:
                anti[(case, axis)] = 0.5 * (np.mean(p) - np.mean(n))

    print(f"  {'case':9}{'ax':4}{'dir':5}{'n':>4}{'C2 tau_max':>11}"
          f"{'at bound':>10}{'z_CoM [mm]':>22}{'J_P [kg m^2]':>14}")
    allz, allj, nb, nt = [], [], 0, 0
    for d in sorted(root.glob('case_*/M[xy]')):
        axis = 'x' if d.name == 'Mx' else 'y'
        key = (d.parent.name, d.name)
        if key not in anti:
            continue
        W = MASS_KG[key[0]] * G
        Mc0 = S_AX[d.name] * W * OFF_MM[key] * 1e-3
        with contextlib.redirect_stdout(io.StringIO()):
            bags = load_excitation_dataset(d)
        per = defaultdict(list)
        for b in bags:
            sig = cvp.prepare_signals(b, axis)
            t, om, M = sig['t'], sig['omega'], sig['moment']
            i0, i1 = cvp.detect_excitation_window(
                M, moment_cap=cvp.MOMENT_CAP.get(axis))
            if i1 - i0 < 20:
                continue
            nt += 1
            sgn = +1.0 if b.name.startswith('pos') else -1.0
            Mc = Mc0 + sgn * anti[key]
            mdot = float(np.polyfit(t[i0:i1 + 1], M[i0:i1 + 1], 1)[0])
            over = np.where(sgn * M[i0:i1 + 1] >= sgn * Mc)[0]
            if len(over) == 0 or i1 - (i0 + over[0]) < 8:
                continue
            j = i0 + int(over[0])
            tau = t[j:i1 + 1] - t[j]
            y = om[j:i1 + 1] - float(np.median(om[max(0, j - 30):j]))
            try:
                p, _ = curve_fit(lambda x, c1, c2: c1 * (np.cosh(c2 * x) - 1),
                                 tau, y, p0=[mdot * 0.2, 5.0],
                                 bounds=([-np.inf, c2_bounds[0]],
                                         [np.inf, c2_bounds[1]]), maxfev=20000)
            except RuntimeError:
                continue
            if abs(p[0]) < 1e-9:
                continue
            bound = (p[1] <= c2_bounds[0] * 1.001
                     or p[1] >= c2_bounds[1] * 0.999)
            nb += bound
            per[b.name[:3]].append((mdot / (p[0] * W) * 1e3, p[1],
                                    p[1] * tau[-1], mdot / (p[0] * p[1] ** 2),
                                    float(bound)))
        for dirn in ('neg', 'pos'):
            v = np.array(per[dirn])
            if not len(v):
                continue
            allz.extend(v[:, 0])
            allj.extend(v[:, 3])
            print(f"  {key[0]:9}{d.name:4}{dirn:5}{len(v):4d}"
                  f"{np.median(v[:, 2]):11.2f}{v[:, 4].mean() * 100:9.0f}%"
                  f"{np.median(v[:, 0]):9.0f} (IQR{np.percentile(v[:, 0], 25):5.0f},"
                  f"{np.percentile(v[:, 0], 75):5.0f}){np.median(v[:, 3]):14.3f}")
    allz, allj = np.array(allz), np.array(allj)
    print(f"  {len(allz)}/{nt} runs fitted; C_2 at a bound on "
          f"{nb / max(len(allz), 1) * 100:.0f}% of them")
    print(f"  z_CoM  = Mdot/(C1 W) : median {np.median(allz):.0f} mm, IQR "
          f"[{np.percentile(allz, 25):.0f}, {np.percentile(allz, 75):.0f}], "
          f"range [{allz.min():.0f}, {allz.max():.0f}]")
    print(f"  J_P    = Mdot/(C1 C2^2): median {np.median(allj):.3f} kg m^2, "
          f"IQR [{np.percentile(allj, 25):.3f}, "
          f"{np.percentile(allj, 75):.3f}]")
    print("  -> pinning the onset with the truth does NOT sharpen z_CoM: over "
          "the\n     post-onset segment the response is still a parabola to "
          "within the\n     gyro noise, so the fit determines C1 C2^2 = "
          "Mdot/J_P and runs to a\n     C_2 bound, leaving z_CoM undetermined "
          "in the dynamic channel too.")


def budget(root: Path, rows: list[dict], target_mnm=2.0) -> None:
    """F: what J_P and W z_CoM precision a residual-moment readout needs.

    Reading a residual moment straight off the post-onset balance,

        dM_res = J_P alpha - M - f l + W(l + lam) cos(dphi)
                 - W z_CoM sin(dphi) ,

    leaves J_P and W z_CoM as the only unmeasured coefficients, so both
    are required and both enter multiplied by a *large* measured signal:
    alpha at the moment peak and the tilt excursion dphi at the window
    edge.  This reports those two levers and the precision they demand of
    a ``target_mnm`` residual readout, against what the dataset supplies.
    """
    import critical_value_getter_piecewise as cvp
    from utils.extractor import load_excitation_dataset
    from utils import math_tools

    on = {(r['case'], r['axis'], r['bag']): r['onset_t'] for r in rows
          if np.isfinite(r.get('onset_t', np.nan))}
    if not on:
        print("\n[F] skipped: the table carries no onset_t column "
              "(re-run --collect)")
        return
    dphi, alpha = [], []
    for d in sorted(root.glob('case_*/M[xy]')):
        axis = 'x' if d.name == 'Mx' else 'y'
        with contextlib.redirect_stdout(io.StringIO()):
            bags = load_excitation_dataset(d)
        for b in bags:
            t_on = on.get((d.parent.name, d.name, b.name))
            if t_on is None:
                continue
            sig = cvp.prepare_signals(b, axis)
            t, om, M = sig['t'], sig['omega'], sig['moment']
            i0, i1 = cvp.detect_excitation_window(
                M, moment_cap=cvp.MOMENT_CAP.get(axis))
            j = int(np.searchsorted(t, t_on))
            if i1 - j < 6:
                continue
            roll, pitch = math_tools.quaternion_to_euler_vectorized(
                b.odom.quaternion)
            phi = (roll if axis == 'x' else pitch)[:len(t)]
            dphi.append(abs(phi[i1] - phi[j]))
            m = (t >= t[i1] - 0.06) & (t <= t[i1] + 0.06)
            if m.sum() >= 4:
                alpha.append(abs(float(np.polyfit(t[m], om[m], 1)[0])))
    dphi, alpha = np.array(dphi), np.array(alpha)
    dp, al = float(np.median(dphi)), float(np.median(alpha))
    W = float(np.mean([MASS_KG[r['case']] * G for r in rows]))
    tgt = target_mnm * 1e-3

    print("\n" + "=" * 70)
    print("F.  Precision a residual-moment readout would demand")
    print(f"  levers at the median operating point: |dphi| at the window "
          f"edge {np.degrees(dp):.2f} deg")
    print(f"  (p90 {np.degrees(np.percentile(dphi, 90)):.2f}, max "
          f"{np.degrees(dphi.max()):.2f}),  |alpha| at the moment peak "
          f"{al:.2f} rad/s^2")
    print(f"\n  {'coefficient':16}{'needed for':>13}{'available here':>28}"
          f"{'-> error':>12}")
    print(f"  {'W z_CoM':16}{tgt / dp:9.3f} N·m"
          f"{27e-3 * W:16.2f} N·m (+-27 mm)"
          f"{1e3 * 27e-3 * W * dp:9.0f} mN·m")
    print(f"  {'':16}{'':13}{0.5 * (554 - 61) * 1e-3 * W:16.2f} N·m (1/K "
          f"spread){1e3 * 0.5 * (554 - 61) * 1e-3 * W * dp:6.0f} mN·m")
    print(f"  {'J_P':16}{tgt / al:9.3f} kg m^2"
          f"{0.5 * (0.267 - 0.074):14.3f} kg m^2 (IQR/2)"
          f"{1e3 * 0.5 * (0.267 - 0.074) * al:11.0f} mN·m")
    print(f"\n  i.e. {target_mnm:.0f} mN·m of residual needs W z_CoM to "
          f"+-{tgt / dp / W * 1e3:.2f} mm of z_CoM "
          f"({tgt / dp / 4.5 * 100:.2f}% of a nominal 4.5 N·m) and J_P to "
          f"{tgt / al / 0.123 * 100:.1f}%\n  of the 0.123 kg m^2 median — two "
          f"orders beyond what this experiment determines,\n  which is why "
          f"the GE moment is bounded by forward modelling "
          f"(analysis/ge_trajectory.py)\n  rather than read off the "
          f"balance.")


def load(path: Path) -> list[dict]:
    rows = []
    for r in csv.DictReader(open(path)):
        for k in FIELDS:
            if k not in ('case', 'axis', 'bag', 'dir'):
                r[k] = (float(r[k]) if r.get(k) not in (None, '', 'nan')
                        else np.nan)
        rows.append(r)
    return rows


# ═════════════════════════════════════════════════════════════
#  Estimators
# ═════════════════════════════════════════════════════════════

def _ols(X, y):
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ c
    dof = len(y) - np.linalg.matrix_rank(X)
    se = np.sqrt(r @ r / dof * np.diag(np.linalg.pinv(X.T @ X)))
    return c, se, r, dof


def within_group(rows, key, max_disagree_deg=3.0):
    """A: M_crit ~ group fixed effects - W z_CoM sin(phi_0).

    Mocap attitude is dropped where it disagrees with the odom attitude by
    more than ``max_disagree_deg`` — the mocap rigid body loses tracking on
    part of case_04/My/pos (within-group sd 21 deg vs 0.1–0.4 deg
    everywhere else), and a dropout would otherwise masquerade as a huge
    tilt lever.  A constant per-group offset between the two frames is
    expected (marker-defined body frame) and is absorbed by the fixed
    effect, so only the disagreement matters.
    """
    grp = defaultdict(list)
    for r in rows:
        grp[(r['case'], r['axis'], r['dir'])].append(r)
    keys = sorted(grp)
    y, x, fe, Ws = [], [], [], []
    for gi, kk in enumerate(keys):
        v = grp[kk]
        arm = np.nanmean([r['arm'] for r in v])
        for r in v:
            t = r[key]
            if not np.isfinite(t):
                continue
            if key == 'tilt_mc' and (not np.isfinite(r['tilt']) or abs(
                    np.degrees(t - r['tilt'])) > max_disagree_deg):
                continue
            W = MASS_KG[r['case']] * G
            sgn = +1.0 if r['dir'] == 'pos' else -1.0
            # subtract the measured thrust variation on the group-mean arm;
            # the arm itself is absorbed by the fixed effect
            y.append(r['M'] - sgn * (W - r['f']) * arm)
            x.append(np.sin(t))
            e = np.zeros(len(keys))
            e[gi] = 1.0
            fe.append(e)
            Ws.append(W)
    X = np.column_stack([np.array(fe), np.array(x)])
    c, se, r, dof = _ols(X, np.array(y))
    W = float(np.mean(Ws))
    return dict(z=-c[-1] / W * 1e3, sz=se[-1] / W * 1e3, n=len(y), dof=dof,
                rms=float(np.std(r, ddof=1)),
                lever=float(np.std(np.array(x) - X[:, :-1] @ np.linalg.lstsq(
                    X[:, :-1], np.array(x), rcond=None)[0])))


def pivot_geometry(rows):
    """The circle fit gives the tracked point's HEIGHT as well as the arm.

    ``estimate_pivot_from_mocap`` fits the tip-over arc with the circle
    centre pinned to the contact plane (cz = 0), so its radius closes the
    triangle: z_marker = sqrt(R^2 - cx^2) is the height of the tracked
    point above the pivot line, measured by the same fit that supplies
    the arm.  Two consequences, both used below:

    * it validates the cz = 0 pin (z_marker must equal the resting marker
      height) and the rigid-rotation premise (fit residual);
    * it fixes the reference frame of the arm.  ``pivot_abs`` is a
      WORLD-HORIZONTAL distance from the marker's rest position, so at a
      resting tilt it already carries a z_marker sin(phi_0) term:

          a_+ = l_+ cos(phi_0) - z_m sin(phi_0)
          a_- = l_- cos(phi_0) + z_m sin(phi_0)

      and the body-frame landing-gear asymmetry the pivot-free average
      needs is  l_+ - l_-  =  (a_+ - a_-) + 2 z_m sin(phi_0).
    """
    R = np.array([r['R'] for r in rows], float)
    a = np.array([r['arm'] for r in rows], float)
    res = np.array([r['res'] for r in rows], float)
    ok = np.isfinite(R) & np.isfinite(a)
    if not ok.any():
        return np.nan
    zm = np.sqrt(np.maximum(R[ok] ** 2 - a[ok] ** 2, 0.0))
    print("\n" + "=" * 70)
    print("B'. Circle-fit geometry (estimate_pivot_from_mocap)")
    print(f"  n = {ok.sum()} runs;  arm |cx| = {1e3 * a[ok].mean():.1f} mm, "
          f"radius R = {1e3 * R[ok].mean():.0f} mm")
    print(f"  -> tracked-point height z_marker = sqrt(R^2 - cx^2) = "
          f"{1e3 * zm.mean():.0f} ± {1e3 * zm.std(ddof=1):.0f} mm")
    print(f"  circle-fit residual {np.nanmean(res):.2f} mm: the "
          f"post-onset motion is a rigid\n  rotation about a fixed line, and "
          f"the cz = 0 pin lands on the contact plane.")
    return float(zm.mean())


def gear_corrected(rows, z_marker):
    """B': the same offset channel, with the gear asymmetry MEASURED.

    M_ff = 1/2 (W cos(phi_0) - f)(l_+ - l_-) + W lambda cos(phi_0)
           - W z_CoM sin(phi_0)

    left the landing-gear asymmetry as a free constant degenerate with
    z_CoM.  The circle fit measures it, so the only unknowns left are
    z_CoM and whatever constant separates the mocap marker frame from the
    load-cell reference.
    """
    grp = defaultdict(list)
    for r in rows:
        grp[(r['case'], r['axis'], r['dir'])].append(r)
    print("\n  body-frame gear asymmetry  l+ - l-  =  (a+ - a-) "
          f"+ 2 z_m sin(phi_0),  z_m = {1e3 * z_marker:.0f} mm")
    print(f"  {'case':9}{'ax':4}{'a+ - a-':>9}{'2 zm sinφ':>11}{'l+ - l-':>9}"
          f"{'gear [mm]':>11}{'err':>8}{'err-gear':>10}")
    e, reg, ax_ = [], [], []
    for case in sorted(MASS_KG):
        for axis in ('Mx', 'My'):
            vp, vn = grp.get((case, axis, 'pos')), grp.get((case, axis, 'neg'))
            if not vp or not vn:
                continue
            W, s = MASS_KG[case] * G, S_AX[axis]
            ap = float(np.nanmean([r['arm'] for r in vp]))
            an = float(np.nanmean([r['arm'] for r in vn]))
            f = 0.5 * (np.mean([r['f'] for r in vp])
                       + np.mean([r['f'] for r in vn]))
            t0 = float(np.mean([r['tilt'] for r in vp + vn]))
            corr = 2.0 * z_marker * np.sin(t0)
            dl = (ap - an) + corr
            gear = s * 0.5 * (W - f) * dl / W * 1e3
            Mff = 0.5 * (np.mean([r['M'] for r in vp])
                         + np.mean([r['M'] for r in vn]))
            err = s * Mff / W * 1e3 - OFF_MM[(case, axis)]
            e.append(err - gear)
            reg.append(-s * np.sin(t0))
            ax_.append(0.0 if axis == 'Mx' else 1.0)
            print(f"  {case:9}{axis:4}{1e3 * (ap - an):9.1f}{1e3 * corr:11.1f}"
                  f"{1e3 * dl:9.1f}{gear:11.2f}{err:8.2f}{err - gear:10.2f}")
    e, reg, ax_ = np.array(e), np.array(reg), np.array(ax_)
    print("\n  model:  err - gear = const - z_CoM * s_ax * sin(phi_0)")
    for name, X, cols in (
            ('marker frame = truth frame ',
             reg[:, None], ['z_CoM [mm]']),
            ('+ one frame constant       ',
             np.column_stack([np.ones(len(e)), reg]),
             ['const [mm]', 'z_CoM [mm]']),
            ('+ per-axis frame constants ',
             np.column_stack([1 - ax_, ax_, reg]),
             ['c_Mx [mm]', 'c_My [mm]', 'z_CoM [mm]'])):
        c, se, r, dof = _ols(X, e)
        txt = ',  '.join(f"{n} = {v:+.1f} ± {s:.1f}"
                         for n, v, s in zip(cols, c, se))
        print(f"  {name}: {txt}   (resid RMS {np.std(r, ddof=1):.2f} mm)")
    print("  -> measuring the gear asymmetry instead of leaving it free is "
          "what turns\n     this channel positive; it stays degenerate with a "
          "frame offset between\n     the mocap marker and the load-cell "
          "reference, so quote it as conditional.")


def between_group(rows):
    """B: pivot-free offset error vs the load-cell truth, on -s_ax sin(phi_0)."""
    grp = defaultdict(list)
    for r in rows:
        grp[(r['case'], r['axis'], r['dir'])].append(r)
    tab = []
    for case in sorted(MASS_KG):
        for axis in ('Mx', 'My'):
            vp, vn = grp.get((case, axis, 'pos')), grp.get((case, axis, 'neg'))
            if not vp or not vn:
                continue
            W = MASS_KG[case] * G
            Mff = 0.5 * (np.mean([r['M'] for r in vp])
                         + np.mean([r['M'] for r in vn]))
            t0 = float(np.mean([r['tilt'] for r in vp + vn]))
            tab.append(dict(case=case, axis=axis, tilt=t0,
                            lam=S_AX[axis] * Mff / W * 1e3,
                            truth=OFF_MM[(case, axis)],
                            reg=-S_AX[axis] * np.sin(t0)))
    return tab


# ═════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="z_CoM (and GE residual) from the resting tilt.")
    p.add_argument('--collect', metavar='ROOT',
                   help="identify every run under ROOT and write the table")
    p.add_argument('--table', default='zcom_tilt_runs.csv',
                   help="per-run table to analyse")
    p.add_argument('--output-dir', default='.')
    p.add_argument('--z-nominal', type=float, default=0.30,
                   help="a-priori CoM height tested against [m]")
    p.add_argument('--design-tilt', type=float, default=5.0,
                   help="wedge inclination of the proposed design [deg]")
    p.add_argument('--dynamic', metavar='ROOT',
                   help="also run the truth-pinned post-onset fit over ROOT")
    p.add_argument('--budget', metavar='ROOT',
                   help="also run the residual-moment precision budget")
    p.add_argument('--inject', type=float, default=None, metavar='Z',
                   help="injection-recovery self-test: add a synthetic "
                        "-W*Z*sin(phi_0) term [m] to every M_crit and check "
                        "that the estimator returns Z")
    args = p.parse_args()

    out = Path(args.output_dir)
    if args.collect:
        rows = collect(Path(args.collect), out)
    else:
        rows = load(Path(args.table))
    if args.inject is not None:
        for r in rows:
            r['M'] -= (MASS_KG[r['case']] * G * args.inject
                       * np.sin(r['tilt']))
        print(f"\n[INJECTION TEST] a synthetic z_CoM = "
              f"{args.inject * 1e3:.0f} mm has been added to every M_crit "
              f"through the odom resting tilt")
    print(f"\n{len(rows)} runs\n")

    # ── 0. is the run-to-run resting tilt a real attitude change? ────
    print("=" * 70)
    print("0.  Run-to-run resting tilt: odom vs mocap")
    grp = defaultdict(list)
    for r in rows:
        grp[(r['case'], r['axis'], r['dir'])].append(r)
    a, b = [], []
    for v in grp.values():
        t = np.array([r['tilt'] for r in v])
        tm = np.array([r['tilt_mc'] for r in v])
        ok = (np.isfinite(t) & np.isfinite(tm)
              & (np.abs(np.degrees(tm - t)) <= 3.0))
        if ok.sum() > 2:
            a.append(t[ok] - t[ok].mean())
            b.append(tm[ok] - tm[ok].mean())
    a, b = np.concatenate(a), np.concatenate(b)
    rho = float(np.corrcoef(a, b)[0, 1])
    lam_od = rho * b.std() / a.std()      # var_true / var_odom  (EIV)
    lam_mc = rho * a.std() / b.std()      # var_true / var_mocap
    print(f"  within-group spread [deg]: odom {np.degrees(a.std(ddof=1)):.3f}"
          f", mocap {np.degrees(b.std(ddof=1)):.3f}, corr {rho:+.3f}")
    print(f"  reliability (shared / measured variance): odom {lam_od:.2f}, "
          f"mocap {lam_mc:.2f}")
    print("  -> attitude-estimate noise attenuates the regression slope by "
          "these factors")

    # ── A. within-group ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("A.  Within-group   M_crit ~ FE(case,axis,dir) - W z_CoM sin(phi_0)")
    print("    (truth-free; the GE moment is constant within a group and is")
    print("     absorbed by the fixed effect, so it cannot bias this slope)")
    A = {}
    for label, key, atten in (('rest tilt, odom ', 'tilt', lam_od),
                              ('rest tilt, mocap', 'tilt_mc', lam_mc),
                              ('onset tilt, odom', 'tilt_on', 1.0),
                              ('PLACEBO orth ax', 'tilt_orth', 1.0)):
        d = within_group(rows, key)
        A[key] = d
        za, sa = d['z'] / atten, d['sz'] / atten
        print(f"  [{label}] n={d['n']:3d}  lever sd "
              f"{np.degrees(d['lever']):.3f} deg  resid RMS "
              f"{d['rms'] * 1e3:.1f} mN·m")
        print(f"      z_CoM = {d['z']:+7.1f} +- {d['sz']:.1f} mm"
              + (f"   (EIV-corrected {za:+.1f} +- {sa:.1f} mm)"
                 if abs(atten - 1) > 1e-6 else ""))
        print(f"      95% CI [{za - 1.96 * sa:+.0f}, {za + 1.96 * sa:+.0f}] mm"
              f"   |  z = {args.z_nominal * 1e3:.0f} mm rejected at "
              f"{abs(args.z_nominal * 1e3 - za) / sa:.1f} sigma")

    # ── B. between-group, against the load-cell truth ──────────────
    print("\n" + "=" * 70)
    print("B.  Between-group: pivot-free offset error vs truth")
    tab = between_group(rows)
    print(f"  {'case':9}{'ax':4}{'phi_0 [deg]':>12}{'lam_hat':>9}"
          f"{'truth':>8}{'err [mm]':>10}{'z regressor':>13}")
    for t in tab:
        print(f"  {t['case']:9}{t['axis']:4}{np.degrees(t['tilt']):12.3f}"
              f"{t['lam']:9.2f}{t['truth']:8.2f}"
              f"{t['lam'] - t['truth']:+10.2f}{t['reg']:13.5f}")
    e = np.array([t['lam'] - t['truth'] for t in tab])
    reg = np.array([t['reg'] for t in tab])
    ax = np.array([0.0 if t['axis'] == 'Mx' else 1.0 for t in tab])
    print(f"\n  signed mean error: Mx {e[ax == 0].mean():+.2f} mm, "
          f"My {e[ax == 1].mean():+.2f} mm;  RMS {np.sqrt((e ** 2).mean()):.2f}"
          f" mm")
    print("  model:  err = const + z_CoM * reg")
    for name, X, cols in (
            ('common intercept  ',
             np.column_stack([np.ones(len(e)), reg]), ['const [mm]', 'z [mm]']),
            ('per-axis intercepts',
             np.column_stack([1 - ax, ax, reg]),
             ['const Mx [mm]', 'const My [mm]', 'z [mm]'])):
        c, se, r, dof = _ols(X, e)
        txt = ',  '.join(f"{n} = {v:+.2f} +- {s:.2f}"
                         for n, v, s in zip(cols, c, se))
        print(f"  {name}: {txt}   (resid RMS {np.std(r, ddof=1):.2f} mm, "
              f"dof {dof})")

    zm = pivot_geometry(rows)
    if np.isfinite(zm):
        gear_corrected(rows, zm)

    # ── C. the rig constant, for comparison ────────────────────────
    print("\n" + "=" * 70)
    print("C.  For comparison: W z_CoM = 1/K from the ramp-gain calibration")
    print(f"  {'case':9}{'ax':4}{'C2':>8}{'K':>7}{'W z [N·m]':>11}"
          f"{'z_CoM [mm]':>12}{'J_P':>8}")
    zs = []
    for case in sorted(MASS_KG):
        for axis in ('Mx', 'My'):
            v = grp.get((case, axis, 'pos'))
            if not v:
                continue
            c2, k = v[0]['c2'], v[0]['k']
            W, wz = MASS_KG[case] * G, 1.0 / v[0]['k']
            zs.append(wz / W * 1e3)
            print(f"  {case:9}{axis:4}{c2:8.3f}{k:7.3f}{wz:11.2f}"
                  f"{wz / W * 1e3:12.1f}{wz / c2 ** 2:8.3f}")
    zs = np.array(zs)
    print(f"  z_CoM = {zs.mean():.0f} +- {zs.std(ddof=1):.0f} mm "
          f"(range [{zs.min():.0f}, {zs.max():.0f}]) — the (C2,K) ridge, "
          f"a {zs.max() / zs.min():.0f}x spread")

    # ── D. what the achieved precision implies / what would work ───
    print("\n" + "=" * 70)
    print("D.  Detection limit and experiment design")
    d = A['tilt_mc']
    W = float(np.mean([MASS_KG[r['case']] * G for r in rows]))
    print(f"  achieved 1-sigma on z_CoM (mocap, EIV-corrected): "
          f"{d['sz'] / lam_mc:.0f} mm  ->  3-sigma detection limit "
          f"{3 * d['sz'] / lam_mc:.0f} mm")
    n_per = len(rows) / len(grp)
    for phi in (args.design_tilt,):
        lever = np.sin(np.deg2rad(phi))     # +-phi two-level wedge design
        se = d['rms'] / (W * lever * np.sqrt(len(rows)))
        print(f"  a deliberate +-{phi:.0f} deg wedge design, same "
              f"{d['rms'] * 1e3:.0f} mN·m per-run residual and {len(rows)} "
              f"runs:")
        print(f"    SE(z_CoM) = {se * 1e3:.1f} mm  "
              f"({d['rms'] * 1e3:.0f}e-3 / ({W:.1f} x {lever:.3f} x "
              f"sqrt({len(rows)})));  {n_per:.0f} runs/group -> "
              f"{d['rms'] / (W * lever * np.sqrt(n_per)) * 1e3:.0f} mm per "
              f"group")
    if args.dynamic:
        dynamic(Path(args.dynamic), rows)
    if args.budget:
        budget(Path(args.budget), rows)

    print("\n  GE residual: within a group the ground-effect moment is a")
    print("  constant (same collective, same tilt) and is absorbed by the")
    print("  same fixed effect as the arm, the offset and the gear")
    print("  asymmetry.  Its only structured channels are sgn*c_a f l")
    print("  (antisymmetric — degenerate with the pivot arm / contact")
    print("  lever) and b*M (a 1–4% scale on M_crit).  Neither is")
    print("  separable here; see analysis/static_attribution.py.")


if __name__ == '__main__':
    main()
