#!/usr/bin/env python3
"""The small-angle terms in the STATIC equilibrium, bounded.

A reviewer asks for the approximation error of the small-angle
trigonometry in the ground static roll/pitch equilibrium, "under real
test conditions (up to 5 degrees rotation)", and specifically for
justification of neglecting sin(phi).

The premise needs separating before the question can be answered,
because two different angles are being conflated.

  phi_exc   the tilt the vehicle reaches DURING the excitation, up to
            5-10 deg.  The static equilibrium is not evaluated there.
            The critical moment is defined at the instant the vehicle is
            still flat on the ground with every landing gear in contact,
            where phi = 0 identically and the trigonometry is exact, not
            approximated.  What the excursion does affect is how well
            that instant is IDENTIFIED, and that is the deviation
            analysis of Sec. VI-E, bounded at (108) and validated
            forwards on the campaign.

  phi_0     the tilt of the body relative to GRAVITY when it is resting
            with every gear in contact -- floor slope plus landing-gear
            unevenness.  This one does enter the static equilibrium, at
            FIRST order, and it is what the reviewer's concern amounts
            to once it is put in the right place.

For the second, work in the ground-plane frame, pivot at (+l_p, 0), CoM
at (a, z).  Gravity is tilted by phi_0, so its direction is
(sin phi_0, -cos phi_0) and the restoring moment about each edge is

    M_+ = + [ W A_+ cos(phi_0) - W z sin(phi_0) ],   A_+ = l_p - a
    M_- = - [ W A_- cos(phi_0) + W z sin(phi_0) ],   A_- = l_p + a

The thrust does not appear: it is body-fixed in direction and its point
of application rotates with the body, so its moment about the pivot is
invariant under phi_0 and cancels out of everything below.

Two consequences, and they differ in order.

    half-sum  ->  a_hat = a cos(phi_0) + z sin(phi_0)
                  a_hat - a  ~=  z phi_0                    FIRST order
    difference ->  M_+ - M_-  =  W (A_+ + A_-) cos(phi_0)
                  the z terms CANCEL exactly, so
                  dW/W = cos(phi_0) - 1  ~=  -phi_0^2/2     SECOND order

So the offset carries the whole first-order sensitivity, with gain
exactly z_CoM, and the weight of (34) carries none of it.  The arm
(cosine) terms the reviewer also mentions are second order everywhere.

Usage: python analysis/static_tilt_sensitivity.py [DATASET_ROOT]
"""
import contextlib
import io
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

W_UNLOADED = 30.08                     # N, the weight the offset divides by
Z_BOX = (0.20, 0.30)                   # m, z_CoM admissible range
A_TYP = 0.0020                         # m, a representative offset
LP = 0.160                             # m, half landing-gear span
BUDGET_MM = 0.400                      # mm, the Sec. VI-E offset bound


def offset_error(phi0, z, a=A_TYP):
    """a_hat - a, exactly: a (cos-1) + z sin."""
    return a * (np.cos(phi0) - 1.0) + z * np.sin(phi0)


def weight_error(phi0):
    """dW/W from the difference channel, exactly: cos(phi_0) - 1."""
    return np.cos(phi0) - 1.0


def rp_from_quat(q):
    """Roll and pitch from [qw, qx, qy, qz], body -> world."""
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))
    return roll, pitch


def measured(root, cache):
    """Pre-onset roll and pitch across the campaign, one value per run."""
    if os.path.exists(cache):
        with open(cache, 'rb') as fh:
            return pickle.load(fh)
    import critical_value_getter_piecewise as cvp
    from utils.extractor import load_excitation_dataset
    from pathlib import Path
    rows = []
    for case in sorted(Path(root).glob('case_*')):
        for ad in ('Mx', 'My'):
            d = case / ad
            if not d.is_dir():
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                bags = load_excitation_dataset(d)
            axis = 'x' if ad == 'Mx' else 'y'
            for bag in bags:
                if cvp.commanded_ramp_rate(bag.name) is None:
                    continue
                with contextlib.redirect_stdout(io.StringIO()):
                    sig = cvp.prepare_signals(bag, axis)
                i0, _ = cvp.detect_excitation_window(
                    sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
                # the settled attitude before anything is commanded
                t_on = sig['t'][i0]
                tq = bag.odom.t - bag.odom.t[0]
                pre = tq < t_on
                if pre.sum() < 20:
                    continue
                roll, pitch = rp_from_quat(bag.odom.quaternion[pre])
                rows.append(dict(case=case.name, axis=ad, name=bag.name,
                                 roll=float(np.median(roll)),
                                 pitch=float(np.median(pitch))))
            print(f"  {case.name}/{ad}: {len(rows)} runs so far")
    with open(cache, 'wb') as fh:
        pickle.dump(rows, fh)
    return rows


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else 'DataSet/exp'
    print(__doc__[__doc__.index('For the second'):__doc__.index('Usage:')])

    print("  the error against the tilt, exactly (no expansion)\n")
    print(f"  {'phi_0':>7}{'offset err, z=0.20':>21}{'offset err, z=0.30':>21}"
          f"{'dW/W':>11}{'first order':>14}")
    print(f"  {'deg':>7}{'mm':>21}{'mm':>21}{'%':>11}{'error':>14}")
    for deg in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0):
        p = np.deg2rad(deg)
        e20, e30 = offset_error(p, 0.20), offset_error(p, 0.30)
        fo = 0.30 * p
        rel = abs(e30 - fo) / abs(e30) if e30 else 0.0
        print(f"  {deg:7.2f}{1e3*e20:21.4f}{1e3*e30:21.4f}"
              f"{100*weight_error(p):11.4f}{rel:13.2%}")

    print(f"\n  the levelling the {BUDGET_MM} mm budget of (108) implies\n")
    for z in Z_BOX:
        phi = np.arcsin(BUDGET_MM * 1e-3 / z)
        print(f"    z_CoM = {z:.2f} m  ->  |phi_0| <= {np.rad2deg(phi):.4f} deg"
              f"  ({1e3*2*LP*np.sin(phi):.3f} mm across the gear span)")

    print(f"\n  and the tilt at which the WEIGHT channel would matter"
          f" as much:")
    phi = np.arccos(1 - BUDGET_MM * 1e-3 / (2 * LP))
    print(f"    dW/W reaching the same {BUDGET_MM} mm needs"
          f" |phi_0| = {np.rad2deg(phi):.2f} deg -- two orders further out,")
    print(f"    which is what 'second order' buys.\n")

    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '.tilt_cache.pkl')
    try:
        rows = measured(root, cache)
    except Exception as exc:                       # dataset not present
        print(f"  [measured attitude unavailable: {exc}]")
        return 0
    if not rows:
        print("  [no runs found]")
        return 0

    print(f"\n  measured pre-onset attitude, {len(rows)} runs\n")
    print(f"  {'case/axis':>14}{'n':>4}{'roll med':>11}{'roll p95':>10}"
          f"{'pitch med':>11}{'pitch p95':>11}{'worst |tilt|':>14}")
    print(f"  {'':14}{'':4}{'deg':>11}{'deg':>10}{'deg':>11}{'deg':>11}"
          f"{'as offset, mm':>14}")
    keys = sorted({(r['case'], r['axis']) for r in rows})
    worst = 0.0
    for k in keys:
        v = [r for r in rows if (r['case'], r['axis']) == k]
        ro = np.rad2deg([r['roll'] for r in v])
        pi = np.rad2deg([r['pitch'] for r in v])
        rel = np.abs(ro) if k[1] == 'Mx' else np.abs(pi)
        wm = 1e3 * offset_error(np.deg2rad(np.percentile(rel, 95)), 0.30, 0.0)
        worst = max(worst, wm)
        print(f"  {k[0] + '/' + k[1]:>14}{len(v):4d}"
              f"{np.median(ro):11.4f}{np.percentile(np.abs(ro), 95):10.4f}"
              f"{np.median(pi):11.4f}{np.percentile(np.abs(pi), 95):11.4f}"
              f"{wm:14.4f}")
    allr = np.rad2deg([r['roll'] for r in rows])
    allp = np.rad2deg([r['pitch'] for r in rows])
    print(f"\n  campaign: |roll| p95 = {np.percentile(np.abs(allr), 95):.4f}"
          f" deg, |pitch| p95 = {np.percentile(np.abs(allp), 95):.4f} deg")
    print(f"  worst per-configuration equivalent offset:"
          f" {worst:.3f} mm at z = 0.30")
    print(f"  against the {BUDGET_MM} mm bound of (108) and the 1.64 mm"
          f" validation RMS.")
    print(f"\n  The tilt is OBSERVED, not assumed, so it is correctable:")
    print(f"  a_corrected = a_hat - z_CoM sin(phi_0) with phi_0 the")
    print(f"  pre-onset attitude above.  What is left after correcting is")
    print(f"  z_CoM times the attitude-estimate error, not the tilt itself.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
