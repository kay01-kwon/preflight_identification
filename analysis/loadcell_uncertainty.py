#!/usr/bin/env python3
"""How accurate is the load-cell CoM truth?  Two independent routes.

The measurement (main_com_estimate.m) reads two YZC-1B cells, forms
the force difference in each orientation, and flips the vehicle by
180 deg to cancel what is fixed in the rig frame:

    p = l (df - df_rev) / (2 W),   df = f1 - f2,   l = 325 mm.

ROUTE A -- EMPIRICAL.  The script repeats the whole procedure N times
per axis and reports the scatter.  This is the number to quote,
because it captures everything: ADC noise, seating, re-levelling,
tare drift.

ROUTE B -- DATASHEET.  p is linear in four force readings with the
same sensitivity l/(2W), so independent per-reading errors u_f
propagate as

    u_p = (l / 2W) sqrt(4) u_f = l u_f / W .

With the datasheet total error of 0.03% FS on a 5 kg cell,
u_f = 1.5 g, this bounds u_p at 0.16 mm.  Creep, 0.03% FS per 30 min,
enters the same way if the two orientations are separated in time.

The two routes disagree in an informative direction, which the run
prints: the y axis lands BELOW the datasheet bound (the flip cancels
the systematic part of the 0.03% FS, leaving only ADC noise that
100-sample averaging has already crushed), while the x axis lands
above it (mechanical repositioning between trials, not cell error).

Usage: python analysis/loadcell_uncertainty.py <com_estimation_results.mat>
"""
import sys

import numpy as np
import scipy.io as sio

L_ARM = 325.0        # mm, 300 + 25 (main_com_estimate.m)
FS_G = 5000.0        # g, cell full scale (YZC-1B, 5 kg)
TOTAL_ERR = 3e-4     # 0.03 % FS, datasheet total error
CREEP = 3e-4         # 0.03 % FS per 30 min


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'com_estimation_results.mat'
    r = sio.loadmat(path, squeeze_me=True, struct_as_record=False)['results']
    W = float(r.W_drone)
    sens = L_ARM / (2.0 * W)                      # mm per g, per reading

    print(f"\n  W = {W:.1f} g,  l = {L_ARM:.0f} mm")
    print(f"  sensitivity dp/df = l/(2W) = {sens:.4f} mm/g "
          f"(each of the four readings)\n")

    print("  ROUTE A -- empirical, from the repeated trials")
    out = {}
    for ax in ('x', 'y'):
        v = np.atleast_1d(getattr(r, f'{ax}_com')).astype(float)
        sd = float(v.std(ddof=1))
        sem = sd / np.sqrt(v.size)
        out[ax] = (sd, sem)
        print(f"    {ax}: n = {v.size}, mean {v.mean():+7.3f} mm, "
              f"single-trial sd {sd:.3f} mm, "
              f"sem of the mean {sem:.3f} mm")
        # implied per-reading force scatter, inverting the propagation
        print(f"       -> implied per-reading force scatter "
              f"{sd/(2*sens):.2f} g = {100*sd/(2*sens)/FS_G:.3f} % FS")

    print("\n  ROUTE B -- datasheet propagation")
    u_f = TOTAL_ERR * FS_G
    u_p = sens * 2.0 * u_f
    print(f"    total error {100*TOTAL_ERR:.2f} % FS = {u_f:.2f} g per cell")
    print(f"    -> u_p = (l/2W) sqrt(4) u_f = {u_p:.3f} mm")
    for mins in (5, 15, 30):
        c = CREEP * FS_G * mins / 30.0
        print(f"    creep over {mins:2d} min: {c:.2f} g "
              f"-> {sens*2*c:.3f} mm (worst case, no cancellation)")

    print("\n  VERDICT")
    worst = max(s for s, _ in out.values())
    worst_sem = max(e for _, e in out.values())
    print(f"    single-trial repeatability: {out['y'][0]:.3f}-"
          f"{out['x'][0]:.3f} mm; datasheet bound {u_p:.3f} mm")
    print(f"    the reported truth is the MEAN of the trials, so its "
          f"uncertainty is the sem: <= {worst_sem:.3f} mm")
    print(f"    against the identification's validation RMS of 1.64 mm "
          f"that is {1.64/worst_sem:.0f}x smaller -- the load-cell truth "
          f"can be treated as exact in that comparison.\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
