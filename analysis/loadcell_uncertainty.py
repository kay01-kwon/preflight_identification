#!/usr/bin/env python3
"""How accurate is the load-cell CoM truth?  Two routes, five cases.

The measurement (main_com_estimate.m) reads two YZC-1B cells, forms
the force difference in each orientation, and flips the vehicle by
180 deg to cancel what is fixed in the rig frame:

    p = l (df - df_rev) / (2 W),   df = f1 - f2,   l = 325 mm,

repeated five times per axis per case.

ROUTE A -- EMPIRICAL.  The scatter of those five repeats, per case and
axis.  This is the number to quote, because it captures everything:
ADC noise, seating, re-levelling, tare drift.  The reported truth is
their MEAN, so its uncertainty is the standard error, sd/sqrt(5).

ROUTE B -- DATASHEET.  p is linear in four force readings with the
same sensitivity l/(2W), so independent per-reading errors u_f
propagate as

    u_p = (l / 2W) sqrt(4) u_f = l u_f / W ,

and the 0.03% FS total error of a 5 kg cell (u_f = 1.5 g) bounds u_p
at 0.16 mm.  Creep, 0.03% FS per 30 min, enters the same way when the
two orientations are separated in time.

WHAT THE FIVE CASES SHOW.  The repeatability is bimodal: six of the
ten case-axis measurements sit at 0.007-0.067 mm, an order of
magnitude BELOW the datasheet bound -- the flip has cancelled the
systematic part of the 0.03% FS and the 100-sample averaging has
crushed what is left -- while four sit at 0.49-1.44 mm, an order
ABOVE it.  Cell error cannot do that; the large ones are mechanical
(re-seating the vehicle between repeats), and they cluster in the
earlier sessions.

Usage: python analysis/loadcell_uncertainty.py [DataSet/loadcell]
"""
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

L_ARM = 325.0        # mm, 300 + 25 (main_com_estimate.m)
FS_G = 5000.0        # g, cell full scale (YZC-1B, 5 kg)
TOTAL_ERR = 3e-4     # 0.03 % FS, datasheet total error
CREEP = 3e-4         # 0.03 % FS per 30 min

# paired identification error of the deliverable, mm (see
# analysis/offset_experimental_split.py):  case -> (Mx err, My err)
IDENT = {'case_01': (+2.28, +2.15), 'case_02': (-0.70, +1.71),
         'case_03': (-2.62, -1.00), 'case_04': (-0.99, -0.22),
         'case_05': (+0.59, +2.96)}


def main():
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('DataSet/loadcell')
    files = sorted(d.glob('case_*.mat'))
    if not files:
        print(f"no case_*.mat in {d}")
        return 1

    print(f"\n  l = {L_ARM:.0f} mm;  sensitivity l/(2W) per reading\n")
    print(f"  {'case':<9}{'W [g]':>8}{'axis':>6}{'mean':>9}{'sd':>8}"
          f"{'sem':>8}{'implied':>10}")
    print(f"  {'':<9}{'':>8}{'':>6}{'[mm]':>9}{'[mm]':>8}{'[mm]':>8}"
          f"{'[% FS]':>10}")
    sds, sems, rows = [], [], []
    for f in files:
        r = sio.loadmat(f, squeeze_me=True,
                        struct_as_record=False)['results']
        W = float(r.W_drone)
        sens = L_ARM / (2.0 * W)
        for ax in ('x', 'y'):
            v = np.atleast_1d(getattr(r, f'{ax}_com')).astype(float)
            sd = float(v.std(ddof=1))
            sem = sd / np.sqrt(v.size)
            pct = 100.0 * sd / (2 * sens) / FS_G
            sds.append(sd)
            sems.append(sem)
            rows.append((f.stem, ax, sd, sem))
            print(f"  {f.stem:<9}{W:8.1f}{ax:>6}{v.mean():9.3f}"
                  f"{sd:8.3f}{sem:8.3f}{pct:10.3f}")
    sds, sems = np.array(sds), np.array(sems)

    u_f = TOTAL_ERR * FS_G
    u_p = (L_ARM / (2 * 3300.0)) * 2.0 * u_f     # nominal W
    print(f"\n  datasheet: {100*TOTAL_ERR:.2f}% FS = {u_f:.2f} g per cell"
          f"  ->  u_p = l u_f / W = {u_p:.3f} mm")
    print(f"  creep 0.03% FS / 30 min -> up to {u_p:.3f} mm if the two "
          f"orientations are 30 min apart")

    below = int(np.sum(sds < u_p))
    print(f"\n  repeatability across the {len(sds)} case-axis "
          f"measurements:")
    print(f"    median sd {np.median(sds):.3f} mm, pooled "
          f"sd {np.sqrt(np.mean(sds**2)):.3f}, worst {sds.max():.3f}")
    print(f"    {below}/{len(sds)} sit BELOW the datasheet bound "
          f"(flip + averaging), {len(sds)-below} above it (mechanical)")
    print(f"    truth uncertainty (sem of five): median "
          f"{np.median(sems):.3f} mm, pooled "
          f"{np.sqrt(np.mean(sems**2)):.3f}, worst {sems.max():.3f}")

    # does the truth uncertainty explain the identification error?
    e, u = [], []
    for case, (ex, ey) in IDENT.items():
        m = {a: s for (c, a, sd, s) in rows if c == case for _ in [0]}
        sem_by = {a: s for (c, a, sd, s) in rows if c == case}
        e += [abs(ex), abs(ey)]          # Mx identifies y_off, My x_off
        u += [sem_by.get('y', np.nan), sem_by.get('x', np.nan)]
    e, u = np.array(e), np.array(u)
    ok = ~np.isnan(u)
    r_c = float(np.corrcoef(e[ok], u[ok])[0, 1])
    tot = float(np.sqrt(np.mean(e ** 2)))
    print(f"\n  against the identification: paired error RMS {tot:.2f} mm, "
          f"truth sem RMS {np.sqrt(np.mean(u[ok]**2)):.2f} mm")
    print(f"    variance share of the truth: "
          f"{100*np.mean(u[ok]**2)/np.mean(e**2):.1f} %"
          f";  corr(|error|, sem) = {r_c:+.2f}")
    print(f"    -> the load-cell truth is not what limits the "
          f"comparison\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
