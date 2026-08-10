#!/usr/bin/env python3
"""Polynomial order against filter cutoff: which free choice matters less?

Both routes to omega_dot leave one knob.  The polynomial leaves the
order K, the filter leaves the cutoff fc.  A knob is acceptable if the
answer barely moves across the range where the knob is defensible.

The verdict, on this dataset: the within-run attitude slope moves by
20.9 mN.m/deg across K = 5, 6, 7 but only 8.3 across fc = 4-20 Hz for
bw and 10.4 for bwk.  That is the quantity the polynomial route could
not report, since its spread exceeded the model slope of -1.9 several
times over.  The symmetric residual behaves the other way for bw --
-36 to -9 across the cutoff -- but bwk holds it to -38 .. -29, which is
why phi must come from the integral of the same filtered rate rather
than from the attitude.

It also shows that K = 6 was a lucky draw: it reads -9.8 where K = 5
and 7 read -30.7 and -24.4, and every filter setting reads -18 to -30.
The attitude dependence does NOT match the model's -1.9; the level
does.

Prepare the dumps first, one per setting:

  for k in 5 6 7; do HD_DERIV=polyk:$k HD_GAIN=0.890 \
      HD_DUMP=hd_polyk$k.npz python analysis/heave_damping.py; done
  for fc in 4 6 8 12 20; do for v in bw bwk; do HD_DERIV=$v:$fc \
      HD_GAIN=0.890 HD_DUMP=hd_$v$fc.npz \
      python analysis/heave_damping.py; done; done

  python analysis/deriv_knob_sweep.py DIR
"""
import sys
from pathlib import Path

import numpy as np

SP = Path(sys.argv[1])
BAND = 0.4


def stats(f):
    d = np.load(f)
    rid, phi = d['rid'], d['phi']
    res = d['resid']
    inv, mod = res + d['model'], d['model']
    grp = np.array([f"{c}/{a}/{t}"
                    for c, a, t in zip(d['case'], d['axis'], d['tip'])])
    cases = sorted(set(d['case']))

    def on(g, y):
        m = (grp[rid] == g) & (phi >= 0) & (phi < BAND)
        return np.median(y[m]) if m.sum() else np.nan

    sym, anti = [], []
    for c in cases:
        for ax in ('Mx', 'My'):
            p, n = on(f'{c}/{ax}/pos', res), on(f'{c}/{ax}/neg', res)
            sym.append(0.5 * (p + n))
            anti.append(0.5 * (p - n))
    sym, anti = np.array(sym), np.array(anti)

    # per-run within-run slope, the other quantity that moved with K
    si = []
    for i in range(len(d['mdot'])):
        s_ = rid == i
        if np.ptp(phi[s_]) < 0.2:
            continue
        si.append(np.polyfit(phi[s_], inv[s_], 1)[0])
    lvl = np.median([on(g, inv) for g in np.unique(grp)])
    mlv = np.median([on(g, mod) for g in np.unique(grp)])
    return dict(level=lvl / mlv, sym_med=np.median(sym),
                sym_rms=np.sqrt(np.mean(sym ** 2)),
                anti_rms=np.sqrt(np.mean(anti ** 2)),
                slope=np.median(si))


ROWS = [(f'poly  K={k}', f'hd_polyk{k}.npz') for k in (5, 6, 7)]
ROWS += [(f'bw  {fc:>4} Hz', f'hd_bw{fc}.npz') for fc in (4, 6, 8, 12, 20)]
ROWS += [(f'bwk {fc:>4} Hz', f'hd_bwk{fc}.npz')
         for fc in (1.5, 2, 2.5, 3, 4, 6, 8, 12, 20)]

print(f"  {'setting':12}{'level/model':>13}{'sym median':>12}{'sym RMS':>10}"
      f"{'anti RMS':>10}{'within-run slope':>18}")
prev = None
for lab, f in ROWS:
    p = SP / f
    if not p.exists():
        print(f"  {lab:12}   (missing)")
        continue
    r = stats(p)
    if prev is not None and lab[:3] != prev:
        print()
    prev = lab[:3]
    print(f"  {lab:12}{r['level']:13.2f}{r['sym_med']:12.1f}"
          f"{r['sym_rms']:10.1f}{r['anti_rms']:10.1f}{r['slope']:18.1f}")
