#!/usr/bin/env python3
"""Which part of the gyro spectrum is rotation about the contact line?

The inversion treats every wiggle in the rate as rigid rotation of the
airframe about its contact line, and multiplies it by J_P.  That is
only legitimate where the moment to produce the rotation is actually
present: rigid rotation needs

    J_P omega_dot  =  net applied moment

band by band.  So compare, in each frequency band, the inertial moment
the gyro implies against the moment the load cell and the collective
actually show there.  A ratio near one is rotation; a large ratio is
structure -- the flight controller sits on the airframe, not at the
pivot, so it also reads bending and motor-mount modes that carry no
moment about the contact line at all.

On this dataset the answer is sharp, and it is the same in all twenty
case/axis groups.  Over 140 runs the median ratio is

    1-3 Hz   3-6    6-9    9-13   13-18   18-25   25-40
       1.2    8.7   64.7   80.0    65.2    59.0   113.1

with the 1-3 Hz quartiles at 1.0 and 1.9 and no group outside 0.9-2.0
there.  Below 3 Hz the balance closes; by 3-6 Hz the gyro already
implies nine times the moment that exists, and above 6 Hz the ratio is
in the tens to hundreds.  The ~10 Hz line in the rate is therefore NOT
rigid-body motion, and a derivative that preserves it -- a Butterworth
at 12 Hz, say -- injects several hundred mN.m of fictitious moment into
a balance whose answer is 160.

Nor does cutting at 3 Hz rescue the gradient, because real rotation and
structure are adjacent rather than separated: filtering below 3 Hz also
removes the ramp's own content, and with phi taken from the integral of
the filtered rate the tilt compresses and the gradient inflates.  The
within-run slope runs -135, -85, -58, -41, -29 mN.m/deg at cutoffs of
1.5, 2, 2.5, 3 and 4 Hz.  The level survives all of it (1.11-1.30 of
the model across seventeen settings of both derivative families); the
gradient does not, and that is a limit of the instrument.

One asymmetry worth noting: at 1-3 Hz every Mx group reads 0.9-1.1
while every My group reads 1.8-2.0, so even in the band where the
balance closes, My closes half as well.  That is the same axis on which
the antisymmetric residual sits (+7.3 mm against -2.1 on Mx).

(For scale: a K = 6 polynomial over a 0.67 s window can represent about
three half-cycles, i.e. roughly 4.5 Hz, so the polynomial route was
already acting as a low pass near this cutoff.  That is why it worked.)

Usage:
  PYTHONPATH=<stubs> python analysis/rate_band_check.py [case] [Mx|My] [bag]
  PYTHONPATH=<stubs> python analysis/rate_band_check.py --all
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent) + '/stubs')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset
from analysis.error_budget import LP

ROOT = Path(__file__).resolve().parents[1] / 'DataSet' / 'exp'
MASS = {'case_01': 3.066, **{f'case_0{i}': 3.220 for i in range(2, 6)}}
J_CAD, Z = dict(x=0.0537, y=0.0537), 0.261
BANDS = [(1, 3), (3, 6), (6, 9), (9, 13), (13, 18), (18, 25), (25, 40)]
ALL = '--all' in sys.argv
argv = [a for a in sys.argv[1:] if not a.startswith('-')]
CASE = argv[0] if argv else 'case_03'
AXNAME = argv[1] if len(argv) > 1 else 'Mx'
BAG = argv[2] if len(argv) > 2 else 'neg_Mx_045'


def spectrum(x, dt):
    """amplitude spectrum after removing the ramp with a cubic"""
    x = np.asarray(x, float)
    k = np.arange(len(x))
    x = x - np.polyval(np.polyfit(k, x, 3), k)
    return np.abs(np.fft.rfft(x * np.hanning(len(x)))) / len(x) * 4


def load_group(case, axname):
    """read one case/axis once -- reloading per bag is quadratic"""
    ax = 'x' if axname == 'Mx' else 'y'
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(ROOT / case / axname)
        crits, _ = cvp.extract_piecewise_batch(bags, ax)
    return ax, {b.name: b for b in bags}, crits


def ratios(case, axname, bagname, group=None):
    ax, by, crits = group if group else load_group(case, axname)
    crit = next((c for c in crits if c.bag_name == bagname), None)
    if crit is None:
        return None
    bag = by[bagname]
    sig = cvp.prepare_signals(bag, ax)
    n = len(sig['t'])
    dt = float(np.median(np.diff(sig['t'][:n])))
    j = crit.onset_idx
    k0, k1 = max(0, j - 150), min(n, j + 250)
    f = np.fft.rfftfreq(k1 - k0, dt)
    xw = spectrum(sig['omega'][k0:k1], dt)
    xm = spectrum(sig['moment'][k0:k1], dt)
    xf = spectrum(sig['f_col'][k0:k1], dt)
    j_p = J_CAD[ax] + MASS[case] * (Z ** 2 + LP[ax] ** 2)
    out = []
    for lo, hi in BANDS:
        b = (f >= lo) & (f < hi)
        if not b.sum():
            out.append(np.nan)
            continue
        fc = f[b][np.argmax(xw[b])]
        need = j_p * (2 * np.pi * fc) * xw[b].max()
        seen = xm[b].max() + xf[b].max() * LP[ax]
        out.append(need / max(seen, 1e-12))
    return np.array(out)


hdr = f"  {'run':26}" + ''.join(f"{f'{lo}-{hi}':>9}" for lo, hi in BANDS)
if not ALL:
    r = ratios(CASE, AXNAME, BAG)
    print("J_P omega_dot implied by the gyro, over the moment actually")
    print("present in that band.  1 = rigid rotation, large = structure.\n")
    print(hdr)
    print(f"  {CASE + '/' + BAG:26}" + ''.join(f"{v:9.1f}" for v in r))
else:
    print("J_P omega_dot implied by the gyro, over the moment actually")
    print("present in that band.  1 = rigid rotation, large = structure.\n")
    print(hdr)
    rows = []
    for case in sorted(p.name for p in ROOT.glob('case_*') if p.is_dir()):
        for axname in ('Mx', 'My'):
            if not (ROOT / case / axname).exists():
                continue
            g = load_group(case, axname)
            for crit in g[2]:
                r = ratios(case, axname, crit.bag_name, group=g)
                if r is None:
                    continue
                rows.append(r)
            print(f"  {case + '/' + axname:26}"
                  + ''.join(f"{v:9.1f}" for v in
                            np.median(np.array(rows[-len(g[2]):]), axis=0)),
                  flush=True)
    rows = np.array(rows)
    print(f"\n  {f'MEDIAN over {len(rows)} runs':26}"
          + ''.join(f"{v:9.1f}" for v in np.median(rows, axis=0)))
    print(f"  {'25th percentile':26}"
          + ''.join(f"{v:9.1f}" for v in np.percentile(rows, 25, axis=0)))
    print(f"  {'75th percentile':26}"
          + ''.join(f"{v:9.1f}" for v in np.percentile(rows, 75, axis=0)))
