#!/usr/bin/env python3
"""How far the applied moment departs from the linear ramp, and what it costs.

The onset model presumes M(tau) = M_0 + Mdot tau.  No model of the
rotor-loop tracking error is needed to bound the departure, because the
moment is not commanded into the analysis -- it is reconstructed from
the measured rotor speeds, and Mdot is the least-squares slope of that
measured trace over the excitation window
(critical_value_getter_piecewise.py, the `m_dot = np.polyfit(...)` line).

Two consequences follow, and together they are the whole argument.

First, the constant and linear parts of any tracking error cannot
appear: they are removed by the normal equations of that fit.  What is
left, rho_track = M_meas - (M_0 + Mdot tau), is the non-affine part
alone, and it is measured per run rather than modelled.

Second, rho_track is orthogonal to span{1, tau} over the window by those
same normal equations, so in the reduction Delta M_crit = -Int rho w it
can only pair with the part of the weight that is NOT affine:

    Delta M_crit = -Int rho_track (w - P_1 w)
    |Delta M_crit| <= ||rho_track||_inf * ||w - P_1 w||_1

with P_1 the least-squares projection onto span{1, tau}.  The weight is
smooth, so that norm is well under 1 and most of the residual is
absorbed before it can move the onset.

This channel is outside the a priori bound of Sec. VI-E, which covers
only the gravity remainder and the ground-effect coupling.  It belongs
to the realised budget, and this script produces the numbers quoted
there.

Usage: python analysis/ramp_linearity.py
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parent) + '/stubs')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset
from constrained_calibration import ROOT

# Widest x over the geometric box of Sec. VI-D; the absorbed fraction
# grows with x, so evaluating there is the unfavourable end.
X_BOX_MAX = 5.21
LINEARITY_GATE = 30.0   # mN.m RMSE about the best-fit line, Sec. VI-B
W_MIN = 30.08           # N, unloaded weight, for the offset conversion


def weight_absorption(x, n=1201):
    """||w - P_1 w||_1 / ||w||_1 for the reduction weight on [0, x].

    w(u) is proportional to Int_u^x sinh(v) cosh(v-u) dv; the constant
    drops out of the ratio, so no normalisation is needed.
    """
    u = np.linspace(0.0, x, n)
    w = np.array([quad(lambda v: np.sinh(v) * np.cosh(v - uu), uu, x,
                       limit=200)[0] for uu in u])
    A = np.vander(u, 2)
    c, *_ = np.linalg.lstsq(A, w, rcond=None)
    return float(np.trapz(np.abs(w - A @ c), u) / np.trapz(np.abs(w), u))


def residuals_of(bag, axis):
    """Peak and RMS of the moment's departure from its best-fit line."""
    with contextlib.redirect_stdout(io.StringIO()):
        sig = cvp.prepare_signals(bag, axis)
    t, M = sig['t'], sig['moment']
    i0, i1 = cvp.detect_excitation_window(M, 0.30,
                                          moment_cap=cvp.MOMENT_CAP.get(axis))
    if (i1 - i0) < 12:
        return None
    w = slice(i0, i1 + 1)
    r = M[w] - np.polyval(np.polyfit(t[w], M[w], 1), t[w])
    return float(np.max(np.abs(r))), float(np.sqrt(np.mean(r ** 2))), \
        float(np.ptp(M[w]))


def main():
    print("departure of the applied moment from the linear ramp\n")
    print(f"  {'case/axis':14}{'n':>4}{'|M| span':>11}{'peak':>10}{'rms':>9}"
          f"{'peak/span':>11}")
    peak, rms = [], []
    for d in sorted(ROOT.glob('case_*/M[xy]')):
        axis = 'x' if d.name == 'Mx' else 'y'
        with contextlib.redirect_stdout(io.StringIO()):
            bags = load_excitation_dataset(d)
        got = [r for r in (residuals_of(b, axis) for b in bags) if r]
        if not got:
            continue
        pk, rm, sp = (np.array([g[i] for g in got]) for i in range(3))
        peak += list(pk)
        rms += list(rm)
        print(f"  {d.parent.name + '/' + d.name:14}{len(got):4d}"
              f"{np.median(sp):11.3f}{1e3 * np.median(pk):10.2f}"
              f"{1e3 * np.median(rm):9.2f}"
              f"{np.median(pk) / np.median(sp):11.4f}")

    peak, rms = np.array(peak), np.array(rms)
    print(f"\n  over all {len(peak)} runs")
    print(f"    peak |residual|   median {1e3 * np.median(peak):6.2f} mN.m,"
          f"   worst {1e3 * peak.max():6.2f} mN.m")
    print(f"    rms  residual     median {1e3 * np.median(rms):6.2f} mN.m,"
          f"   worst {1e3 * rms.max():6.2f} mN.m")
    print(f"    linearity gate of Sec. VI-B is {LINEARITY_GATE:.0f} mN.m RMSE;"
          f" the worst run is {LINEARITY_GATE / (1e3 * rms.max()):.1f}x inside it")

    f = weight_absorption(X_BOX_MAX)
    print(f"\n  rho_track is orthogonal to span(1, tau) by the polyfit normal")
    print(f"  equations, so only the non-affine part of the weight can pair")
    print(f"  with it:  ||w - P_1 w||_1 / ||w||_1 = {f:.4f} at x = {X_BOX_MAX}\n")
    print(f"  {'':18}{'peak [mN.m]':>13}{'absorbed':>11}{'as offset':>12}")
    for lab, v in (('median run', np.median(peak)), ('worst run', peak.max())):
        print(f"  {lab:18}{1e3 * v:13.2f}{1e3 * v * f:11.2f}"
              f"{1e3 * v * f / W_MIN:11.3f} mm")
    print(f"\n  This channel is outside the Sec. VI-E bound, which covers only")
    print(f"  the gravity remainder and the ground-effect coupling.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
