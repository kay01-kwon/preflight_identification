#!/usr/bin/env python3
"""How well can the cosh fit, given the approximations that produced it?

Equations (9)->(10) reach the piecewise rate dynamics by setting
cos(phi) ~= 1 and sin(phi) ~= 0, and the ground-effect moment is
linearised alongside.  A reviewer asks for the resulting approximation
error to be bounded rather than asserted, over the tilt the runs
actually reach.  That is a question about FIT QUALITY, and it has an a
priori answer.

Carry the neglected terms instead of dropping them.  What (10b) omits is

    rho(tau) = rho_phi(tau) + rho_GE(tau),
    rho_phi <= (1/2) W a phi_max^2          the small-angle remainder
    rho_GE  <= |beta_M| dM_win phi_max      the GE linearisation

and the window average obeys rho_bar <= rho_phi,max/7 + rho_GE,max/5 by
the reduction factors of (95).  Propagating rho through the deviation
dynamics gives the pointwise envelope

    |e_omega(tau)| <= E(tau) = rho_bar sinh(C2 tau)/(J_P C2).

The step to fit quality is one line and needs nothing else.  The nominal
cosh is itself an admissible member of the fitted family, so the
minimiser cannot do worse than it does:

    ||y - g_hat|| = min_g ||y - g|| <= ||y - f|| = ||e|| <= ||E||,

and dividing by the window length turns that into an RMS cap on the
residual that can be quoted before any run is flown,

    RMS(residual) <= rho_bar K C2 sqrt(B(x)/x),
    B(x) = (1/4) sinh(2x) - x/2,   x = C2 tau_end,   J_P = 1/(K C2^2).

This computes that cap per ramp rate from the geometry box alone, and
sets it against the residual the campaign actually achieves.  A measured
residual ABOVE the cap does not mean the rho analysis failed -- the cap
assumes the true nominal is in the fitted family, so exceeding it points
at the pinned C2 or K, or at a disturbance that is not rho at all.

What comes out, over 140 runs reaching 5.0-5.7 deg of tilt:

  at the 10 deg box     cap 1.13-5.45 deg/s, 82/140 runs inside
  at the realised tilt  cap 0.30-1.38 deg/s, 10/140 runs inside
  measured residual     1.21-2.32 deg/s, pre-onset noise 0.04-0.17

so the named approximations bound the residual at the box tilt but not
at the tilt the runs actually reach, and subtracting the noise floor
changes nothing.  The excess is therefore NOT the small-angle or GE
approximation.  Two candidates, neither of them the reviewer's: the
pinned (C2, K), which six of ten configurations already place above
their geometric ceiling; and the mechanical condition, since before the
onset the vehicle is carried by every landing gear and after it pivots
on one edge with much less damping, which makes the pre-onset floor an
under-estimate of the in-window disturbance.  That is stated as a
question, not an answer.

None of this touches the reported threshold.  |dM_crit| <= rho_bar is a
statement about the projection of e_omega onto the onset direction, and
the residual is by construction orthogonal to it; the threshold bound is
validated forwards in analysis/bound_validation_figure.py.

Usage: python analysis/fit_quality_bound.py [DATASET_ROOT]
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

W = 31.59
ARMS = {'Mx': 0.160, 'My': 0.130}
BETA_M = 0.0345
R_PHI, R_GE = 1.0 / 7.0, 1.0 / 5.0     # the reduction factors of (95)


def rho_bar(axis_dir, phi_max, dm_win):
    """(97b): the window-average forcing the model does not carry."""
    r_phi = 0.5 * W * ARMS[axis_dir] * phi_max ** 2
    r_ge = abs(BETA_M) * dm_win * phi_max
    return R_PHI * r_phi + R_GE * r_ge, r_phi, r_ge


def rms_cap(rb, k, c2, x):
    """||E|| / sqrt(tau_end), in rad/s."""
    b = 0.25 * np.sinh(2 * x) - 0.5 * x
    return rb * k * c2 * np.sqrt(b / x)


def collect(root, cache):
    if os.path.exists(cache):
        with open(cache, 'rb') as fh:
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
            r = om[j:] - pw['omega_pred'][j:]
            # The tilt the run ACTUALLY reaches, integrated from the
            # measured rate.  The excitation window closes on the moment
            # cap, not on a tilt cap, so this is well short of the 5 or
            # 10 deg the a priori box assumes, and rho_phi goes as phi^2.
            oc = om[j:] - float(pw['c'])
            phi_end = float(abs(np.trapz(oc, tau)))
            # The gyro noise floor, from the record before anything is
            # commanded.  It is in the measured residual and is NOT part
            # of e_omega, so comparing raw would charge the model for it.
            q = sig['omega'][:i0]
            noise = float(np.std(q)) if q.size > 50 else np.nan
            rows.append(dict(
                case=case, axis=ad, rate=rate, c2=c2, k=k,
                mdot=abs(md), te=float(tau[-1]), x=float(c2 * tau[-1]),
                dm_win=abs(md) * float(tau[-1]),
                res=float(np.sqrt(np.mean(r ** 2))),
                noise=noise, phi_end=phi_end,
                peak=float(np.max(np.abs(oc)))))
        print(f"  {case}/{ad} done ({len(rows)} runs)")
    with open(cache, 'wb') as fh:
        pickle.dump(rows, fh)
    return rows


def table(rows, phi_max_deg):
    """phi_max_deg = None uses each run's OWN measured peak tilt."""
    own = phi_max_deg is None
    pm = 0.0 if own else np.deg2rad(phi_max_deg)
    print(f"\n  tilt = {'each run its own measured peak' if own else str(int(phi_max_deg)) + ' deg (a priori box)'}\n")
    print(f"  {'Mdot':>6}{'n':>4}{'x':>7}{'phi':>7}{'rho_bar':>9}{'cap':>9}"
          f"{'resid':>8}{'noise':>8}{'resid-n':>9}{'ratio':>8}")
    print(f"  {'N m/s':>6}{'':4}{'':7}{'deg':>7}{'mN.m':>9}{'deg/s':>9}"
          f"{'deg/s':>8}{'deg/s':>8}{'deg/s':>9}{'':8}")
    g = collections.defaultdict(list)
    for r in rows:
        g[r['rate']].append(r)
    worst = 0.0
    for rate in sorted(g):
        v = g[rate]
        caps, meas, nz, net, rb, ph = [], [], [], [], [], []
        for r in v:
            pmi = r['phi_end'] if own else pm
            b, _, _ = rho_bar(r['axis'], pmi, r['dm_win'])
            c = rms_cap(b, r['k'], r['c2'], r['x'])
            n = 0.0 if np.isnan(r['noise']) else r['noise']
            rb.append(1e3 * b)
            ph.append(np.rad2deg(pmi))
            caps.append(np.rad2deg(c))
            meas.append(np.rad2deg(r['res']))
            nz.append(np.rad2deg(n))
            net.append(np.rad2deg(np.sqrt(max(r['res'] ** 2 - n ** 2, 0.0))))
        ratio = np.median(net) / np.median(caps)
        worst = max(worst, max(np.array(net) / np.array(caps)))
        print(f"  {rate:6.2f}{len(v):4d}{np.median([r['x'] for r in v]):7.3f}"
              f"{np.median(ph):7.2f}{np.median(rb):9.2f}{np.median(caps):9.4f}"
              f"{np.median(meas):8.4f}{np.median(nz):8.4f}"
              f"{np.median(net):9.4f}{ratio:8.3f}")
    def net_of(r):
        n = 0.0 if np.isnan(r['noise']) else r['noise']
        return np.sqrt(max(r['res'] ** 2 - n ** 2, 0.0))
    inside = sum(1 for r in rows
                 if net_of(r) <= rms_cap(
                     rho_bar(r['axis'],
                             r['phi_end'] if own else pm, r['dm_win'])[0],
                     r['k'], r['c2'], r['x']))
    print(f"\n  {inside}/{len(rows)} runs inside the cap;"
          f" worst ratio {worst:.3f}")
    return inside, worst


def old_model_cost(xs):
    """What setting sin(phi) ~= 0 outright used to cost.

    The reviewer's objection was raised against the earlier model, in
    which sin(phi) ~= 0 removed the destabilising term Wz sin(phi)
    ALTOGETHER.  With it gone, (10b) is a constant angular acceleration,
    the rate grows linearly and the tilt quadratically.  That is not a
    small-angle refinement -- it deletes the leading term of the
    dynamics, and it is a first-order omission.

    The present model keeps it, linearised only to Wz phi, which is what
    turns the quadratic into cosh with C2^2 = Wz/J_P.  Nothing is set to
    zero; what is left over is third order in phi from sin and second
    order from cos, and that is rho.

    This measures the gap between the two model SHAPES over the windows
    the campaign actually uses: fit the old quadratic (onset and scale
    free) to the cosh and see what it cannot match.
    """
    print("\n\n  what the earlier model omitted, over the observed windows")
    print("  cosh(x) - 1  against  x^2/2, the quadratic it replaced\n")
    print(f"  {'x':>6}{'endpoint ratio':>16}{'quad fit resid':>17}"
          f"{'onset error':>14}")
    print(f"  {'':6}{'cosh / quad':>16}{'% of peak':>17}{'% of window':>14}")
    for x in xs:
        tau = np.linspace(0.0, 1.0, 4001)          # normalised window
        f = np.cosh(x * tau) - 1.0
        peak = f[-1]
        best = (np.inf, 0.0)
        for d in np.linspace(-0.45, 0.45, 1801):   # onset free
            u = np.clip(tau - d, 0.0, None) ** 2
            A = np.column_stack([u, np.ones_like(u)])   # scale + baseline
            co, *_ = np.linalg.lstsq(A, f, rcond=None)
            v = float(np.sum((f - A @ co) ** 2))
            if v < best[0]:
                best = (v, d)
        r = np.sqrt(best[0] / len(tau))
        print(f"  {x:6.2f}{(np.cosh(x) - 1) / (0.5 * x ** 2):16.3f}"
              f"{100 * r / peak:17.2f}{100 * best[1]:14.2f}")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else 'DataSet/exp'
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '.fitq_cache.pkl')
    rows = collect(root, cache)
    print(f"\n  {len(rows)} runs.  The cap is a priori: it uses the geometry")
    print(f"  box, the tilt cap and the per-configuration (C2, K) only --")
    print(f"  no measured rate enters it.\n")
    print(f"  rho_phi,max = (1/2) W a phi_max^2, rho_GE,max ="
          f" |beta_M| dM_win phi_max,")
    print(f"  rho_bar <= rho_phi,max/7 + rho_GE,max/5,"
          f"  RMS cap = rho_bar K C2 sqrt(B(x)/x)")
    for pmv in (None, 5.0, 10.0):
        table(rows, pmv)
    old_model_cost(sorted({round(np.median([r['x'] for r in rows
                                            if r['rate'] == q]), 2)
                           for q in {r['rate'] for r in rows}}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
