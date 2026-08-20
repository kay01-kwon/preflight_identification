#!/usr/bin/env python3
"""What band-limited denoising does to the dynamic ground-effect check.

The omega_dot differentiator is run at the band-limit window of the
noise model (docs/noise_model_notes.tex, w ~ 2/(f_c T_s), i.e. 41
samples for f_c = 5 Hz at T_s = 9.9 ms) at two polynomial orders, the
deployed 2 and a curvature-matched 7, and the results are drawn
together.

WHY THE ORDER MATTERS.  omega_dot ~ sinh(C2 tau) with an e-folding
time 1/C2 = 163 ms, so a 41-sample (410 ms) window spans 2.5
e-foldings -- far more curvature than a local parabola can follow
(rate_derivative.omega_dot defaults to poly=2, deriv=1: the analytic
slope of a parabola fitted to the raw omega, so the raw difference is
never formed).  The parabola clips the growth of J_P omega_dot, which
is exactly the term that has to cancel -W z_CoM sin(phi), and the
fitted slope opens up: -89.7 mN.m/deg at order 2 against -38.2 at
order 7 on this run.  On a NOISE-FREE cosh with these constants the
endpoint error of the derivative is -6.4% at order 3 and -0.2% at
order 5, so the difference is signal distortion, not noise (see
analysis/sg_derivative_order.py).  The cost of the high order is
noise gain, visible in panel (b): it buys fidelity only once sinh has
taken off, and pays for it everywhere before that.

WHY THE TRAILING ZONE IS SUSPECT -- and why the usual reason is the
wrong one.  It is NOT that samples run out: the pipeline
differentiates the FULL bag trace and slices afterwards, and 335
samples follow the excitation window on this run, so every point
inside the window has two-sided filter support and nothing is
extrapolated.  The real reason is regime mixing: within a half-width
of the window end (20 samples, 0.20 s for w = 41) the filter's
support reaches PAST the moment cap at i1, where the commanded moment
stops ramping and becomes constant.  Panels (a) and (b) are therefore
displayed only up to one full filter length before the window end
(--trim-end, default 0.41 s), and the same zone is shaded in panel
(c).  The fits themselves always use the whole window.

WHAT THE TRIMMED VIEW CAN AND CANNOT SHOW.  Over the trusted span
both orders sit on the model line -- but phi stays below 0.8 deg
there, so that agreement is the INTERCEPT, the static balance which
analysis/mcrit_prediction.py already establishes, and not evidence
about the attitude dependence.  Panel (c) makes the trade explicit:
excluding trailing samples does pull the slope toward the model
(-89.7 -> about -13 at k = 20 for order 2), but phi grows
exponentially, so those samples carry nearly all of the attitude
range -- dropping 20 of 79 leaves 2.1 deg of the 6.9 deg excursion,
dropping 41 leaves 0.8 deg, where the regression has no lever arm and
the estimate collapses.  Trimming the tail removes the very quantity
the check exists to measure, so it cannot rescue it.

The noise model's window rule therefore does not transfer unchanged:
there the smooth curve is DISCARDED and only the residue above f_c is
kept, so distorting it errs safe; here the smooth curve IS the
measurement.

Usage: python analysis/ge_savgol_overlay.py [out.png]
       [--dir DataSet/exp/case_02/Mx] [--bag pos_Mx_01] [--z-com 0.261]
       [--trim-end 0.41] [--jp-mode parallel]
"""
import argparse
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
from analysis.rate_derivative import omega_dot            # noqa: E402
from analysis.ge_dynamics_check import (                  # noqa: E402
    MASS_KG, G, OFF_SIGN, OFF_MM, j_parallel)

#            window, poly order, colour, dash
VARIANTS = [(41, 2, '#c0392b', '-'),       # parabola: cannot follow sinh
            (41, 7, '#148f77', '--')]      # order matched to the curvature
DROPS = [0, 3, 5, 8, 10, 15, 20, 30, 41]   # trailing samples excluded


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out', nargs='?', default='ge_savgol_overlay.png')
    p.add_argument('--dir', default='DataSet/exp/case_02/Mx')
    p.add_argument('--bag', default=None)
    p.add_argument('--z-com', type=float, default=0.261)
    p.add_argument('--trim-end', type=float, default=None,
                   help="seconds trimmed off the END of panels (a) and (b), "
                        "display only -- the fits and panel (c) always use "
                        "the whole window.  Default: one full 41-sample "
                        "filter length (0.41 s).")
    p.add_argument('--jp-mode', choices=['parallel', 'identity'],
                   default='parallel',
                   help="parallel: J_COM + m(z^2 + l_p^2), the CAD "
                        "parallel-axis value the batch uses; identity: "
                        "W z / C2^2, which sits below the rigid-body floor")
    a_ = p.parse_args()

    d = Path(a_.dir)
    axis = 'x' if d.name == 'Mx' else 'y'
    case, axname = d.parent.name, d.name
    W = MASS_KG[case] * G
    off_truth = OFF_SIGN[axname] * OFF_MM[(case, axname)] * 1e-3

    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
        c2, k_gain = cvp.estimate_rig_constants(bags, axis)
        crits, _ = cvp.extract_piecewise_batch(bags, axis, cosh_c2=c2,
                                               ramp_gain=k_gain)
    name = a_.bag or sorted(c.bag_name for c in crits
                            if c.bag_name.startswith('pos'))[0]
    crit = next(c for c in crits if c.bag_name == name)
    bag = next(b for b in bags if b.name == name)
    pos = name.startswith('pos')
    s = 1.0 if pos else -1.0

    piv = cvp.estimate_pivot_from_mocap(bag, crit.onset_time, axis)
    lp = (piv['pivot_abs'] * 1e-3 if not np.isnan(piv['pivot_abs'])
          else LP[axis])
    arm = lp + s * off_truth
    j_p = (j_parallel(axis, a_.z_com, MASS_KG[case])
           if a_.jp_mode == 'parallel' else W * a_.z_com / c2 ** 2)

    sig = cvp.prepare_signals(bag, axis)
    roll, pitch = math_tools.quaternion_to_euler_vectorized(
        bag.odom.quaternion)
    phi_all = roll if axis == 'x' else pitch
    n = min(len(phi_all), len(sig['t']))
    _, i1 = cvp.detect_excitation_window(
        sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
    j = crit.onset_idx
    i1 = min(i1, n - 1)
    sl = slice(j, i1 + 1)

    tau = sig['t'][sl] - sig['t'][j]
    phi = s * (phi_all[sl] - phi_all[j])
    m = s * sig['moment'][sl]
    f = sig['f_col'][sl]
    dt = float(np.median(np.diff(tau)))
    om_full = s * sig['omega'][:n]
    # ge_moment returns the whole trace; `window` only scopes its R/4 test
    ge_mod = s * ge_moment(bag, sig, axis, n, pos, window=sl)[sl]

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16.6, 4.9))
    fig.subplots_adjust(left=0.048, right=0.99, top=0.775, bottom=0.115,
                        wspace=0.27)

    print(f"\n  {case}/{axname}/{name}:  C2 = {c2:.3f} rad/s, "
          f"1/C2 = {1e3/c2:.0f} ms,  T_s = {1e3*dt:.1f} ms,  "
          f"window {len(tau)} samples / {tau[-1]:.2f} s")
    print(f"  J_P = {j_p:.4f} kg m^2 ({a_.jp_mode});  for reference: "
          f"parallel axis {j_parallel(axis, a_.z_com, MASS_KG[case]):.4f}, "
          f"identity W z/C2^2 {W*a_.z_com/c2**2:.4f}\n")
    a3_slopes, vis_a1, vis_a2 = [], [], []
    print(f"  {'w':>4}{'ord':>5}{'[ms]':>7}{'e-fold':>9}{'f_c~[Hz]':>10}"
          f"{'|om_dot|max':>13}{'slope':>10}{'at phi=0':>10}")
    for w, p_, c, ls in VARIANTS:
        od = omega_dot(om_full, dt, w, p_)[sl]
        ge_dyn = (j_p * od - m - f * lp
                  + W * arm * np.cos(phi) - W * a_.z_com * np.sin(phi))
        sd, id_ = np.polyfit(phi, ge_dyn, 1)
        lab = (f'SG {w} samples ({1e3*w*dt:.0f} ms), order {p_}')
        a1.plot(tau, 1e3 * ge_dyn, ls, color=c, lw=1.5, label=lab)
        a2.plot(tau, od, ls, color=c, lw=1.5, label=lab)
        vis_a1.append(1e3 * ge_dyn)
        vis_a2.append(od)
        print(f"  {w:4d}{p_:5d}{1e3*w*dt:7.0f}{w*dt*c2:9.2f}{2.0/(w*dt):10.1f}"
              f"{np.abs(od).max():13.2f}"
              f"{1e3*sd*np.pi/180:10.1f}{1e3*id_:10.1f}")
        # slope refitted with the last k samples excluded
        ks, sl_k = [], []
        for k in DROPS:
            e = len(phi) - k
            if e < 15:
                continue
            sk, _ = np.polyfit(phi[:e], ge_dyn[:e], 1)
            ks.append(k)
            sl_k.append(1e3 * sk * np.pi / 180)
        a3.plot(ks, sl_k, ls, color=c, lw=1.5, marker='o', ms=4, label=lab)
        a3_slopes.append(sl_k)
        print(f"      slope vs trailing samples dropped: "
              + "  ".join(f"k={kk}:{vv:.1f}" for kk, vv in zip(ks, sl_k)))

    a1.plot(tau, 1e3 * ge_mod, '-', color='#e08214', lw=2.2,
            label='image-superposition model')
    vis_a1.append(1e3 * ge_mod)
    a1.axhline(0, color='0.6', lw=0.7)

    # display trim: keep only where the widest filter's support lies
    # inside the excitation window (fits are unaffected)
    trim = 41 * dt if a_.trim_end is None else a_.trim_end
    t_cut = max(tau[-1] - trim, 3 * dt)
    vis = tau <= t_cut
    for ax_, series in ((a1, vis_a1), (a2, vis_a2)):
        ax_.set_xlim(0, t_cut)
        v = np.concatenate([np.asarray(y)[vis] for y in series])
        pad = 0.09 * (v.max() - v.min())
        ax_.set_ylim(v.min() - pad, v.max() + pad)
    a1.set_xlabel(r'$\tau$ [s]', fontsize=10)
    a1.set_ylabel(r'$\Delta M_{GE}$ [mN$\cdot$m]', fontsize=10)
    a1.set_title('(a) the dynamic inversion over the trusted span:\n'
                 'both orders sit on the model', fontsize=11)
    a1.legend(fontsize=8.5, loc='lower left')
    a1.grid(alpha=0.22, lw=0.4)

    a2.set_xlabel(r'$\tau$ [s]', fontsize=10)
    a2.set_ylabel(r'$\dot\omega$ [rad/s$^2$]', fontsize=10)
    a2.set_title(r'(b) $\dot\omega$ over the same span: order 7 costs '
                 'noise\nhere, and buys fidelity only once sinh takes off',
                 fontsize=11)
    a2.legend(fontsize=8.5, loc='upper left')
    a2.grid(alpha=0.22, lw=0.4)

    # ---- (c) does excluding the trailing samples rescue it? ----
    sm_, _ = np.polyfit(phi, ge_mod, 1)
    a3.axhline(1e3 * sm_ * np.pi / 180, color='#e08214', lw=2.0,
               label='image-superposition model')
    a3.axhline(0, color='0.6', lw=0.7)
    lo_y = min(min(sk_) for sk_ in a3_slopes)
    a3.set_ylim(1.18 * lo_y, max(30.0, -0.12 * lo_y))
    # k below the half-width: the filter's support still reaches past the
    # moment cap at the window end, so those fits mix the two regimes
    a3.axvspan(0, 20, color='0.85', alpha=0.55, lw=0, zorder=0)
    a3.text(10, a3.get_ylim()[1], 'filter support crosses\nthe moment cap',
            fontsize=7.5, color='0.35', ha='center', va='top')
    for k in (0, 10, 20, 30, 41):
        e = len(phi) - k
        if e < 15:
            continue
        a3.text(k, 0.055 * lo_y,
                rf'{np.rad2deg(phi[e-1]):.1f}$^\circ$ left',
                fontsize=7.5, color='0.35', ha='center', va='top',
                rotation=90)
    a3.set_xlabel('trailing samples excluded from the fit', fontsize=10)
    a3.set_ylabel(r'fitted slope [mN$\cdot$m/deg]', fontsize=10)
    a3.set_title('(c) past the shaded zone the two orders agree, but '
                 r'$\varphi$ grows' '\n'
                 f'exponentially: dropping 20 of {len(phi)} leaves '
                 f'{np.rad2deg(phi[len(phi)-21]):.1f}$^\\circ$ of '
                 f'{np.rad2deg(phi[-1]):.1f}$^\\circ$', fontsize=11)
    a3.legend(fontsize=8.5, loc='lower left')
    a3.grid(alpha=0.22, lw=0.4)

    jp_note = ('CAD parallel axis' if a_.jp_mode == 'parallel'
               else r'from $Wz/C_2^2$')
    fig.suptitle('Denoising the GE dynamic inversion, and why trimming '
                 'the window end does not rescue it\n'
                 f'({case}/{axname}/{name}, '
                 rf'$z_{{CoM}}$ = {a_.z_com:.3f} m, '
                 rf'$J_P$ = {j_p:.3f} kg m$^2$, {jp_note})',
                 fontsize=11.5, y=0.995)
    fig.savefig(a_.out, dpi=150)
    print(f"\n  wrote {a_.out}\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
