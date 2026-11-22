#!/usr/bin/env python3
"""The RMS inequality on one measured record.

On data the true onset and the true constants are unknown, so the role
of the nominal model h(tau; theta_p^0) is played by the calibrated
family of the dataset: C1 = K* Mdot, C2 = C2*, the baseline from the
pre-onset samples, and only the onset searched (the COSH detector, as
in analysis/cosh_overview_figure.py).  Against it stands the free PNLS
minimiser h(tau; theta*) with (C1, C2, t_c, C) all fitted
(analysis/pnls_cost_figure.py).  The free fit can only do better, so

    RMS(r) = RMS(y - h(theta*))  <=  RMS(y - h(theta_cal))

is the data-side counterpart of (94)-(95).

  (a) t-omega: the samples, the two curves, the two onsets; an inset
      magnifies the late window where the curves separate
  (b) the two residuals with their RMS values

Usage: python analysis/witness_rms_data_figure.py [case] [axis] [run] [out.png]
"""
import contextlib
import io
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import critical_value_getter_piecewise as cvp
from constrained_calibration import ROOT
from pnls_constants import PNLS_CONSTANTS
from utils.extractor import load_excitation_dataset

FIT, NOM, SIG = '#c0392b', '#1f77b4', '0.55'
PAD = 0.30               # seconds drawn before the detected onset


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else 'case_05'
    axis_dir = sys.argv[2] if len(sys.argv) > 2 else 'My'
    run = sys.argv[3] if len(sys.argv) > 3 else 'pos_My_065'
    out = sys.argv[4] if len(sys.argv) > 4 else 'docs/exp_witness_rms_data.png'
    axis = 'x' if axis_dir == 'Mx' else 'y'
    c2s, ks = PNLS_CONSTANTS[(case, axis_dir)]
    deg = np.degrees

    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(ROOT / case / axis_dir)
        bag = next(b for b in bags if b.name == run)
        sig = cvp.prepare_signals(bag, axis)
    i0, i1 = cvp.detect_excitation_window(
        sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
    w = slice(i0, i1 + 1)
    t, om, M = sig['t'][w], sig['omega'][w], sig['moment'][w]
    t = t - t[0]
    md = float(np.polyfit(t, M, 1)[0])

    # calibrated family: C1 = K* Mdot, C2 = C2*, onset searched
    cal = cvp.cosh_onset_fit(t, om, np.zeros_like(t), onset_guess=None,
                             c2_fixed=c2s, moment_floor=0.0,
                             ramp_gain=ks, ramp_rate=md)
    # free PNLS minimiser: seed + bracket, (C1, C2, C) refitted per onset
    seed = cvp.piecewise_onset_fit(t, om)['onset_idx']
    free = cvp.cosh_onset_fit(t, om, M, onset_guess=seed, moment_floor=0.0)

    jc, jf = cal['onset_idx'], free['onset_idx']
    h_cal, h_free = cal['omega_pred'], free['omega_pred']
    C1f, C2f, Cf = free['params']
    r_cal, r_free = om - h_cal, om - h_free
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)))
    # a common comparison window: from PAD before the earlier onset
    j_lo = max(0, min(jc, jf) - int(PAD / np.median(np.diff(t))))
    s = slice(j_lo, len(t))
    rc, rf = rms(r_cal[s]), rms(r_free[s])

    print(f'{case}/{axis_dir} {bag.name}   Mdot = {md:+.4f} N.m/s')
    print(f'  calibrated : C1 = K*Mdot = {ks * md:+.4f}, C2* = {c2s:.3f}, '
          f't_c = {t[jc]:.3f} s, C = {float(cal["c"]):+.4f}')
    print(f'  free PNLS  : C1 = {C1f:+.4f}, C2 = {C2f:.3f}, '
          f't* = {t[jf]:.3f} s ({1e3 * (t[jf] - t[jc]):+.0f} ms), C = {Cf:+.4f}')
    print(f'  RMS over [{t[j_lo]:.2f}, {t[-1]:.2f}] s: free {deg(rf):.3f}  <=  '
          f'calibrated {deg(rc):.3f} deg/s   (ratio {rf / rc:.2f})')
    print(f'  end of window: y = {deg(om[-1]):.2f}, h_free = {deg(h_free[-1]):.2f}, '
          f'h_cal = {deg(h_cal[-1]):.2f} deg/s')

    fig, (A, B) = plt.subplots(2, 1, figsize=(6.4, 5.4), sharex=True,
                               gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.07))
    tt = t[s]

    # ── (a) t-omega ────────────────────────────────────────────
    A.plot(tt, deg(om[s]), 'o', ms=2.6, mfc=SIG, mec='none', zorder=1,
           label=r'$y_i$  measured $\omega_%s$' % axis)
    A.plot(tt, deg(h_cal[s]), '--', lw=2.0, color=NOM, zorder=3,
           label=r'$\hat h(\tau;\theta_{\mathrm{cal}})$: $C_1 = K^{*}\dot M$, $C_2 = C_2^{*}$, onset searched')
    A.plot(tt, deg(h_free[s]), '-', lw=1.8, color=FIT, zorder=4,
           label=r'$\hat h(\tau;\theta^{*})$: free PNLS minimiser')
    A.axvline(t[jc], ls=':', lw=1.0, color=NOM)
    A.axvline(t[jf], ls=':', lw=1.0, color=FIT)
    y_top = deg(om[s]).max() * 1.10
    A.set_ylim(deg(om[s]).min() - 0.5, y_top)
    A.annotate(r'$t_c$ (calibrated)', (t[jc], y_top * 0.15), xytext=(5, 0),
               textcoords='offset points', color=NOM, fontsize=8.5, va='center')
    A.annotate(rf'$t^{{*}}$ ({1e3 * (t[jf] - t[jc]):+.0f} ms)', (t[jf], y_top * 0.22),
               xytext=(-5, 0), textcoords='offset points', color=FIT, fontsize=8.5,
               va='center', ha='right')
    A.set_ylabel(r'$\omega$ [deg/s]', fontsize=10.5)
    A.legend(fontsize=7.6, loc='upper left', framealpha=0.92)
    A.set_title(f'{case}/{axis_dir}/{run}:  '
                rf'$\dot M = {md:+.2f}$ N$\cdot$m/s,  $C_2^{{*}} = {c2s:.2f}$,  $K^{{*}} = {ks:.3f}$',
                fontsize=9.5, color='0.3')

    # inset: the late window magnified
    n_in = max(12, int(0.12 / np.median(np.diff(t))))
    z = slice(len(t) - n_in, len(t))
    ins = inset_axes(A, width='100%', height='100%', loc='lower left',
                     bbox_to_anchor=(0.46, 0.36, 0.33, 0.38), bbox_transform=A.transAxes,
                     borderpad=0)
    ins.plot(t[z], deg(om[z]), 'o', ms=3.2, mfc=SIG, mec='none', zorder=1)
    ins.plot(t[z], deg(h_cal[z]), '--', lw=2.0, color=NOM, zorder=3)
    ins.plot(t[z], deg(h_free[z]), '-', lw=1.8, color=FIT, zorder=4)
    ins.tick_params(labelsize=6.5)
    ins.grid(alpha=0.25, lw=0.4)
    ins.set_title('zoom: last %.2f s' % (t[-1] - t[z][0]), fontsize=7.5, pad=2)
    for sp in ins.spines.values():
        sp.set_edgecolor('0.5')
    yz = np.concatenate([deg(om[z]), deg(h_cal[z]), deg(h_free[z])])
    A.add_patch(Rectangle((t[z][0], yz.min() - 1), t[-1] - t[z][0] + 0.01, yz.max() - yz.min() + 2,
                          fc='none', ec='0.5', lw=0.8, ls='-', zorder=6))

    # ── (b) residuals ──────────────────────────────────────────
    B.plot(tt, deg(r_cal[s]), '-', lw=1.3, color=NOM, zorder=3,
           label=rf'$y_i - \hat h(\theta_{{\mathrm{{cal}}}})$:  RMS {deg(rc):.3f}')
    B.plot(tt, deg(r_free[s]), '-', lw=1.3, color=FIT, zorder=4,
           label=rf'$r_i = y_i - \hat h(\theta^{{*}})$:  RMS {deg(rf):.3f}')
    B.axhline(0, lw=0.5, color='0.5')
    lim = max(abs(deg(r_cal[s])).max(), abs(deg(r_free[s])).max())
    B.set_ylim(-1.25 * lim, 1.9 * lim)
    B.set_ylabel('residual [deg/s]', fontsize=10.5)
    B.set_xlabel('$t$ [s]  (from the start of the excitation window)', fontsize=10.5)
    B.legend(fontsize=8, loc='upper left', framealpha=0.92)
    B.text(0.98, 0.93,
           rf'$\mathrm{{RMS}}(r) = {deg(rf):.3f} \leq \mathrm{{RMS}}(y - \hat h(\theta_{{\mathrm{{cal}}}})) = {deg(rc):.3f}$ deg/s',
           transform=B.transAxes, fontsize=8.2, ha='right', va='top',
           bbox=dict(fc='white', ec='0.8', lw=0.6, pad=3))
    for ax in (A, B):
        ax.tick_params(labelsize=8.5)
        ax.grid(alpha=0.25, lw=0.4)
    A.set_xlim(tt[0], tt[-1] + 0.02)
    fig.savefig(out, dpi=600, bbox_inches='tight')
    print('written', out)


if __name__ == '__main__':
    main()
