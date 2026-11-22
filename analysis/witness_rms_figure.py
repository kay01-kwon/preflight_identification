#!/usr/bin/env python3
"""The RMS inequality (94)-(97) drawn on one simulated record.

Exact nonlinear tip-over (analysis/witness_verify.py) so the nominal
parameters theta_p^0 = {C1, C2, 0, C} are known exactly.  Three curves
share the measured samples y_i:

  (a) y_i, the true signal omega_nom + e_omega, the free PNLS fit
      h(tau; theta*) and the nominal model h(tau; theta_p^0); the fit
      pulls the onset earlier (t* < 0) to absorb the sinh part of e_omega
  (b) the two residuals the inequality compares: r_i = y_i - h(theta*)
      and n_i + e_omega(tau_i) = y_i - h(theta_p^0), with the envelope
      E(tau) of (84); their RMS values and the a priori bound (96)-(97)
      are printed on the panel

Usage: python analysis/witness_rms_figure.py [mdot] [out.png]
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import witness_verify as wv                          # noqa: E402

# the box-worst rho_bar configuration the shift validation uses
LP = 0.140
wv.M_KG, wv.L_ARM, wv.Z = 3.220, LP + 0.025, 0.272
wv.J_CAD = 0.050564
wv.W = wv.M_KG * 9.81
wv.J_P = wv.J_CAD + wv.M_KG * (wv.Z ** 2 + LP ** 2)
wv.C2 = np.sqrt(wv.W * wv.Z / wv.J_P)
wv.K = 1.0 / (wv.W * wv.Z)
wv.RNG = np.random.default_rng(3)

FIT = '#c0392b'
NOM = '#1f77b4'
TRU = '#222222'
ENV = '#f39c12'


def model(t, p):
    c1, c2, ts, c = p
    h = np.full_like(t, c)
    m = t >= ts
    h[m] += c1 * (np.cosh(c2 * (t[m] - ts)) - 1.0)
    return h


def main():
    mdot = float(sys.argv[1]) if len(sys.argv) > 1 else 0.45
    out = sys.argv[2] if len(sys.argv) > 2 else 'docs/exp_witness_rms.png'
    deg = np.degrees

    t, y, tau, e_om, x = wv.simulate(mdot)
    n_pre = int(wv.T_PRE * wv.FS)
    p_star, rms_fit = wv.fit(t, y)
    p_nom = (wv.K * mdot, wv.C2, 0.0, wv.BIAS)         # theta_p^0 (+ bias)
    h_star, h_nom = model(t, p_star), model(t, p_nom)
    om_true = np.concatenate([np.zeros(n_pre), e_om]) + h_nom
    r_fit, r_nom = y - h_star, y - h_nom
    noise = y - om_true

    # a priori pieces of (96)-(97)
    rho_bar = wv.R_PHI * 0.5 * wv.W * wv.L_ARM * wv.PHI_MAX ** 2
    B = 0.25 * np.sinh(2 * x) - 0.5 * x
    cap_e = rho_bar * wv.K * wv.C2 * np.sqrt(B / x)      # (97), post-onset RMS
    env = rho_bar * wv.K * wv.C2 * np.sinh(wv.C2 * tau)  # (84)
    # all RMS values on the same (full) window, as in (95)-(96)
    e_full = np.concatenate([np.zeros(n_pre), e_om])
    rms_e = float(np.sqrt(np.mean(e_full ** 2)))
    rms_n = float(np.sqrt(np.mean(noise ** 2)))
    rms_nom = float(np.sqrt(np.mean(r_nom ** 2)))
    # the bound is stated on the full window; scale the post-onset cap
    # to the full-window RMS by the sample share
    cap_full = cap_e * np.sqrt(len(tau) / len(t))

    print(f'Mdot = {mdot} N.m/s, window {t[0]:.2f}..{t[-1]:.2f} s, '
          f'{len(t)} samples ({len(tau)} post-onset)')
    print(f'  theta_p^0 : C1 = {p_nom[0]:.5f}, C2 = {p_nom[1]:.3f}, t_c = 0, C = {p_nom[3]:.4f}')
    print(f'  theta*    : C1 = {p_star[0]:.5f}, C2 = {p_star[1]:.3f}, '
          f't* = {1e3 * p_star[2]:+.1f} ms, C = {p_star[3]:.4f}')
    print(f'  RMS(r)        = {deg(rms_fit):.4f} deg/s   (free fit)')
    print(f'  RMS(n + e_w)  = {deg(rms_nom):.4f} deg/s   (nominal, witness)')
    print(f'  RMS(e_w)      = {deg(rms_e):.4f}  RMS(n) = {deg(rms_n):.4f}  '
          f'sum = {deg(rms_e + rms_n):.4f}')
    print(f'  (97) cap on RMS(e_w): {deg(cap_e):.4f} (post-onset) '
          f'-> {deg(cap_full):.4f} full window')

    fig, (A, B_) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True,
                                gridspec_kw=dict(height_ratios=[1.25, 1], hspace=0.08))

    # ── (a) the three curves on the same samples ─────────────────
    A.plot(t, deg(y), '.', ms=2.4, color='0.6', zorder=1, label=r'$y_i$ (measured)')
    A.plot(t, deg(om_true), '-', lw=0.9, color=TRU, alpha=0.8, zorder=2,
           label=r'true $\omega_{\mathrm{nom}} + e_\omega$')
    A.plot(t, deg(h_nom), '--', lw=1.6, color=NOM, zorder=3,
           label=r'$\hat h(\tau;\,\theta_p^{0})$  nominal, $t_c = 0$')
    A.plot(t, deg(h_star), '-', lw=1.6, color=FIT, zorder=4,
           label=r'$\hat h(\tau;\,\theta^{*})$  free PNLS minimiser')
    A.axvline(0, ls=':', lw=1.0, color=NOM)
    A.axvline(p_star[2], ls=':', lw=1.0, color=FIT)
    A.text(0, A.get_ylim()[1] * 0.97, r' $t_c{=}0$', color=NOM, fontsize=8.5, va='top')
    A.text(p_star[2], A.get_ylim()[1] * 0.80, rf'$t^{{*}} = {1e3 * p_star[2]:+.0f}$ ms ',
           color=FIT, fontsize=8.5, va='top', ha='right')
    A.set_ylabel(r'$\omega$ [deg/s]', fontsize=10)
    A.legend(fontsize=8, loc='upper left', framealpha=0.9)
    A.set_title(rf'exact contact dynamics, $\dot M = {mdot:.2f}$ N$\cdot$m/s, '
                rf'$\sigma_n = {deg(wv.SIGMA):.3f}$ deg/s', fontsize=10)

    # ── (b) the two residuals and the envelope ───────────────────
    tt = np.concatenate([t[:n_pre], tau])
    env_full = np.concatenate([np.zeros(n_pre), env])
    B_.fill_between(tt, -deg(env_full), deg(env_full), color=ENV, alpha=0.18, lw=0,
                    label=r'envelope $\pm E(\tau)$ (84), clipped')
    B_.plot(t, deg(r_nom), '-', lw=1.3, color=NOM, zorder=3,
            label=rf'$y_i - \hat h(\theta_p^0) = n_i + e_\omega$:  RMS {deg(rms_nom):.3f}')
    B_.plot(t, deg(r_fit), '-', lw=1.3, color=FIT, zorder=4,
            label=rf'$r_i = y_i - \hat h(\theta^{{*}})$:  RMS {deg(rms_fit):.3f}')
    B_.axhline(0, lw=0.5, color='0.5')
    B_.set_ylim(-0.45, 0.45)          # the envelope runs past the frame on purpose
    B_.text(t[-1], 0.43, rf' $E(\tau_{{\mathrm{{end}}}}) = {deg(env[-1]):.2f}$', fontsize=8,
            color='#a5690a', ha='right', va='top')
    B_.set_ylabel(r'residual [deg/s]', fontsize=10)
    B_.set_xlabel(r'$\tau$ [s]  (true onset at 0)', fontsize=10)
    B_.legend(fontsize=8, loc='upper left', framealpha=0.9)
    B_.text(0.98, 0.06,
            rf'$\mathrm{{RMS}}(r) = {deg(rms_fit):.3f} \leq \mathrm{{RMS}}(n+e_\omega) = {deg(rms_nom):.3f}$'
            '\n'
            rf'$\leq \mathrm{{RMS}}(e_\omega) + \mathrm{{RMS}}(n) = {deg(rms_e):.3f} + {deg(rms_n):.3f}$'
            '\n'
            rf'$\leq$ a priori (97): ${deg(cap_full):.3f}$ (+ noise term)   [deg/s]',
            transform=B_.transAxes, fontsize=8.2, ha='right', va='bottom',
            bbox=dict(fc='white', ec='0.8', lw=0.6, pad=3))
    for ax in (A, B_):
        ax.tick_params(labelsize=8.5)
        ax.grid(alpha=0.25, lw=0.4)
    fig.savefig(out, dpi=600, bbox_inches='tight')
    print('written', out)


if __name__ == '__main__':
    main()
