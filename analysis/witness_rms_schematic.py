#!/usr/bin/env python3
"""Schematic t-omega picture of the RMS inequality (94)-(97).

Not a data figure: the equivalent input rho(tau) and the gyro noise are
exaggerated (about 25 % and 2.5 % of the end value) so that the three
curves the inequality compares separate visibly on a slide:

  y_i                 measured samples        = omega_nom + e_omega + n
  h(tau; theta_p^0)   nominal model, t_c = 0  (the witness)
  h(tau; theta*)      free PNLS minimiser     (onset pulled earlier)

The dynamics are the linearised tip-over J_P phi'' = W Z phi + M_dot tau
+ rho(tau), rho(tau) = rho_bar (tau / tau_end)^2, integrated exactly on
the sample grid, so e_omega has the sinh-like growth of (84).

Usage: python analysis/witness_rms_schematic.py [out.png]
"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

FIT, NOM, TRU, ENV = '#c0392b', '#1f77b4', '#222222', '#f39c12'

# nominal constants in the range of the paper (C2 ~ 5.5 1/s, K = 1/(W Z))
C2, K, MDOT = 5.5, 0.116, 0.45
C1 = K * MDOT
T_PRE, T_POST, FS = 0.25, 0.55, 200.0
RHO_FRAC = 0.25            # e_omega(end) / omega_nom(end), exaggerated
SIGMA_FRAC = 0.025         # sigma_n / omega_nom(end), exaggerated
BIAS = 0.0


def model(t, p):
    c1, c2, ts, c = p
    h = np.full_like(t, c)
    m = t >= ts
    h[m] += c1 * (np.cosh(c2 * (t[m] - ts)) - 1.0)
    return h


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'docs/exp_witness_rms_schematic.png'
    deg = np.degrees
    rng = np.random.default_rng(7)

    n_pre, n_post = int(round(T_PRE * FS)), int(round(T_POST * FS))
    t = np.arange(-n_pre, n_post + 1) / FS
    tau = t[n_pre:]
    t_end = tau[-1]
    om_nom = model(t, (C1, C2, 0.0, 0.0))
    om_end = om_nom[-1]

    # linearised tip-over with the equivalent input rho(tau);
    # state (phi, omega); omega' = C2^2 phi + K C2^2 (M_dot tau + rho)/(...)
    # written so that rho = 0 reproduces omega_nom exactly:
    # omega_nom = C1 (cosh C2 tau - 1)  <=>  omega'' = C2^2 omega + C2^2 C1
    # => with rho:  omega'' = C2^2 omega + C2^2 K (M_dot + rho'(tau))
    def rhs(s, x, rb):
        om, dom = x
        drho = rb * 2.0 * s / T_POST ** 2
        return [dom, C2 ** 2 * om + C2 ** 2 * K * (MDOT + drho)]

    # scale rho_bar so that e_omega(end) = RHO_FRAC * omega_nom(end)
    sol1 = solve_ivp(rhs, (0, t_end), [0.0, 0.0], args=(1.0,), t_eval=tau, rtol=1e-10, atol=1e-12)
    e_unit = sol1.y[0] - om_nom[t >= 0]
    rho_bar = RHO_FRAC * om_end / e_unit[-1]
    e_om = rho_bar * e_unit
    e_full = np.concatenate([np.zeros(len(t) - len(tau)), e_om])
    om_true = om_nom + e_full
    sigma = SIGMA_FRAC * om_end
    noise = rng.normal(0.0, sigma, len(t))
    y = om_true + noise + BIAS

    # free PNLS minimiser
    def res(p):
        return model(t, p) - y
    p0 = (C1, C2, -0.02, 0.0)
    p_star = least_squares(res, p0, bounds=([1e-4, 3.0, -0.15, -1], [1.0, 8.0, 0.15, 1])).x
    p_nom = (C1, C2, 0.0, BIAS)
    h_star, h_nom = model(t, p_star), model(t, p_nom)
    r_fit, r_nom = y - h_star, y - h_nom
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)))
    print(f'rho_bar = {rho_bar:.4f} N.m (exaggerated), sigma_n = {deg(sigma):.3f} deg/s')
    print(f'theta*: C1 = {p_star[0]:.4f} (C1^0 {C1:.4f}), C2 = {p_star[1]:.3f}, '
          f't* = {1e3 * p_star[2]:+.1f} ms, C = {p_star[3]:.4f}')
    print(f'RMS(r) {deg(rms(r_fit)):.3f} <= RMS(n+e) {deg(rms(r_nom)):.3f} '
          f'<= RMS(e) + RMS(n) = {deg(rms(e_full)):.3f} + {deg(rms(noise)):.3f} [deg/s]')

    fig, (A, B) = plt.subplots(2, 1, figsize=(6.4, 5.4), sharex=True,
                               gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.07))

    # ── (a) t - omega ────────────────────────────────────────────
    A.plot(t, deg(y), 'o', ms=2.6, mfc='0.55', mec='none', zorder=1,
           label=r'$y_i = \omega_{\mathrm{nom}} + e_\omega + n_i$  (samples)')
    A.plot(t, deg(om_nom), ':', lw=1.4, color='0.35', zorder=2,
           label=r'$\omega_{\mathrm{nom}}(\tau) = C_1(\cosh C_2\tau - 1)$  no perturbation')
    A.plot(t, deg(h_nom), '--', lw=2.0, color=NOM, zorder=3,
           label=r'$\hat h(\tau;\theta_p^{0})$  nominal, $t_c = 0$  (witness)')
    A.plot(t, deg(h_star), '-', lw=2.0, color=FIT, zorder=4,
           label=r'$\hat h(\tau;\theta^{*})$  free PNLS minimiser')
    # shade e_omega between nominal and true
    A.fill_between(t, deg(om_nom), deg(om_true), color=ENV, alpha=0.25, lw=0, zorder=0,
                   label=r'$e_\omega(\tau)$  (model-error response, (84))')
    A.plot(t, deg(om_true), '-', lw=0.9, color=TRU, alpha=0.7, zorder=2)
    y_top = deg(y).max() * 1.08
    A.set_ylim(deg(y).min() - 0.6, y_top)
    A.axvline(0, ls=':', lw=1.0, color=NOM)
    A.axvline(p_star[2], ls=':', lw=1.0, color=FIT)
    A.annotate(r'$t_c = 0$', (0, y_top * 0.50), xytext=(6, 0), textcoords='offset points',
               color=NOM, fontsize=9, va='center')
    A.annotate(rf'$t^{{*}} = {1e3 * p_star[2]:+.0f}$ ms', (p_star[2], y_top * 0.40),
               xytext=(-6, 0), textcoords='offset points', color=FIT, fontsize=9,
               va='center', ha='right')
    # brace at the end: e_omega size
    xe = t[-1]
    A.annotate('', (xe, deg(om_nom[-1])), (xe, deg(om_true[-1])),
               arrowprops=dict(arrowstyle='<->', color='#a5690a', lw=1.2))
    A.text(xe + 0.008, 0.5 * deg(om_nom[-1] + om_true[-1]), r'$e_\omega$', color='#a5690a',
           fontsize=10, ha='left', va='center')
    A.set_xlim(t[0] - 0.01, t[-1] + 0.045)
    A.set_ylabel(r'$\omega$ [deg/s]', fontsize=10.5)
    A.legend(fontsize=8, loc='upper left', framealpha=0.92)
    A.set_title('schematic: perturbation and noise exaggerated for visibility', fontsize=9.5,
                color='0.35')

    # ── (b) residuals ───────────────────────────────────────────
    B.plot(t, deg(r_nom), '-', lw=1.4, color=NOM, zorder=3,
           label=rf'$y_i - \hat h(\theta_p^0) = n_i + e_\omega$:  RMS {deg(rms(r_nom)):.2f}')
    B.plot(t, deg(r_fit), '-', lw=1.4, color=FIT, zorder=4,
           label=rf'$r_i = y_i - \hat h(\theta^{{*}})$:  RMS {deg(rms(r_fit)):.2f}')
    B.axhline(0, lw=0.5, color='0.5')
    B.set_ylabel('residual [deg/s]', fontsize=10.5)
    B.set_xlabel(r'$\tau$ [s]  (true onset at 0)', fontsize=10.5)
    B.legend(fontsize=8, loc='upper left', framealpha=0.92)
    B.set_ylim(deg(r_nom).min() * 1.4, deg(r_nom).max() * 1.75)
    B.text(0.98, 0.93,
           rf'$\mathrm{{RMS}}(r) = {deg(rms(r_fit)):.2f} \leq \mathrm{{RMS}}(n+e_\omega) = {deg(rms(r_nom)):.2f}$'
           '\n'
           rf'$\leq \mathrm{{RMS}}(e_\omega) + \mathrm{{RMS}}(n) = {deg(rms(e_full)):.2f} + {deg(rms(noise)):.2f}$',
           transform=B.transAxes, fontsize=8.2, ha='right', va='top',
           bbox=dict(fc='white', ec='0.8', lw=0.6, pad=3))
    for ax in (A, B):
        ax.tick_params(labelsize=8.5)
        ax.grid(alpha=0.25, lw=0.4)
    fig.savefig(out, dpi=600, bbox_inches='tight')
    print('written', out)


if __name__ == '__main__':
    main()
