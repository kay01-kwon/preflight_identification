#!/usr/bin/env python3
"""What the model term costs, route by route, against the same data.

The residual bound is always cap = (model term) + (noise term).  Five
routes evaluate the model term, each needing more of the analysis than
the last, and the noise term is held fixed at the calibrated estimate
n_hat so that only the model term is being compared:

  sup E      the pointwise envelope at its supremum.  Needs rho_bar,
             C2, J_P and nothing else -- the manuscript's (C6) route.
  RMS(E)     the same envelope in RMS, sqrt(B(x)/x) instead of sinh x.
  Phi        the envelope with the projection pushed inside the Duhamel
             superposition: the family absorbs the cosh part of every
             kernel column before rho_bar is invoked (2-column span).
  Phi_4      the same against the full tangent space of the shifted-cosh
             family {dC1, d(dt_c), dC2, dC}.
  RMS(e)     the deviation itself, from the exact nonlinear tip-over.

At the slowest ramp the model term falls 13.4 -> 4.88 -> 0.40 -> 0.28
-> 0.24 deg/s along that ladder, a factor 57 end to end, and the used
fraction of the bound rises 0.07 -> 0.17 -> 0.67 -> 0.73 -> 0.76.

Coverage is the cost, and it is stated rather than hidden: with the
ESTIMATED noise term n_hat the two envelope routes cover 140/140, Phi
covers 136 and the two tightest 126.  The guarantee that covers 140/140
with a projected model term is the kappa_b version of tight_rms_bound.py
-- envelope noise, not estimated.  Tightening the model term alone does
not buy coverage; it buys information.

The horizontal band is the criterion for the stronger statement.  The
fit removes p degrees of freedom from the noise, so RMS(r) <= sigma_n
holds exactly when the unabsorbed model error is below
sqrt(p/N)*sigma_n.  Only the projected routes clear it: with sup E or
RMS(E) one can prove the residual is inside a bound, but not that the
residual IS the noise.

Usage: python analysis/model_term_ladder.py [out.png]
"""
import os
import pickle
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_quality_bound import rho_bar
from rms_check import measure, PHI_BOX
from tight_rms_bound import enrich
from nonlinear_band import exact

HERE = os.path.dirname(os.path.abspath(__file__))
ROUTES = [('$\\sup E$', '#c6dbef'), ('$\\mathrm{RMS}(E)$', '#9ecae1'),
          ('$\\Phi$', '#4292c6'), ('$\\Phi_4$', '#1a5276'),
          ('$\\mathrm{RMS}(e_\\omega)$', '#08306b')]
KEYS = ['supE', 'rms_E', 'phi', 'phi4', 'eth']
C_MEAS, C_CRIT = '#148f77', '#c0392b'


def tangent_phi(tau, c2, jp, rb, ncol):
    """A priori bound on the part of e_omega the family cannot absorb.

    Projects each Duhamel kernel column off the tangent space of the
    shifted-cosh family, then bounds the sum by |rho| <= rb.  ncol = 2
    is span{u, 1} (amplitude and baseline free); 3 adds the onset
    direction sinh; 4 adds the exponent direction tau*sinh.
    """
    N = len(tau)
    ct = np.cosh(np.clip(c2 * tau, 0, 30))
    st = np.sinh(np.clip(c2 * tau, 0, 30))
    cols = [ct - 1.0, np.ones_like(tau)]
    if ncol >= 3:
        cols.insert(1, st)
    if ncol >= 4:
        cols.append(tau * st)
    Q, _ = np.linalg.qr(np.column_stack(cols))
    ds = np.gradient(tau)
    T = tau[:, None] - tau[None, :]
    Km = np.where(T >= 0.0,
                  np.cosh(np.clip(c2 * np.maximum(T, 0.0), 0, 30)) / jp,
                  0.0) * ds[None, :]
    R = Km - Q @ (Q.T @ Km)
    col = rb * (np.sqrt((R ** 2).sum(axis=0)) / np.sqrt(N)).sum()
    row = rb * np.sqrt(np.sum(np.abs(R).sum(axis=1) ** 2) / N)
    return float(np.rad2deg(min(col, row)))


def build():
    with open(os.path.join(HERE, '.failing_cache.pkl'), 'rb') as fh:
        rows, kqmax, s_med, kb = enrich(measure(pickle.load(fh)))
    kqm = float(np.median([d['kq'] for d in rows if d['kq'] is not None]))
    rates = sorted({d['rate'] for d in rows})
    eth = {}
    for m in rates:
        s = exact(m, n=6001)
        eth[m] = float(np.rad2deg(np.sqrt(
            np.trapz(s['e'] ** 2, s['tau']) / s['tau'][-1])))
    for d in rows:
        tau, c2, k = d['tau'], d['c2'], d['k']
        jp = 1.0 / (k * c2 ** 2)
        rb, _, _ = rho_bar(d['axis'], PHI_BOX, d['dm_win'])
        d['supE'] = float(np.rad2deg(
            rb * np.sinh(np.clip(c2 * tau[-1], 0, 30)) / (jp * c2)))
        d['phi4'] = tangent_phi(tau, c2, jp, rb, 4)
        d['eth'] = eth[d['rate']]
        kq = d['kq'] if d['kq'] is not None else kqm
        d['nh'] = d['rms_n'] * np.sqrt(1.0 + (s_med * kq) ** 2)
        d['crit'] = np.sqrt(4.0 / len(tau)) * d['nh']
    return rows, rates


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'model_term_ladder.png'
    rows, rates = build()
    grp = [[d for d in rows if d['rate'] == r] for r in rates]
    mean = lambda v, kk: float(np.mean([d[kk] for d in v]))
    i = np.arange(len(rates))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.6, 5.2))
    fig.subplots_adjust(left=0.055, right=0.995, top=0.805, bottom=0.125,
                        wspace=0.20)

    # ---- (a) the model term itself, and the criterion band ---------
    for (lab, col), kk in zip(ROUTES, KEYS):
        a1.plot(i, [mean(v, kk) for v in grp], 'o-', color=col, lw=2.2,
                ms=7, mec='0.3', mew=0.5, label=lab)
    a1.plot(i, [mean(v, 'rms_min') for v in grp], 's--', color=C_MEAS,
            lw=2.0, ms=7, label=r'measured $\mathrm{RMS}(r)$')
    a1.plot(i, [mean(v, 'crit') for v in grp], ':', color=C_CRIT, lw=2.4,
            label=r'criterion $\sqrt{p/N}\,\hat n$')
    a1.set_yscale('log')
    a1.set_xticks(i)
    a1.set_xticklabels([f'{r:.2f}' for r in rates], fontsize=9)
    a1.set_xlabel(r'$\dot M$ [N m/s]', fontsize=10)
    a1.set_ylabel(r'model term of the cap [$^\circ$/s]', fontsize=10)
    a1.set_title('(a) the model term, five routes\n'
                 'only the projected routes clear the criterion',
                 fontsize=11.5)
    a1.set_ylim(0.09, 32.0)
    a1.legend(fontsize=8.6, loc='lower left', ncol=2, framealpha=0.95)
    a1.grid(alpha=0.25, lw=0.4, which='both')

    # ---- (b) what that buys: the used fraction --------------------
    w = 0.16
    for j, ((lab, col), kk) in enumerate(zip(ROUTES, KEYS)):
        u = [mean(v, 'rms_min') / (mean(v, kk) + mean(v, 'nh'))
             for v in grp]
        ins = sum(1 for d in rows if d['rms_min'] <= d[kk] + d['nh'])
        a2.bar(i + (j - 2) * w, u, w, color=col, ec='0.35', lw=0.5,
               label=f'{lab}  ({ins})')
    a2.axhline(1.0, color='k', lw=1.0, ls=':')
    a2.set_xticks(i)
    a2.set_xticklabels([f'{r:.2f}' for r in rates], fontsize=9)
    a2.set_ylim(0, 1.14)
    a2.set_xlabel(r'$\dot M$ [N m/s]', fontsize=10)
    a2.set_ylabel('fraction of the bound used', fontsize=10)
    a2.set_title('(b) what each route buys, and what it costs\n'
                 'in brackets: runs of $140$ still covered, with $\\hat n$',
                 fontsize=11.5)
    a2.legend(fontsize=8.4, loc='upper left', ncol=5, columnspacing=0.7,
              handlelength=1.2, framealpha=0.95)
    a2.grid(alpha=0.25, lw=0.4, axis='y')

    fig.suptitle('The model term sets how much the residual bound says; '
                 'the noise term is held fixed at $\\hat n$', fontsize=12.5,
                 y=0.975)
    fig.savefig(out, dpi=150)

    print(f"\n  wrote {out}\n")
    print(f"  {'Mdot':>6}" + ''.join(f"{k:>9}" for k in KEYS)
          + f"{'n_hat':>8}{'meas':>7}{'crit':>7}")
    for r, v in zip(rates, grp):
        print(f"  {r:6.2f}" + ''.join(f"{mean(v, k):9.3f}" for k in KEYS)
              + f"{mean(v, 'nh'):8.3f}{mean(v, 'rms_min'):7.3f}"
              f"{mean(v, 'crit'):7.3f}")
    print(f"\n  used fraction (measured / (model + n_hat)), and coverage:")
    print(f"  {'route':>16}{'slow':>8}{'fast':>8}{'inside':>9}"
          f"{'clears criterion':>18}")
    for (lab, _), kk in zip(ROUTES, KEYS):
        u = [mean(v, 'rms_min') / (mean(v, kk) + mean(v, 'nh'))
             for v in grp]
        ins = sum(1 for d in rows if d['rms_min'] <= d[kk] + d['nh'])
        cl = sum(1 for d in rows if d[kk] <= d['crit'])
        name = lab.replace('$', '').replace('\\mathrm', '') \
                  .replace('\\sup', 'sup ').replace('_\\omega', '_omega') \
                  .replace('\\Phi', 'Phi').replace('{', '').replace('}', '')
        print(f"  {name:>16}{u[0]:8.3f}{u[-1]:8.3f}{ins:6d}/140"
              f"{cl:15d}/140")
    return 0


if __name__ == '__main__':
    sys.exit(main())
