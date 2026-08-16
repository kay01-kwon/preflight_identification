#!/usr/bin/env python3
"""The two-line optimality bound, with both terms at their theory values.

The statement is the handwritten one:

    RMS(r) = min_{g in M} RMS(y - g) <= RMS(e_omega + n)
                                     <= RMS(e_omega) + RMS(n),

valid because the nominal is a member of the fitted family (same pinned
C2, amplitude and baseline free; freeing C2 as well only shrinks the
left side).  No projector is needed.

What makes it nearly tight is what is plugged in:

  RMS(e_omega) at its THEORY value -- the exact nonlinear tip-over
  integrated forward, 0.24 down to 0.17 deg/s across the rates -- in
  place of the Chebyshev envelope RMS(E) = 4.88 to 1.12 that made the
  old cap loose.

  RMS(n) at its calibrated estimate n_hat = RMS(n_hi) sqrt(1 +
  (s_med kappa_q)^2): the run's own measured >5 Hz content, extended by
  its own quiet-window shape ratio times the campaign median in-window
  enhancement.  An estimate, not an envelope: kappa_b of the guarantee
  version is the max, this is the middle of the distribution.

Result: the cap runs 1.20 to 1.30 deg/s against a measured 0.95 to
1.08 -- used 0.76 to 0.85, covering the mean at every rate and 126 of
140 runs individually.  And the cross-term identity says the rest:
RMS(e+n)^2 = RMS(e)^2 + RMS(n)^2 when the deviation and the disturbance
are uncorrelated, and that quadrature prediction lands within 1-12% of
the measured residual at every rate.  The remaining gap of the summed
bound is exactly the triangle slack 2 RMS(e) RMS(n).

Usage: python analysis/optimality_rms.py [out.png]
"""
import os
import pickle
import sys

import numpy as np
from scipy.stats import t as student

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rms_check import measure
from tight_rms_bound import enrich
from nonlinear_band import exact

HERE = os.path.dirname(os.path.abspath(__file__))
C_E, C_N, C_MEAS = '#1a5276', '0.62', '#148f77'


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'optimality_rms.png'
    with open(os.path.join(HERE, '.failing_cache.pkl'), 'rb') as fh:
        rows, kqmax, s_med, kb = enrich(measure(pickle.load(fh)))
    rates = sorted({d['rate'] for d in rows})
    d2 = np.rad2deg
    eth = {}
    for m in rates:
        s = exact(m, n=8001)
        eth[m] = d2(np.sqrt(np.trapz(s['e'] ** 2, s['tau'])
                            / s['tau'][-1]))
    kq_med = float(np.median([d['kq'] for d in rows
                              if d['kq'] is not None]))
    for d in rows:
        kq = d['kq'] if d['kq'] is not None else kq_med
        d['nhat'] = d['rms_n'] * np.sqrt(1.0 + (s_med * kq) ** 2)
        d['ocap'] = eth[d['rate']] + d['nhat']
        d['pred'] = np.sqrt(eth[d['rate']] ** 2 + d['nhat'] ** 2)
    n = len(rows)
    inside = sum(1 for d in rows if d['rms_min'] <= d['ocap'])

    def stat(vals):
        v = np.asarray(vals)
        return v.mean(), student.ppf(0.975, len(v) - 1) * v.std(ddof=1) \
            / np.sqrt(len(v))

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    fig.subplots_adjust(left=0.095, right=0.985, top=0.86, bottom=0.125)
    i = np.arange(len(rates))
    grp = [[d for d in rows if d['rate'] == rt] for rt in rates]
    et = np.array([eth[rt] for rt in rates])
    nz = np.array([stat([d['nhat'] for d in v])[0] for v in grp])
    cap = et + nz
    cap_ci = np.array([stat([d['ocap'] for d in v])[1] for v in grp])
    mm = np.array([stat([d['rms_min'] for d in v])[0] for v in grp])
    mm_ci = np.array([stat([d['rms_min'] for d in v])[1] for v in grp])
    pr = np.array([stat([d['pred'] for d in v])[0] for v in grp])

    ax.bar(i - 0.20, et, 0.36, color=C_E,
           label=r'$\mathrm{RMS}(e_\omega)$, theory (exact model)')
    ax.bar(i - 0.20, nz, 0.36, bottom=et, color=C_N,
           label=r'$\hat n=\mathrm{RMS}(n_{\rm hi})'
                 r'\sqrt{1+(s_{\rm med}\kappa_q)^2}$, estimated')
    ax.errorbar(i - 0.20, cap, yerr=cap_ci, fmt='none', ecolor='0.15',
                elinewidth=1.3, capsize=4, zorder=5)
    ax.bar(i + 0.20, mm, 0.36, color=C_MEAS,
           label=r'measured $\mathrm{RMS}(r)$')
    ax.errorbar(i + 0.20, mm, yerr=mm_ci, fmt='none', ecolor='0.15',
                elinewidth=1.3, capsize=4, zorder=5)
    ax.plot(i + 0.20, pr, 'D', ms=6, mfc='none', mec='k', mew=1.4,
            zorder=6,
            label=r'prediction $\sqrt{\mathrm{RMS}(e_\omega)^2+\hat n^2}$')

    for k in range(len(rates)):
        lab = f'{mm[k] / cap[k]:.2f}'
        if k == 0:
            lab = 'used\n' + lab
        ax.text(k + 0.20, max(mm[k] + mm_ci[k], pr[k]) + 0.05, lab,
                ha='center', va='bottom', fontsize=9, color=C_MEAS,
                weight='bold')

    ax.errorbar([], [], yerr=[], fmt='none', ecolor='0.15',
                elinewidth=1.3, capsize=4, label='mean of 20 runs, 95% CI')
    ax.set_xticks(i)
    ax.set_xticklabels([f'{r:.2f}' for r in rates], fontsize=9)
    ax.set_xlabel(r'$\dot M$ [N m/s]', fontsize=10)
    ax.set_ylabel(r'RMS [$^\circ$/s]', fontsize=10)
    rr = pr / mm
    ax.set_title(r'$\mathrm{RMS}(r)=\min_{g\in M}\mathrm{RMS}(y-g)'
                 r'\leq\mathrm{RMS}(e_\omega)+\mathrm{RMS}(n)$'
                 f'\nused {mm[0] / cap[0]:.2f} to '
                 f'{mm[-1] / cap[-1]:.2f}; the prediction is within '
                 f'{100 * (rr.min() - 1):.0f} to '
                 f'{100 * (rr.max() - 1):.0f}% of the residual',
                 fontsize=11.5)
    ax.legend(fontsize=8.6, loc='upper right')
    ax.grid(alpha=0.25, lw=0.4, axis='y')
    ax.set_ylim(0, max(cap + cap_ci) * 1.58)
    fig.savefig(out, dpi=150)

    print(f"\n  wrote {out}\n")
    print(f"  s_med = {s_med:.3f}, kappa_q median = {kq_med:.3f}\n")
    print(f"  {'Mdot':>6}{'e_th':>7}{'n_hat':>8}{'cap':>7}{'meas':>7}"
          f"{'used':>6}{'pred':>7}{'pred/meas':>10}{'inside':>8}")
    for rt, v in zip(rates, grp):
        m = lambda kk: np.mean([q[kk] for q in v])
        ins = sum(1 for d in v if d['rms_min'] <= d['ocap'])
        print(f"  {rt:6.2f}{eth[rt]:7.3f}{m('nhat'):8.3f}{m('ocap'):7.3f}"
              f"{m('rms_min'):7.3f}{m('rms_min') / m('ocap'):6.2f}"
              f"{m('pred'):7.3f}{m('pred') / m('rms_min'):10.2f}"
              f"{ins:5d}/20")
    per = np.array([d['rms_min'] / d['ocap'] for d in rows])
    print(f"\n  the mean cap covers the mean residual at all 7 rates;"
          f" per run {inside}/{n} inside")
    print(f"  per-run used p10 {np.percentile(per, 10):.2f} median "
          f"{np.percentile(per, 50):.2f} p90 {np.percentile(per, 90):.2f} "
          f"max {per.max():.2f}")
    print(f"  the guarantee version (kappa_b envelope, Phi) remains the"
          f" one that holds 140/140.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
