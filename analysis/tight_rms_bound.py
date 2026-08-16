#!/usr/bin/env python3
"""The tightened RMS residual bound, checked per run and drawn.

The old cap RMS(E) + RMS(n) bounds the residual through the deviation
e_omega itself, and that is why it is loose: the minimiser absorbs
whatever part of e_omega the model family can represent, and the cosh
impulse responses that build e_omega lie mostly inside
span{cosh(C2 tau) - 1, 1} -- a shifted cosh is a combination of
cosh(C2 tau) and sinh(C2 tau) at the SAME exponent, and the first is in
the span exactly.  Only the orthogonal remainder can appear in the
residual, and it has an a priori bound:

    RMS(r) <= Phi + RMS((I-P)n),
    Phi = ||rho_bar * sum_j ds_j |(I-P)k_j| ||_RMS,

with k_j(tau) = cosh(C2 (tau - s_j))/J_P the rate-deviation kernel
column for an impulse at sample j and P the projector onto the family's
span at the pinned C2.  Phi runs 0.26 to 0.40 deg/s across the rates --
a factor 4.3 to 12 below RMS(E).  Verified on the exact nonlinear
tip-over: the kernel reconstructs e_omega to 3e-4 deg/s and Phi holds
with a factor ~4 to spare (2x from rho_bar vs the realised |rho|, 2x
from sign coherence the triangle step discards).

The noise term is where the remaining width lives.  RMS((I-P)n) <=
RMS(n) (projections contract), and RMS(n) is estimated from the
in-window content above 5 Hz -- which the model cannot produce, so it is
disturbance by construction -- extended to the full band by a spectral
shape ratio kappa = RMS(<5Hz)/RMS(>5Hz).  Per run kappa is not
predictable (quiet-window kappa and in-window kappa are uncorrelated,
r = 0.04), so the bound uses a campaign constant

    kappa_b = max_runs kappa_quiet = 1.31   (113 runs),

measured entirely OUTSIDE the windows being bounded.  The cap

    cap = Phi + RMS(n_hi) * sqrt(1 + kappa_b^2)

holds on all 140 runs (worst 0.82, mean used 0.58), is FLAT across the
rates, 1.66 to 1.87 deg/s, matching the flat residual, and is below the
old cap at every rate.  The old cap used 0.17 at the slow end and its
noise term was the hi-band content alone -- an underestimate its RMS(E)
slack happened to cover.  The in-window spectra are heavier than the
quiet ones in the median (enhancement 1.60), so on a few runs the
noise term alone would be exceeded (5/140) and Phi's margin covers
them; the conservative variant kappa_b = 1.60 * 1.31 = 2.09 also holds
on all 140 with mean used 0.44.

Usage: python analysis/tight_rms_bound.py [out.png]
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
from failing_runs import split
from rms_check import measure, PHI_BOX

HERE = os.path.dirname(os.path.abspath(__file__))
C_CAP, C_PHI, C_MEAS = '#aed6f1', '#1a5276', '#148f77'


def model_term(d):
    """Phi and the norm-weighted kernel absorption for one run.

    The impulse response of the RATE deviation is
    k(tau, s) = cosh(C2 (tau - s)) / J_P for tau >= s -- the derivative
    of the attitude kernel sinh(C2 (tau - s)) / (J_P C2), and the one
    whose rho_bar-weighted integral is the envelope
    E(tau) = rho_bar sinh(C2 tau) / (J_P C2).  Verified against the
    exact nonlinear solution: e reconstructs to 3e-4 deg/s and Phi
    bounds RMS((I-P)e) with a factor 3.5-4.2 to spare.
    """
    tau, c2, k = d['tau'], d['c2'], d['k']
    jp = 1.0 / (k * c2 ** 2)
    N = len(tau)
    u = np.cosh(np.clip(c2 * tau, 0, 30)) - 1.0
    ut = u - u.mean()
    su2 = float(ut @ ut)
    rb, _, _ = rho_bar(d['axis'], PHI_BOX, d['dm_win'])
    ds = np.gradient(tau)
    T = tau[:, None] - tau[None, :]
    Km = np.where(T >= 0.0,
                  np.cosh(np.clip(c2 * np.maximum(T, 0.0), 0, 30)) / jp,
                  0.0) * ds[None, :]
    R = Km - Km.mean(axis=0)[None, :] \
        - ut[:, None] * ((ut @ Km) / su2)[None, :]
    cn = np.sqrt((R ** 2).sum(axis=0))
    kn = np.sqrt((Km ** 2).sum(axis=0))
    # two rigorous evaluations of the same supremum; the pointwise
    # (row-sum) one is ~25% tighter and is the one reported
    phi_col = rb * (cn / np.sqrt(N)).sum()
    phi_row = rb * np.sqrt(np.sum(np.abs(R).sum(axis=1) ** 2) / N)
    return (float(np.rad2deg(min(phi_col, phi_row))),
            float(1.0 - cn.sum() / kn.sum()))


def enrich(rows):
    """Per run: Phi, absorption, quiet and implied shape ratios."""
    for d in rows:
        d['phi'], d['absorb'] = model_term(d)
        q = np.asarray(d['quiet'], float)
        d['kq'] = None
        if q.size >= 8:
            ql, qh = split(q - q.mean(), d['dt'])
            d['kq'] = float(np.sqrt(np.mean(ql ** 2) / np.mean(qh ** 2)))
        lo2 = max(d['rms_min'] ** 2 - d['rms_n'] ** 2, 0.0)
        d['kimp'] = float(np.sqrt(lo2) / d['rms_n'])
    kqmax = max(d['kq'] for d in rows if d['kq'] is not None)
    s_med = float(np.median([d['kimp'] / d['kq'] for d in rows if d['kq']]))
    # the campaign constant is the largest quiet-window shape ratio --
    # measured entirely outside the windows being bounded.  The
    # enhanced variant s_med * kqmax (in-window spectra are heavier in
    # the median) also holds on all 140 runs and is reported by main()
    # for pricing; it is not needed.
    kb = kqmax
    for d in rows:
        d['tcap'] = d['phi'] + d['rms_n'] * np.sqrt(1.0 + kb ** 2)
    return rows, kqmax, s_med, kb


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'tight_rms_bound.png'
    with open(os.path.join(HERE, '.failing_cache.pkl'), 'rb') as fh:
        rows = measure(pickle.load(fh))
    rows, kqmax, s_med, kb = enrich(rows)
    rates = sorted({d['rate'] for d in rows})
    n = len(rows)
    used = np.array([d['rms_min'] / d['tcap'] for d in rows])
    inside = int(np.sum(used <= 1.0))

    # ---- the bar chart: per-rate means with 95% t intervals --------
    from scipy.stats import t as student

    def stat(vals):
        v = np.asarray(vals)
        return v.mean(), student.ppf(0.975, len(v) - 1) * v.std(ddof=1) \
            / np.sqrt(len(v))

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    fig.subplots_adjust(left=0.095, right=0.985, top=0.86, bottom=0.125)
    i = np.arange(len(rates))
    grp = [[d for d in rows if d['rate'] == rt] for rt in rates]
    ph = np.array([stat([d['phi'] for d in v])[0] for v in grp])
    nz = np.array([stat([d['tcap'] - d['phi'] for d in v])[0] for v in grp])
    cap = np.array([stat([d['tcap'] for d in v])[0] for v in grp])
    cap_ci = np.array([stat([d['tcap'] for d in v])[1] for v in grp])
    mm = np.array([stat([d['rms_min'] for d in v])[0] for v in grp])
    mm_ci = np.array([stat([d['rms_min'] for d in v])[1] for v in grp])

    ax.bar(i - 0.20, ph, 0.36, color=C_PHI,
           label=r'$\Phi$, the unabsorbable model error')
    ax.bar(i - 0.20, nz, 0.36, bottom=ph, color='0.62',
           label=r'$\mathrm{RMS}(n_{\rm hi})\sqrt{1+\kappa_b^2}$,'
                 ' the disturbance')
    ax.errorbar(i - 0.20, cap, yerr=cap_ci, fmt='none', ecolor='0.15',
                elinewidth=1.3, capsize=4, zorder=5)
    ax.bar(i + 0.20, mm, 0.36, color=C_MEAS,
           label=r'measured $\mathrm{RMS}(r)$')
    ax.errorbar(i + 0.20, mm, yerr=mm_ci, fmt='none', ecolor='0.15',
                elinewidth=1.3, capsize=4, zorder=5)

    for k in range(len(rates)):
        lab = f'{mm[k] / cap[k]:.2f}'
        if k == 0:
            lab = 'used\n' + lab
        ax.text(k + 0.20, mm[k] + mm_ci[k] + 0.05, lab, ha='center',
                va='bottom', fontsize=9, color=C_MEAS, weight='bold')

    ax.errorbar([], [], yerr=[], fmt='none', ecolor='0.15',
                elinewidth=1.3, capsize=4, label='mean of 20 runs, 95% CI')
    ax.set_xticks(i)
    ax.set_xticklabels([f'{r:.2f}' for r in rates], fontsize=9)
    ax.set_xlabel(r'$\dot M$ [N m/s]', fontsize=10)
    ax.set_ylabel(r'RMS [$^\circ$/s]', fontsize=10)
    ax.set_title(r'$\mathrm{RMS}(r)\leq\Phi+'
                 r'\mathrm{RMS}(n_{\rm hi})\sqrt{1+\kappa_b^2}$'
                 f' on all {n} runs\n'
                 'the bound is flat, like the residual; used '
                 f'{mm[0] / cap[0]:.2f} to {mm[-1] / cap[-1]:.2f} '
                 '(the old cap: 0.17 to 0.53)', fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.25, lw=0.4, axis='y')
    ax.set_ylim(0, max(cap + cap_ci) * 1.30)
    fig.savefig(out, dpi=150)

    # ---- the table ------------------------------------------------
    print(f"\n  wrote {out}\n")
    print(f"  kappa_quiet max = {kqmax:.3f} over "
          f"{sum(1 for d in rows if d['kq'] is not None)} runs, "
          f"median in-window enhancement s = {s_med:.3f}, "
          f"kappa_b = {kb:.3f}, cap factor sqrt(1+kb^2) = "
          f"{np.sqrt(1 + kb ** 2):.3f}\n")
    print(f"  {'Mdot':>6}{'Phi':>8}{'RMS(E)':>8}{'E/Phi':>7}{'new cap':>9}"
          f"{'old cap':>9}{'rms_min':>9}{'used':>6}{'old used':>9}"
          f"{'inside':>8}")
    for rt in rates:
        v = [d for d in rows if d['rate'] == rt]
        m = lambda kk: np.mean([q[kk] for q in v])
        i = sum(1 for d in v if d['rms_min'] <= d['tcap'])
        print(f"  {rt:6.2f}{m('phi'):8.3f}{m('rms_E'):8.3f}"
              f"{m('rms_E') / m('phi'):7.0f}{m('tcap'):9.3f}{m('cap'):9.3f}"
              f"{m('rms_min'):9.3f}{m('rms_min') / m('tcap'):6.2f}"
              f"{m('rms_min') / m('cap'):9.2f}{i:5d}/20")
    print(f"\n  inside {inside}/{n}; per-run used p10 "
          f"{np.percentile(used, 10):.2f} median "
          f"{np.percentile(used, 50):.2f} p90 {np.percentile(used, 90):.2f} "
          f"max {used.max():.2f}")
    print(f"  norm-weighted kernel absorption: min "
          f"{100 * min(d['absorb'] for d in rows):.1f}%, median "
          f"{100 * np.median([d['absorb'] for d in rows]):.1f}% across runs")
    fmin = max((d['rms_min'] - d['phi']) / d['rms_n'] for d in rows)
    print(f"  smallest cap factor any bound of this form could use: "
          f"{fmin:.3f} (ours {np.sqrt(1 + kb ** 2):.3f}); at that limit "
          f"the mean used would be "
          f"{np.mean([d['rms_min'] / (d['phi'] + d['rms_n'] * fmin) for d in rows]):.2f}")
    ke = s_med * kqmax
    fe = np.sqrt(1 + ke ** 2)
    ue = np.array([d['rms_min'] / (d['phi'] + d['rms_n'] * fe)
                   for d in rows])
    print(f"  enhanced variant kappa_b = {ke:.2f}: inside "
          f"{int((ue <= 1).sum())}/{n}, mean used {ue.mean():.2f}")
    print(f"  runs where the residual exceeds the noise term alone: "
          f"{sum(1 for d in rows if d['rms_min'] > d['rms_n'] * np.sqrt(1 + kb ** 2))}/{n}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
