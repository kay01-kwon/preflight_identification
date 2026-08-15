#!/usr/bin/env python3
"""The Sec. VIII error-verification figure, from the real campaign.

Six panels, all measured, none simulated.  Between them they carry the
whole chain: what the a priori bound claims, whether the fit meets it,
what the method actually repeats to, and which domain the residual error
lives in.

    (a) M_crit against the ramp rate.  It is a static property, so the
        ideal is flat; every departure is error, and no model is used.
    (b) the same pooled by rate.  The slowest ramp reads low.
    (c) the domain test: sigma_M flat and sigma_t going as 1/Mdot says
        the dominant error is in the moment, not in the onset.
    (d) the residual against its cap, split at 5 Hz.
    (e) the reported half-sum, per configuration, all rates against the
        fast subset.
    (f) the budget: what is bounded, what is measured.

Reads the caches written by mcrit_repeatability.py and
fit_quality_bound.py; run those first.

Usage: python analysis/sec8_figure.py [out.png]
"""
import collections
import os
import pickle
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_quality_bound import rho_bar, rms_cap

W_MIN = 30.08
HERE = os.path.dirname(os.path.abspath(__file__))
C_A, C_B, C_C, C_D = '#2874a6', '#c0392b', '#148f77', '0.45'


def load(name):
    with open(os.path.join(HERE, name), 'rb') as fh:
        return pickle.load(fh)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'sec8_figure.png'
    mc, fq = load('.mcrit_cache.pkl'), load('.fitq_cache.pkl')
    d2 = np.rad2deg

    g = collections.defaultdict(list)
    for r in mc:
        g[(r['case'], r['axis'], r['sign'])].append(r)

    fig = plt.figure(figsize=(15.5, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.30,
                          left=0.055, right=0.985, top=0.885, bottom=0.085)

    # (a) M_crit against the ramp rate, each group scaled by its own mean
    ax = fig.add_subplot(gs[0, 0])
    by = collections.defaultdict(list)
    for v in g.values():
        m = np.array([abs(r['mcrit']) for r in v])
        o = np.argsort([r['mdot'] for r in v])
        rt = np.array([r['rate'] for r in v])[o]
        ax.plot(rt, 100 * (m[o] / m.mean() - 1), '-', color=C_D, lw=0.8,
                alpha=0.55)
        for r, mi in zip(v, m):
            by[round(r['rate'], 2)].append(100 * (mi / m.mean() - 1))
    rr = sorted(by)
    ax.plot(rr, [np.median(by[k]) for k in rr], 'o-', color=C_B, lw=2.4,
            ms=7, label='median of 20 groups')
    ax.axhline(0, color='k', lw=1.0, ls=':')
    ax.set_title('(a) $M_{\\rm crit}$ is static, so this should be flat\n'
                 '20 configuration/direction groups', fontsize=10.5)
    ax.set_xlabel(r'$\dot M$ [N m/s]', fontsize=9)
    ax.set_ylabel('deviation from own mean [%]', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.4)

    # (b) pooled, in mN.m, with standard errors
    ax = fig.add_subplot(gs[0, 1])
    abs_ = collections.defaultdict(list)
    for v in g.values():
        m = np.array([abs(r['mcrit']) for r in v])
        for r, mi in zip(v, m):
            abs_[round(r['rate'], 2)].append(1e3 * (mi - m.mean()))
    mu = [np.mean(abs_[k]) for k in rr]
    se = [np.std(abs_[k], ddof=1) / np.sqrt(len(abs_[k])) for k in rr]
    ax.errorbar(rr, mu, yerr=se, fmt='o-', color=C_A, lw=2.0, ms=7,
                capsize=4)
    ax.axhline(0, color='k', lw=1.0, ls=':')
    ax.annotate(f'{mu[0]:.1f} mN.m\n$t = -4.0$', (rr[0], mu[0]),
                textcoords='offset points', xytext=(16, -4), fontsize=8.5,
                color=C_B)
    ax.set_title('(b) the slowest ramp reads low\n'
                 'pooled over all groups, $\\pm$1 SE', fontsize=10.5)
    ax.set_xlabel(r'$\dot M$ [N m/s]', fontsize=9)
    ax.set_ylabel(r'$M_{\rm crit}$ less own mean [mN.m]', fontsize=9)
    ax.grid(alpha=0.25, lw=0.4)

    # (c) the domain test
    ax = fig.add_subplot(gs[0, 2])
    sm = np.array([np.std(abs_[k], ddof=1) for k in rr])
    st = sm / np.array(rr)
    ax.plot(rr, sm / sm.mean(), 'o-', color=C_A, lw=2.2, ms=7,
            label=r'$\sigma_M$  (moment domain)')
    ax.plot(rr, st / st.mean(), 's--', color=C_B, lw=2.2, ms=7,
            label=r'$\sigma_t = \sigma_M/\dot M$  (time domain)')
    ax.axhline(1.0, color='k', lw=1.0, ls=':')
    ax.set_yscale('log')
    ax.set_title('(c) which domain is flat?\n'
                 r'$\sigma_M$ is: the error is in the moment', fontsize=10.5)
    ax.set_xlabel(r'$\dot M$ [N m/s]', fontsize=9)
    ax.set_ylabel('normalised to own mean', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.4, which='both')

    # (d) residual against the cap, split at 5 Hz
    ax = fig.add_subplot(gs[1, 0])
    q = collections.defaultdict(list)
    for r in fq:
        q[r['rate']].append(r)
    rt = sorted(q)
    cap, lo, hi = [], [], []
    for k in rt:
        v = q[k]
        cap.append(np.median([
            d2(rms_cap(rho_bar(r['axis'], np.deg2rad(10.0),
                               r['dm_win'])[0], r['k'], r['c2'], r['x']))
            + d2(0.0 if np.isnan(r['noise']) else r['noise']) for r in v]))
        lo.append(np.median([d2(r['res_lo']) for r in v]))
        hi.append(np.median([d2(r['res_hi']) for r in v]))
    i = np.arange(len(rt))
    ax.bar(i - 0.22, cap, 0.30, color=C_D, label=r'cap $\|E\|+\|n\|$')
    ax.bar(i + 0.12, lo, 0.24, color=C_C, label=r'residual $<5$ Hz')
    ax.bar(i + 0.12, hi, 0.24, bottom=lo, color=C_B, alpha=0.75,
           label=r'residual $>5$ Hz')
    ax.set_xticks(i)
    ax.set_xticklabels([f'{k:.2f}' for k in rt], fontsize=8)
    ax.set_title('(d) the fit against its cap, split at 5 Hz\n'
                 'only the low part can be $\\rho$: 114/140 inside',
                 fontsize=10.5)
    ax.set_xlabel(r'$\dot M$ [N m/s]', fontsize=9)
    ax.set_ylabel(r'RMS [$^\circ$/s]', fontsize=9)
    ax.legend(fontsize=7.8)
    ax.grid(alpha=0.25, lw=0.4, axis='y')

    # (e) the reported half-sum, per configuration
    ax = fig.add_subplot(gs[1, 1])
    hs = collections.defaultdict(dict)
    for r in mc:
        hs[(r['case'], r['axis'])].setdefault(
            round(r['rate'], 2), {})[r['sign']] = r['mcrit']
    lab, s_all, s_fast = [], [], []
    for k in sorted(hs):
        a = [0.5 * (d[1] + d[-1]) for d in hs[k].values() if 1 in d and -1 in d]
        f = [0.5 * (d[1] + d[-1]) for rt_, d in hs[k].items()
             if rt_ >= 0.65 and 1 in d and -1 in d]
        if len(a) >= 3 and len(f) >= 3:
            lab.append(k[0].replace('case_', '') + '/' + k[1])
            s_all.append(1e3 * np.std(a, ddof=1) / W_MIN)
            s_fast.append(1e3 * np.std(f, ddof=1) / W_MIN)
    i = np.arange(len(lab))
    ax.bar(i - 0.19, s_all, 0.36, color=C_D, label='all 7 rates')
    ax.bar(i + 0.19, s_fast, 0.36, color=C_C, label=r'$\dot M \geq 0.65$')
    ax.axhline(0.400, color=C_B, ls='--', lw=1.6,
               label='(108) bound, 0.400 mm')
    ax.set_xticks(i)
    ax.set_xticklabels(lab, rotation=60, fontsize=7, ha='right')
    ax.set_title('(e) the reported offset, repeated 7 times\n'
                 'median 0.52 mm, 0.33 on the fast subset', fontsize=10.5)
    ax.set_ylabel('spread, 1$\\sigma$ [mm]', fontsize=9)
    ax.legend(fontsize=7.8)
    ax.grid(alpha=0.25, lw=0.4, axis='y')

    # (f) the budget
    ax = fig.add_subplot(gs[1, 2])
    items = [('small-angle\n+ GE, (108)', 0.400, C_D, 'bounded'),
             ('realised $\\rho$\np90', 0.161 / W_MIN * 1e3 / 1e3, C_D, ''),
             ('rig repeat.\nper direction', 21.3 / W_MIN, C_A, 'measured'),
             ('reported\nall rates', 0.523, C_C, ''),
             ('reported\n$\\dot M\\geq0.65$', 0.327, C_C, '')]
    v = [it[1] for it in items]
    ax.bar(np.arange(len(items)), v,
           color=[it[2] for it in items], width=0.62)
    for j, (nm, val, _, _) in enumerate(items):
        ax.text(j, val * 1.06, f'{val:.3f}', ha='center', fontsize=8.5)
    ax.set_xticks(np.arange(len(items)))
    ax.set_xticklabels([it[0] for it in items], fontsize=7.5)
    ax.set_yscale('log')
    ax.set_ylabel('equivalent offset [mm]', fontsize=9)
    ax.set_title('(f) the budget: bounded on the left,\n'
                 'measured on the right', fontsize=10.5)
    ax.grid(alpha=0.25, lw=0.4, axis='y', which='both')

    fig.suptitle('Sec. VIII: the error of the cosh method, verified on the '
                 'campaign', fontsize=13.5, y=0.965)
    fig.savefig(out, dpi=145)
    print(f"\n  wrote {out}")
    print(f"\n  (b) pooled deviation by rate [mN.m]")
    for k, m, s in zip(rr, mu, se):
        print(f"    {k:5.2f}  {m:+8.2f} +/- {s:.2f}")
    print(f"\n  (e) half-sum spread [mm]: median all rates"
          f" {np.median(s_all):.3f}, fast subset {np.median(s_fast):.3f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
