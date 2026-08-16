#!/usr/bin/env python3
"""The Sec. VIII error figure: the check in deg/s, the answer in mm.

The two domains are kept apart on purpose, one row each, because they
answer different questions and mixing them misleads.  The top row asks
whether the measured gyro record stays inside the envelope the error
analysis predicts, and it is in deg/s throughout -- converting a
per-direction residual into millimetres would suggest it is an offset
error, which it is not.  The bottom row asks what the identification
actually reports and how far it can be trusted, and that is in
millimetres because the answer is a length.

    (a) one run inside the band: the measured rate, the fitted curve and
        the envelope rho_bar sinh(C2 tau)/(J_P C2) evaluated at its
        supremum, widened by three times the in-window disturbance.
    (b) the same test on every sample of all 140 runs, normalised so the
        envelope is 1.  140/140 runs are entirely inside.
    (c) the RMS form (VIII.3), deployed fit against family minimiser.
        The bound is about the minimiser; the deployed fit has C2 and K
        pinned per configuration, which is what puts 26 runs outside.
    (d) M_crit against the ramp rate.  It is a static property, so the
        ideal is flat; every departure is error and no model is used.
    (e) the reported half-sum offset, repeated seven times per
        configuration, against the a priori bound.
    (f) what the pinned gain is worth: the reported offset while K is
        swept over the whole degenerate ridge, a factor of three.

Reads the caches written by
    python analysis/failing_runs.py    DataSet/exp
    python analysis/stage1_gain_fix.py DataSet/exp

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
from fit_quality_bound import rho_bar
from failing_runs import split, amplitude_best

W_MIN = 30.08
PHI_BOX = np.deg2rad(10.0)
HERE = os.path.dirname(os.path.abspath(__file__))
C_A, C_B, C_C, C_D = '#2874a6', '#c0392b', '#148f77', '0.45'


def load(name):
    with open(os.path.join(HERE, name), 'rb') as fh:
        return pickle.load(fh)


def prepare(rows):
    """Per run: the envelope, the deployed residual and the free one."""
    for d in rows:
        tau, c2, k, dt = d['tau'], d['c2'], d['k'], d['dt']
        jp = 1.0 / (k * c2 ** 2)
        rb, _, _ = rho_bar(d['axis'], PHI_BOX, d['dm_win'])
        E = rb * np.sinh(np.clip(c2 * tau, 0, 30)) / (jp * c2)
        w = np.gradient(tau)
        w[0] *= 0.5
        w[-1] *= 0.5
        T = float(w.sum())
        rms = lambda v: float(np.sqrt(np.sum(v ** 2 * w) / T))
        lo, hi = split(d['r'], dt)
        c1a, ra = amplitude_best(tau, d['om'], c2)
        _, hif = split(ra, dt)
        u = np.cosh(np.clip(c2 * tau, 0, 30)) - 1.0
        cap = rms(E) + rms(hif)          # (VIII.3): full residual, in-window n
        amp = abs(abs(c1a) - k * d['md_full']) * rms(u)
        d['E'] = E
        d['band'] = float(E.max()) + 3.0 * float(np.sqrt(np.mean(hi ** 2)))
        d['fit'] = d['om'] - d['r']
        d['inside'] = np.abs(d['r']) <= d['band']
        d['ratio'] = rms(d['r']) / cap
        d['ratio_f'] = rms(ra) / cap
        d['ratio_a'] = rms(d['r']) / (cap + amp)
    return rows


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'sec8_figure.png'
    rows = prepare(load('.failing_cache.pkl'))
    ridge = load('.ridge_summary.pkl')
    d2 = np.rad2deg
    rates = sorted({d['rate'] for d in rows})
    cmap = plt.get_cmap('viridis')
    col = {r: cmap(i / max(len(rates) - 1, 1)) for i, r in enumerate(rates)}

    fig = plt.figure(figsize=(15.5, 8.8))
    gs = fig.add_gridspec(2, 3, hspace=0.46, wspace=0.27,
                          left=0.072, right=0.985, top=0.885, bottom=0.085)

    # ---- (a) one run inside the band -------------------------------
    ax = fig.add_subplot(gs[0, 0])
    cand = [d for d in rows if d['rate'] == rates[-1]]
    shown = max(cand, key=lambda z: np.max(np.abs(z['r'])) / z['band'])
    d = shown
    tau, fit = d['tau'], d['fit']
    s = np.sign(np.mean(fit)) or 1.0
    ax.fill_between(tau, d2(s * fit - d['band']), d2(s * fit + d['band']),
                    color=C_A, alpha=0.16,
                    label=r'$\pm(\sup E + 3\sigma)$')
    ax.plot(tau, d2(s * fit), '-', color=C_A, lw=2.0, label='fitted cosh')
    ax.plot(tau, d2(s * d['om']), '.', color='0.15', ms=4.0,
            label='measured')
    ax.set_title(f"(a) one run inside the envelope\n"
                 f"{d['case'].replace('case_', 'case ')}/{d['axis']},"
                 f" $\\dot M = {d['rate']:.2f}$,"
                 f" {int(d['inside'].sum())}/{len(tau)} samples inside",
                 fontsize=10.5)
    ax.set_xlabel(r'$\tau$ after onset [s]', fontsize=9)
    ax.set_ylabel(r'$|\omega|$ [$^\circ$/s]', fontsize=9)
    j = int(0.45 * len(tau))
    ax.annotate('the pinned gain sits low;\nthe envelope absorbs it',
                (tau[j], d2(s * fit[j])), xycoords='data',
                xytext=(0.52, 0.16), textcoords='axes fraction', fontsize=8,
                color='0.25', ha='left',
                bbox=dict(fc='white', ec='0.75', lw=0.6, pad=2.4),
                arrowprops=dict(arrowstyle='->', color='0.5', lw=0.8))
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.25, lw=0.4)

    # ---- (b) every sample of all 140 runs --------------------------
    ax = fig.add_subplot(gs[0, 1])
    worst = 0.0
    full = 0
    for d in rows:
        u = d['tau'] / d['tau'][-1]
        y = np.abs(d['r']) / d['band']
        ax.plot(u, y, '-', color=col[d['rate']], lw=0.55, alpha=0.55)
        worst = max(worst, float(y.max()))
        full += bool(np.all(d['inside']))
    ax.axhline(1.0, color=C_B, lw=2.0, ls='--',
               label=r'envelope $\sup E + 3\sigma$')
    ax.set_ylim(0, max(1.25, worst * 1.08))
    ax.set_title(f'(b) every sample of all {len(rows)} runs\n'
                 f'{full}/{len(rows)} runs entirely inside,'
                 f' worst sample {worst:.2f}', fontsize=10.5)
    ax.set_xlabel(r'$\tau/\tau_{\rm end}$', fontsize=9)
    ax.set_ylabel('residual / envelope', fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(rates[0], rates[-1]))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.045)
    cb.set_label(r'$\dot M$ [N m/s]', fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.25, lw=0.4)

    # ---- (c) the RMS form, deployed against the minimiser ----------
    ax = fig.add_subplot(gs[0, 2])
    rng = np.random.RandomState(0)
    for lab, key, cc, mk in (('deployed fit, $C_2$ and $K$ pinned',
                              'ratio', C_B, 'o'),
                             ('deployed $+\\,|\\Delta C_1|$ term',
                              'ratio_a', '#e08214', 's'),
                             ('family minimiser, amplitude freed',
                              'ratio_f', C_C, '^')):
        x = np.array([d['rate'] for d in rows])
        x = x * (1.0 + 0.035 * rng.uniform(-1, 1, len(x)))
        ax.plot(x, [d[key] for d in rows], mk, color=cc, ms=4.2,
                alpha=0.65, mew=0,
                label=f"{lab}: "
                      f"{sum(1 for d in rows if d[key] <= 1)}/{len(rows)}")
    ax.axhline(1.0, color='k', lw=1.6, ls='--')
    ax.set_yscale('log')
    ax.set_title('(c) the RMS form (VIII.3)\n'
                 'full residual, in-window noise, no frequency split',
                 fontsize=10.5)
    ax.set_xlabel(r'$\dot M$ [N m/s]', fontsize=9)
    ax.set_ylabel(r'$\mathrm{RMS}(r)\,/\,$cap', fontsize=9)
    ax.legend(fontsize=7.8, loc='lower right')
    ax.grid(alpha=0.25, lw=0.4, which='both')

    # ---- (d) M_crit against the ramp rate --------------------------
    ax = fig.add_subplot(gs[1, 0])
    g = collections.defaultdict(list)
    for d in rows:
        g[(d['case'], d['axis'], d['sign'])].append(d)
    by = collections.defaultdict(list)
    for v in g.values():
        m = np.array([abs(d['mcrit']) for d in v])
        o = np.argsort([d['rate'] for d in v])
        ax.plot(np.array([d['rate'] for d in v])[o],
                100 * (m[o] / m.mean() - 1), '-', color=C_D, lw=0.8,
                alpha=0.55)
        for d, mi in zip(v, m):
            by[d['rate']].append(100 * (mi / m.mean() - 1))
    ax.plot(rates, [np.median(by[r]) for r in rates], 'o-', color=C_B,
            lw=2.4, ms=7, label='median of 20 groups')
    ax.axhline(0, color='k', lw=1.0, ls=':')
    ax.set_title('(d) $M_{\\rm crit}$ is static, so this should be flat\n'
                 '20 configuration/direction groups, no model',
                 fontsize=10.5)
    ax.set_xlabel(r'$\dot M$ [N m/s]', fontsize=9)
    ax.set_ylabel('deviation from own mean [%]', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.4)

    # ---- (e) the reported offset, per configuration ----------------
    ax = fig.add_subplot(gs[1, 1])
    hs = collections.defaultdict(dict)
    for d in rows:
        hs[(d['case'], d['axis'])].setdefault(
            round(d['rate'], 2), {})[d['sign']] = d['mcrit']
    lab, s_all, s_fast = [], [], []
    for k in sorted(hs):
        a = [0.5 * (v[1] + v[-1]) for v in hs[k].values()
             if 1 in v and -1 in v]
        f = [0.5 * (v[1] + v[-1]) for rt, v in hs[k].items()
             if rt >= 0.65 and 1 in v and -1 in v]
        if len(a) >= 3 and len(f) >= 3:
            lab.append(k[0].replace('case_', '') + '/' + k[1])
            s_all.append(1e3 * np.std(a, ddof=1) / W_MIN)
            s_fast.append(1e3 * np.std(f, ddof=1) / W_MIN)
    i = np.arange(len(lab))
    ax.bar(i - 0.19, s_all, 0.36, color=C_D, label='all 7 rates')
    ax.bar(i + 0.19, s_fast, 0.36, color=C_C, label=r'$\dot M \geq 0.65$')
    ax.axhline(0.400, color=C_B, ls='--', lw=1.6,
               label=r'(108) bound on $\rho$, 0.400 mm')
    ax.set_xticks(i)
    ax.set_xticklabels(lab, rotation=60, fontsize=7, ha='right')
    ax.set_title(f'(e) the reported offset, repeated 7 times\n'
                 f'median {np.median(s_all):.2f} mm,'
                 f' {np.median(s_fast):.2f} on the fast subset',
                 fontsize=10.5)
    ax.set_ylabel(r'spread, 1$\sigma$ [mm]', fontsize=9)
    ax.legend(fontsize=7.6)
    ax.grid(alpha=0.25, lw=0.4, axis='y')

    # ---- (f) the offset along the gain ridge -----------------------
    ax = fig.add_subplot(gs[1, 2])
    sc = np.array(sorted(ridge['scales']))
    sh = np.array([[1e3 * (c['half'][s] - c['half'][1.0]) / W_MIN
                    for c in ridge['per_cfg'].values()] for s in sc])
    sp = np.array([[c['spread'][s] for c in ridge['per_cfg'].values()]
                   for s in sc])
    ax.fill_between(sc, sh.min(1), sh.max(1), color=C_A, alpha=0.16,
                    label='all 10 configurations')
    ax.plot(sc, np.median(sh, 1), '-', color=C_A, lw=2.2, label='median')
    ax.plot(sc, np.median(sp, 1) - np.median(sp[np.argmin(np.abs(sc - 1.0))]),
            '--', color=C_C, lw=2.0, label=r'change in $1\sigma$ spread')
    ax.axvline(1.0, color=C_B, lw=1.6, ls=':')
    ax.annotate('stage 2', (1.0, ax.get_ylim()[1]),
                textcoords='offset points', xytext=(4, -12), fontsize=8.5,
                color=C_B)
    ax.axhline(0, color='k', lw=0.9, ls=':')
    ax.set_title('(f) the gain the residual cannot pin down\n'
                 r'$K \to sK$ over the degenerate ridge', fontsize=10.5)
    ax.set_xlabel(r'gain scale $s$', fontsize=9)
    ax.set_ylabel('shift of the reported offset [mm]', fontsize=9)
    ax.legend(fontsize=7.8, loc='lower left')
    ax.grid(alpha=0.25, lw=0.4)

    fig.text(0.007, 0.685, 'the check, in the gyro domain',
             fontsize=10.5, style='italic', color='0.35',
             rotation=90, va='center')
    fig.text(0.007, 0.245, 'the answer, in millimetres',
             fontsize=10.5, style='italic', color='0.35',
             rotation=90, va='center')
    fig.suptitle('Sec. VIII: the error of the cosh method, verified on the '
                 f'{len(rows)}-run campaign', fontsize=13.5, y=0.968)
    fig.savefig(out, dpi=145)
    print(f"\n  wrote {out}\n")
    print(f"  (a) {shown['case']}/{shown['axis']} at {shown['rate']:.2f},"
          f" band {d2(shown['band']):.2f} deg/s")
    print(f"  (b) {full}/{len(rows)} runs entirely inside,"
          f" worst sample {worst:.3f} of the envelope")
    print(f"  (c) deployed {sum(1 for r in rows if r['ratio'] <= 1)}/"
          f"{len(rows)}, with the amplitude term"
          f" {sum(1 for r in rows if r['ratio_a'] <= 1)}/{len(rows)},"
          f" minimiser {sum(1 for r in rows if r['ratio_f'] <= 1)}/{len(rows)}")
    print(f"  (e) spread median {np.median(s_all):.3f} mm,"
          f" fast subset {np.median(s_fast):.3f}")
    print(f"  (f) over s = {sc[0]:.2f} to {sc[-1]:.2f} the offset moves"
          f" at most {np.abs(np.median(sh, 1)).max():.3f} mm in the median,"
          f" {np.abs(sh).max():.3f} worst")
    return 0


if __name__ == '__main__':
    sys.exit(main())
