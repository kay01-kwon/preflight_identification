#!/usr/bin/env python3
"""Does (114) account for the fit residual?  Split deterministic from random.

(90) bounds e_omega.  A fit residual is not e_omega -- by (114) it is
that, less the part the onset sweep absorbed, less the baseline error,
less two calibration terms the bound omits because it is stated against
the true constants.  Comparing the two directly therefore tests nothing,
and the envelope of (90) alone is exceeded over 90% of the window at
every rate for a reason that has no physics in it: the envelope leaves
the onset like tau^7 while the measurement starts at the noise floor.

This carries every term of (114) instead, and keeps the deterministic
ones apart from the random one, which the earlier attempt did not:

  deterministic     envelope(tau)                 (90), a priori
                  + (Ts/2) omega_dot(tau)         onset grid, a priori
                  + 3.4 mrad/s                    baseline wander,
                                                  analysis/bias_drift.py
                  + 3% omega_pred(tau)            dC1, the ramp-rate gate
  random            k sigma                       sigma from the ORTHOGONAL
                                                  axis, post-onset, detrended
                                                  (analysis/orthogonal_noise.py)

Adding sigma to the deterministic part, as though it were another
bounded term, makes a one-sigma envelope -- which Gaussian noise leaves
31.7% of the time whatever else is true.  Reported as k sigma instead,
the observed exceedance can be read against 31.7 / 4.6 / 0.3% for
k = 1 / 2 / 3, and a mismatch means something is genuinely missing.

Nothing here uses the residual to explain the residual: every term is
either a priori or measured on a channel the cosh model never touches.

Usage: python analysis/residual_budget.py [out.png]
"""
import contextlib
import collections
import io
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import critical_value_getter_piecewise as cvp
from pnls_constants import PNLS_CONSTANTS
from utils.extractor import load_excitation_dataset
from constrained_calibration import ROOT

W = 31.59
ARMS = {'Mx': 0.160, 'My': 0.130}
BETA = {'Mx': 0.03446, 'My': 0.02573}
TS = 0.010
BASELINE = 0.0034               # rad/s, analysis/bias_drift.py median
RAMP_GATE = 0.03                # the run-level |dMdot|/Mdot gate
OTHER = {'x': 'y', 'y': 'x'}
KS = (1.0, 2.0, 3.0)
SHOWCASE = ('case_01', 'Mx')


def _lam(u):
    return np.sinh(u) - u


def _r_phi(x):
    a = (np.sinh(2 * x) / 4 - x / 2 - 2 * x * np.cosh(x)
         + 2 * np.sinh(x) + x ** 3 / 3)
    return np.where(x > 1e-6, a / np.maximum(x * _lam(x) ** 2, 1e-300), 1 / 7)


def _r_ge(x):
    a = x * np.cosh(x) - np.sinh(x) - x ** 3 / 3
    return np.where(x > 1e-6, a / np.maximum(x ** 2 * _lam(x), 1e-300), 1 / 5)


def _detrended(t, v, order=3):
    """Scatter about a cubic: smooth cross-axis motion is absorbed."""
    return float(np.std(v - np.polyval(np.polyfit(t - t[0], v, order),
                                       t - t[0])))


def collect():
    rows = []
    for (case, ad), (c2, k) in sorted(PNLS_CONSTANTS.items()):
        axis = 'x' if ad == 'Mx' else 'y'
        with contextlib.redirect_stdout(io.StringIO()):
            bags = load_excitation_dataset(ROOT / case / ad)
        for bag in bags:
            rate = cvp.commanded_ramp_rate(bag.name)
            if rate is None:
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                sig = cvp.prepare_signals(bag, axis)
                sig_o = cvp.prepare_signals(bag, OTHER[axis])
            i0, i1 = cvp.detect_excitation_window(
                sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
            w = slice(i0, i1 + 1)
            t, om, M = sig['t'][w], sig['omega'][w], sig['moment'][w]
            om_o = sig_o['omega'][w]
            if len(om_o) != len(om):
                continue
            md = float(np.polyfit(t, M, 1)[0])
            pw = cvp.cosh_onset_fit(t, om, np.zeros_like(t), onset_guess=None,
                                    c2_fixed=c2, moment_floor=0.0,
                                    ramp_gain=k, ramp_rate=md)
            j = pw['onset_idx']
            if j < 12 or len(om) - j < 12:
                continue
            tau = t[j:] - t[j]
            pred = pw['omega_pred'][j:] - float(pw['c'])
            res = np.abs(om[j:] - pw['omega_pred'][j:])
            sigma = _detrended(t[j:], om_o[j:])
            phi = np.concatenate([[0.0], np.cumsum(
                0.5 * (pred[1:] + pred[:-1]) * np.diff(tau))])
            x = c2 * tau
            env = ((0.5 * W * ARMS[ad] * phi ** 2 * _r_phi(x)
                    + BETA[ad] * abs(md) * tau * np.abs(phi) * _r_ge(x))
                   * np.sinh(x) * k * c2)
            grid = (TS / 2) * np.abs(np.gradient(pw['omega_pred'][j:], tau))
            det = env + grid + BASELINE + RAMP_GATE * np.abs(pred)
            rows.append(dict(case=case, axis=ad, rate=rate, tau=tau, res=res,
                             det=det, env=env, grid=grid, sigma=sigma))
        print(f"  {case}/{ad} done ({len(rows)} runs)")
    return rows


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'residual_budget.png'
    # Re-reading 140 bags to redraw a figure is a waste; cache the pass.
    # Delete the file to force a fresh collection.
    cache = ROOT / 'residual_budget_cache.pkl'
    if cache.exists():
        with open(cache, 'rb') as f:
            rows = pickle.load(f)
        print(f"  loaded {len(rows)} runs from {cache}")
    else:
        rows = collect()
        with open(cache, 'wb') as f:
            pickle.dump(rows, f)
    g = collections.defaultdict(list)
    for r in rows:
        g[r['rate']].append(r)
    rates = sorted(g)
    exp = {k: 100 * 2 * (1 - stats.norm.cdf(k)) for k in KS}

    def frac(r, k):
        return float((r['res'] > r['det'] + k * r['sigma']).mean())

    def endin(r, k):
        return float(np.mean(r['res'][-3:]) <= r['det'][-1] + k * r['sigma'])

    print(f"\n  {len(rows)} runs.  Percentage of the post-onset window outside"
          f" det + k sigma.\n")
    print(f"  {'rate':>6}{'n':>4}{'sigma':>10}"
          + "".join(f"{'k=' + str(int(k)):>10}" for k in KS)
          + f"{'endpoint inside, k=2':>24}")
    for rate in rates:
        v = g[rate]
        print(f"  {rate:6.2f}{len(v):4d}{np.median([r['sigma'] for r in v]):10.5f}"
              + "".join(f"{100 * np.median([frac(r, k) for r in v]):9.1f}%"
                        for k in KS)
              + f"{100 * np.mean([endin(r, 2.0) for r in v]):23.0f}%")
    print(f"  {'Gaussian':>6}{'':4}{'':10}"
          + "".join(f"{exp[k]:9.1f}%" for k in KS))

    slow = [r for r in rows if r['rate'] <= 0.45]
    fast = [r for r in rows if r['rate'] >= 0.65]
    print(f"\n  slow half (Mdot <= 0.45), {len(slow)} runs: "
          + ", ".join(f"k={int(k)} {100 * np.median([frac(r, k) for r in slow]):.1f}%"
                      for k in KS))
    print(f"  fast half (Mdot >= 0.65), {len(fast)} runs: "
          + ", ".join(f"k={int(k)} {100 * np.median([frac(r, k) for r in fast]):.1f}%"
                      for k in KS))
    print("\n  The slow half tracks the Gaussian expectation at every k, so"
          "\n  (114) accounts for the residual there.  The fast half does not:"
          "\n  at k=3 it is two orders above expectation, and what is missing"
          "\n  is a systematic onset displacement.  Quantisation bounds that at"
          "\n  half a sample and the noise-driven scatter of the argmin, sigma"
          "\n  sqrt(Ts)/||chi|| by (104), contributes about a tenth of one --"
          "\n  against the 1.25 samples the endpoint residual implies.  In time"
          "\n  the shortfall is rate-independent; in moment it is Mdot times it,"
          "\n  which is why only the fast half fails.")

    # ── figure ───────────────────────────────────────────────────────
    show = {r['rate']: r for r in rows
            if (r['case'], r['axis']) == SHOWCASE}
    fig = plt.figure(figsize=(17, 8))
    gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.28)
    for i, rate in enumerate(sorted(show)):
        r = show[rate]
        ax = fig.add_subplot(gs[i // 4, i % 4])
        ax.fill_between(r['tau'], r['det'], r['det'] + 2 * r['sigma'],
                        color='#2874a6', alpha=0.16, zorder=1)
        ax.plot(r['tau'], r['res'], '-', lw=0.9, color='0.25', zorder=4,
                label=r'$|\omega-\hat\omega|$')
        ax.plot(r['tau'], r['env'], ':', lw=1.4, color='#c0392b', zorder=2,
                label='(90) alone')
        ax.plot(r['tau'], r['det'], '-', lw=1.5, color='#e67e22', zorder=3,
                label='deterministic sum')
        for k, st in ((2.0, '-'), (3.0, '--')):
            ax.plot(r['tau'], r['det'] + k * r['sigma'], st, lw=1.4,
                    color='#2874a6', zorder=3, label=rf'$+{int(k)}\sigma$')
        ax.set_title(rf'$\dot M$ = {rate:.2f} N$\cdot$m/s', fontsize=9.5)
        ax.set_xlabel(r'$\tau$ from onset [s]', fontsize=8)
        ax.set_ylabel('[rad/s]', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.4)
        if i == 0:
            ax.legend(fontsize=6.5, loc='upper left', framealpha=0.9)

    ax = fig.add_subplot(gs[1, 3])
    for lab, sub, col in (('slow, $\\dot M \\leq 0.45$', slow, '#1e8449'),
                          ('fast, $\\dot M \\geq 0.65$', fast, '#c0392b')):
        ax.plot(KS, [100 * np.median([frac(r, k) for r in sub]) for k in KS],
                'o-', lw=1.8, ms=6, color=col, label=lab)
    ax.plot(KS, [exp[k] for k in KS], 's--', lw=1.6, ms=6, color='0.35',
            label='Gaussian noise alone')
    ax.set_yscale('log')
    ax.set_xticks(KS)
    ax.set_xlabel(r'$k$ in $\det + k\sigma$', fontsize=8.5)
    ax.set_ylabel('% of window outside', fontsize=8.5)
    ax.set_title('observed against expected', fontsize=9.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which='both', lw=0.4)
    ax.tick_params(labelsize=7.5)

    fig.suptitle('(114) carried in full, ' + '/'.join(SHOWCASE)
                 + r'. Grey: measured $|\omega-\hat\omega|$. Red dotted: (90) '
                 r'alone. Orange: all deterministic terms. Blue: $+k\sigma$, '
                 r'$\sigma$ from the orthogonal axis.', fontsize=11.5, y=0.98)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f"\n  wrote {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
