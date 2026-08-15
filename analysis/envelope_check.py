#!/usr/bin/env python3
"""The deviation bound as an envelope in time, in rad/s, not normalised.

Two things were wrong with reading the bound at the window end and in
per cent of the peak.

The bound (90) is stated in rad/s.  Dividing by the peak mixes in the
peak's own spread -- it runs 0.15 to 0.92 rad/s across these runs -- so
a percentage compares the deviation against a moving yardstick and says
as much about the excursion as about the error.

And it is a bound for every tau, not only for tau_end:

    |e_omega(tau)| <= rho_bar(tau) sinh(C2 tau) / (J_P C2),

with rho_bar(tau) the window average up to tau and the reduction
factors evaluated at x = C2 tau.  Whether the measurement sits inside
that envelope from the onset onwards is a different question from
whether it sits inside at the last sample, and the endpoint is the
worst place to ask: it is where the onset sweep has the most leverage,
so a trade the fit makes early is paid for there.

This evaluates the envelope pointwise.  For every run it reports the
fraction of the post-onset window spent outside it, where the first
crossing is, and by how much at worst -- all in rad/s.

Constants are the frozen two-stage pair, so J_P C2 = 1/(K C2) comes
from the same calibration the curve does.

Usage: python analysis/envelope_check.py [out.png]
"""
import contextlib
import collections
import io
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import critical_value_getter_piecewise as cvp
from pnls_constants import PNLS_CONSTANTS
from utils.extractor import load_excitation_dataset
from constrained_calibration import ROOT

W = 31.59                       # N, ballasted: unfavourable for rho_phi
ARMS = {'Mx': 0.160, 'My': 0.130}
BETA = {'Mx': 0.03446, 'My': 0.02573}
TS = 0.010


def _lam(u):
    return np.sinh(u) - u


def _r_phi(x):
    a = (np.sinh(2 * x) / 4 - x / 2 - 2 * x * np.cosh(x)
         + 2 * np.sinh(x) + x ** 3 / 3)
    return np.where(x > 1e-6, a / np.maximum(x * _lam(x) ** 2, 1e-300), 1 / 7)


def _r_ge(x):
    a = x * np.cosh(x) - np.sinh(x) - x ** 3 / 3
    return np.where(x > 1e-6, a / np.maximum(x ** 2 * _lam(x), 1e-300), 1 / 5)


def envelope(tau, c2, k, m_dot, arm, beta, pred_post):
    """rho_bar(tau) sinh(C2 tau) / (J_P C2), pointwise in tau."""
    phi = np.concatenate([[0.0], np.cumsum(
        0.5 * (pred_post[1:] + pred_post[:-1]) * np.diff(tau))])
    x = c2 * tau
    rho = (0.5 * W * arm * phi ** 2 * _r_phi(x)
           + beta * np.abs(m_dot) * tau * np.abs(phi) * _r_ge(x))
    return rho * np.sinh(x) * k * c2          # 1/(J_P C2) = K C2


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'envelope_check.png'
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
            i0, i1 = cvp.detect_excitation_window(
                sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
            w = slice(i0, i1 + 1)
            t, om, M = sig['t'][w], sig['omega'][w], sig['moment'][w]
            md = float(np.polyfit(t, M, 1)[0])
            pw = cvp.cosh_onset_fit(t, om, np.zeros_like(t), onset_guess=None,
                                    c2_fixed=c2, moment_floor=0.0,
                                    ramp_gain=k, ramp_rate=md)
            j = pw['onset_idx']
            if j < 5 or len(om) - j < 8:
                continue
            tau = t[j:] - t[j]
            res = np.abs(om[j:] - pw['omega_pred'][j:])
            env = envelope(tau, c2, k, md, ARMS[ad], BETA[ad],
                           pw['omega_pred'][j:] - float(pw['c']))
            grid = (TS / 2) * np.abs(np.gradient(pw['omega_pred'][j:], tau))
            out_of = res > env
            first = (np.argmax(out_of) / len(tau)) if out_of.any() else np.nan
            rows.append(dict(
                case=case, axis=ad, rate=rate, tau=tau, res=res, env=env,
                grid=grid, frac=float(out_of.mean()), first=first,
                end_res=float(np.mean(res[-3:])), end_env=float(env[-1]),
                end_grid=float(grid[-1]),
                worst=float(np.max(res - env))))
    print(f"  {len(rows)} runs\n")
    g = collections.defaultdict(list)
    for r in rows:
        g[r['rate']].append(r)

    print("  the endpoint, in rad/s -- the bound's own units\n")
    print(f"  {'rate':>6}{'n':>4}{'|e_w| meas':>12}{'bound':>10}{'grid':>9}"
          f"{'bound+grid':>12}{'runs inside':>13}")
    for rate in sorted(g):
        v = g[rate]
        me = np.median([r['end_res'] for r in v])
        be = np.median([r['end_env'] for r in v])
        gr = np.median([r['end_grid'] for r in v])
        ins = np.mean([r['end_res'] <= r['end_env'] for r in v])
        insg = np.mean([r['end_res'] <= r['end_env'] + r['end_grid']
                        for r in v])
        print(f"  {rate:6.2f}{len(v):4d}{me:12.4f}{be:10.4f}{gr:9.4f}"
              f"{be + gr:12.4f}{100 * ins:8.0f}% /{100 * insg:4.0f}%")
    print("      (last column: inside the bound alone / bound + onset grid)\n")

    print("  pointwise over the post-onset window\n")
    print(f"  {'rate':>6}{'% of window outside':>21}{'first crossing':>17}"
          f"{'worst excess':>15}")
    print(f"  {'':6}{'median':>11}{'p90':>10}{'[frac of window]':>17}"
          f"{'[rad/s]':>15}")
    for rate in sorted(g):
        v = g[rate]
        fr = np.array([r['frac'] for r in v])
        fi = np.array([r['first'] for r in v])
        print(f"  {rate:6.2f}{100 * np.median(fr):10.1f}%"
              f"{100 * np.percentile(fr, 90):9.1f}%"
              f"{np.nanmedian(fi):17.2f}"
              f"{np.median([r['worst'] for r in v]):15.4f}")

    # ── figure: envelope against measurement, one run per rate ───────
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    pick = {}
    for r in rows:
        if r['case'] == 'case_01' and r['axis'] == 'Mx':
            pick[r['rate']] = r
    for i, rate in enumerate(sorted(pick)):
        ax = axes.flat[i]
        r = pick[rate]
        ax.plot(r['tau'], r['res'], '-', lw=1.0, color='0.25',
                label=r'$|\omega-\hat\omega|$')
        ax.plot(r['tau'], r['env'], '-', lw=1.8, color='#c0392b',
                label='(90) envelope')
        ax.plot(r['tau'], r['env'] + r['grid'], '--', lw=1.4,
                color='#2874a6', label='+ onset grid')
        ax.set_yscale('log')
        ax.set_title(rf'$\dot M$ = {rate:.2f} N$\cdot$m/s', fontsize=9.5)
        ax.set_xlabel(r'$\tau$ from onset [s]', fontsize=8)
        ax.set_ylabel(r'[rad/s]', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, which='both', lw=0.4)
        if i == 0:
            ax.legend(fontsize=7, loc='lower right')
    ax = axes.flat[7]
    for rate in sorted(g):
        v = g[rate]
        ax.plot([rate] * len(v), [100 * r['frac'] for r in v], 'o', ms=4,
                color='0.55', alpha=0.6)
        ax.plot(rate, 100 * np.median([r['frac'] for r in v]), 's', ms=8,
                color='#c0392b')
    ax.set_xscale('log')
    ax.set_xticks(sorted(g))
    ax.set_xticklabels([f'{r:g}' for r in sorted(g)])
    ax.set_xlabel(r'$\dot M$ [N$\cdot$m/s]', fontsize=8)
    ax.set_ylabel('% of post-onset window outside (90)', fontsize=8)
    ax.set_title('per run (grey) and median (red)', fontsize=9.5)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.4)
    fig.suptitle('The deviation bound as an envelope in time, case_01/Mx, '
                 'log scale. Grey: measured deviation. Red: (90) evaluated '
                 r'at each $\tau$.', fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=140)
    print(f"\n  wrote {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
