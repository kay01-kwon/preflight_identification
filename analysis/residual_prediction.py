#!/usr/bin/env python3
"""Predict the residual from the theory, and compare like with like.

Earlier attempts set the measured residual against the envelope of (90),
which bounds e_omega.  That is the wrong quantity: the minimiser removes
whatever part of e_omega lies along chi, so the residual is only what is
left after the removal.  This predicts THAT instead.

    1. build rho along the measured trajectory, gravity and coupling;
    2. propagate it, e_omega = (1/J_P) int cosh(C2 (tau-s)) rho ds;
    3. remove what the estimator absorbs -- the least-squares projection
       of e_omega onto span{sinh(C2 tau), 1}, the onset and the baseline;
    4. what remains is the residual the theory predicts.

By docs/minimizer_absorption.tex the remainder is P(tau) cosh(C2 tau)/J_P
to leading order, so it should GROW across the window like cosh.

Two comparisons are reported, the second sharper than the first.

  envelope    |r_meas| against |r_theory| pointwise.  Weak, because the
              measured residual carries noise the theory never claimed.
  projection  the coefficient of cosh(C2 tau) in each.  A matched
              filter: noise contributes to it only through its own
              projection, which averages down, so this can detect a
              predicted component well below the noise amplitude.

The amplitude is pinned, so the cosh direction is not absorbed and the
projection test is available at all -- see Sec. 6 of that document.

Usage: python analysis/residual_prediction.py [out.png]
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

W = 31.59
ARMS = {'Mx': 0.160, 'My': 0.130}
BETA = {'Mx': 0.03446, 'My': 0.02573}
SHOWCASE = ('case_01', 'Mx')


def duhamel(tau, rho, c2, j_p):
    """(1/J_P) int_0^tau cosh(C2 (tau-s)) rho(s) ds, trapezoid."""
    out = np.zeros_like(tau)
    for i in range(1, len(tau)):
        k = np.cosh(np.clip(c2 * (tau[i] - tau[:i + 1]), 0, 30))
        out[i] = np.trapz(k * rho[:i + 1], tau[:i + 1]) / j_p
    return out


def remove_span(tau, y, c2):
    """Least-squares removal of {sinh(C2 tau), 1}: the onset and baseline."""
    A = np.column_stack([np.sinh(np.clip(c2 * tau, 0, 30)),
                         np.ones_like(tau)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


def cosh_coeff(tau, y, c2):
    """The coefficient of cosh(C2 tau) - 1, after the span is removed."""
    b = np.cosh(np.clip(c2 * tau, 0, 30)) - 1.0
    b = remove_span(tau, b, c2)
    d = float(b @ b)
    return float(b @ remove_span(tau, y, c2)) / d if d > 0 else np.nan


def collect():
    rows = []
    for (case, ad), (c2, k) in sorted(PNLS_CONSTANTS.items()):
        axis = 'x' if ad == 'Mx' else 'y'
        j_p = 1.0 / (k * c2 ** 2)
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
            if j < 12 or len(om) - j < 12:
                continue
            tau = t[j:] - t[j]
            pred = pw['omega_pred'][j:] - float(pw['c'])
            phi = np.concatenate([[0.0], np.cumsum(
                0.5 * (pred[1:] + pred[:-1]) * np.diff(tau))])
            # the two channels, along the trajectory the run actually flew
            g2 = W * ARMS[ad] * np.cos(phi) - W * 0.30 * np.sin(phi)
            rho = 0.5 * g2 * phi ** 2 + BETA[ad] * md * tau * phi
            e = duhamel(tau, rho, c2, j_p)
            r_th = remove_span(tau, e, c2)
            r_me = om[j:] - pw['omega_pred'][j:]
            rows.append(dict(
                case=case, axis=ad, rate=rate, tau=tau,
                r_th=r_th, r_me=r_me, e=e, rho=rho, c2=c2,
                a_th=cosh_coeff(tau, e, c2),
                a_me=cosh_coeff(tau, r_me, c2),
                out=float((np.abs(r_me) > np.abs(r_th)).mean())))
        print(f"  {case}/{ad} done ({len(rows)} runs)")
    return rows


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'residual_prediction.png'
    rows = collect()
    g = collections.defaultdict(list)
    for r in rows:
        g[r['rate']].append(r)

    print(f"\n  {len(rows)} runs.  All in rad/s.\n")
    print(f"  {'rate':>6}{'n':>4}{'|r| theory':>13}{'|r| measured':>15}"
          f"{'ratio':>9}{'% of window':>14}")
    print(f"  {'':6}{'':4}{'median':>13}{'median':>15}{'':9}{'meas > theory':>14}")
    for rate in sorted(g):
        v = g[rate]
        a = np.median([np.median(np.abs(r['r_th'])) for r in v])
        b = np.median([np.median(np.abs(r['r_me'])) for r in v])
        print(f"  {rate:6.2f}{len(v):4d}{a:13.5f}{b:15.5f}{b / a:9.1f}"
              f"{100 * np.median([r['out'] for r in v]):13.0f}%")

    print(f"\n  the projection test: coefficient of cosh(C2 tau) - 1,"
          f" span removed\n")
    print(f"  {'rate':>6}{'theory':>12}{'measured':>12}{'meas/theory':>14}"
          f"{'measured p10-p90':>22}")
    for rate in sorted(g):
        v = g[rate]
        at = np.median([r['a_th'] for r in v])
        am = np.median([r['a_me'] for r in v])
        lo, hi = np.percentile([r['a_me'] for r in v], [10, 90])
        print(f"  {rate:6.2f}{at:12.5f}{am:12.5f}{am / at:14.1f}"
              f"{lo:12.5f}{hi:10.5f}")
    at = np.array([r['a_th'] for r in rows])
    am = np.array([r['a_me'] for r in rows])
    from scipy import stats
    sp = stats.spearmanr(at, am)
    print(f"\n  Spearman(theory, measured) over all runs ="
          f" {sp[0]:+.3f} (p = {sp[1]:.4f})")
    print(f"  If the predicted component were present and dominant this"
          f" would be near +1;\n  if it is present but buried, positive and"
          f" small; if absent, zero.")

    show = {r['rate']: r for r in rows
            if (r['case'], r['axis']) == SHOWCASE}
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for i, rate in enumerate(sorted(show)):
        ax = axes.flat[i]
        r = show[rate]
        ax.plot(r['tau'], np.abs(r['r_me']), '-', lw=0.9, color='0.30',
                label='measured')
        ax.plot(r['tau'], np.abs(r['r_th']), '-', lw=1.9, color='#c0392b',
                label='predicted')
        ax.set_yscale('log')
        ax.set_title(rf'$\dot M$ = {rate:.2f}', fontsize=9.5)
        ax.set_xlabel(r'$\tau$ [s]', fontsize=8)
        ax.set_ylabel('[rad/s]', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, which='both', lw=0.4)
        if i == 0:
            ax.legend(fontsize=7.5, loc='lower right')
    ax = axes.flat[7]
    ax.loglog(np.abs(at), np.abs(am), 'o', ms=4, alpha=0.5, color='#2874a6')
    lim = [min(np.abs(at).min(), np.abs(am).min()) * 0.5,
           max(np.abs(at).max(), np.abs(am).max()) * 2]
    ax.plot(lim, lim, 'k--', lw=1.2, label='equality')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('predicted cosh coefficient', fontsize=8)
    ax.set_ylabel('measured', fontsize=8)
    ax.set_title('the projection test', fontsize=9.5)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25, which='both', lw=0.4)
    ax.tick_params(labelsize=7)
    fig.suptitle('The residual the theory predicts, against the residual '
                 'measured. Same quantity on both axes.', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=140)
    print(f"\n  wrote {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
