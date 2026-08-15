#!/usr/bin/env python3
"""Validate (109) the only way it can be validated: forwards.

The bound is on Delta M_crit, the displacement of the identified onset.
It cannot be checked against a fit residual, and the attempt is not
merely loose but ill-posed: the minimiser chooses the onset to minimise
the residual, so whatever part of the deviation lies along
chi ~ sinh(C2 tau) is removed by construction.  The residual is
therefore everything orthogonal to the estimator's span, and
Delta M_crit is precisely the chi-projection the residual discards.  The
two partition the deviation between them and neither carries information
about the other.

So the check runs the other way.  Take the perturbation along the
measured trajectory, project out the estimator's own degrees of freedom
{1, tau, dphi}, propagate with the Duhamel integral, correlate against
chi, and compare the result with the bound.  Nothing in that chain reads
the residual.

Panel A, one run: the perturbation, the normalised weight it is
correlated against, and the resulting displacement against (109).
Panel B, all runs: the same displacement per run, by ramp rate, on the
bound.

Reads error_budget_runs.csv from analysis/error_budget.py, which must
have been run first with the reported (frozen two-stage) constants.

Usage: python analysis/bound_validation_figure.py CSVDIR [out.png]
"""
import collections
import contextlib
import csv
import io
import os
import sys
from pathlib import Path

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

BOUND_MNM = 12.04                # (109), roll, worst admissible
BOUND_PITCH = 9.69
SHOW = ('case_01', 'Mx', 1.20)
CH = ('gravity', 'ge', 'bilinear', 'ramp')
COL = {'gravity': '#c0392b', 'ge': '#2874a6', 'bilinear': '#8e44ad',
       'ramp': '#e67e22'}


def panel_a(ax_rho, ax_w, ax_bar):
    """The chain on one run: rho, the weight, and what they give."""
    case, ad, rate = SHOW
    axis = 'x' if ad == 'Mx' else 'y'
    c2, k = PNLS_CONSTANTS[(case, ad)]
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(ROOT / case / ad)
    bag = next(b for b in bags
               if cvp.commanded_ramp_rate(b.name) == rate
               and b.name.lower().startswith('pos'))
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
    tau = t[j:] - t[j]
    pred = pw['omega_pred'][j:] - float(pw['c'])
    phi = np.concatenate([[0.0], np.cumsum(
        0.5 * (pred[1:] + pred[:-1]) * np.diff(tau))])
    W_, arm = 31.59, 0.160
    rho_phi = 0.5 * W_ * arm * phi ** 2          # the gravity remainder
    ax_rho.plot(tau, 1e3 * rho_phi, '-', lw=1.8, color=COL['gravity'])
    ax_rho.set_xlabel(r'$\tau$ from onset [s]', fontsize=8.5)
    ax_rho.set_ylabel(r'$\rho_\varphi$ [mN$\cdot$m]', fontsize=8.5)
    ax_rho.set_title(r'1. $\rho_\varphi$ along the measured trajectory,'
                     '\n     before the span projection', fontsize=9.5)
    ax_rho.grid(alpha=0.25, lw=0.4)
    ax_rho.tick_params(labelsize=7.5)

    # the normalised weight of (107): w(s) ~ int_s^end sinh cosh, int w = 1
    sh = np.sinh(np.clip(c2 * tau, 0, 30))
    wgt = np.array([np.trapz(np.sinh(c2 * tau[i:])
                             * np.cosh(c2 * (tau[i:] - tau[i])), tau[i:])
                    for i in range(len(tau))])
    wgt = wgt / np.trapz(wgt, tau)
    ax_w.plot(tau, wgt, '-', lw=1.8, color='#1e8449')
    ax_w.fill_between(tau, 0, wgt, color='#1e8449', alpha=0.15)
    ax_w.set_xlabel(r'$s$ from onset [s]', fontsize=8.5)
    ax_w.set_ylabel(r'$w(s)$ [1/s]', fontsize=8.5)
    ax_w.set_title(r'2. the weight, $\int w = 1$, non-increasing',
                   fontsize=9.5)
    ax_w.grid(alpha=0.25, lw=0.4)
    ax_w.tick_params(labelsize=7.5)
    return float(np.trapz(rho_phi * wgt, tau))


def main():
    csvdir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT
    out = sys.argv[2] if len(sys.argv) > 2 else 'bound_validation.png'
    rows = list(csv.DictReader(open(csvdir / 'error_budget_runs.csv')))
    for r in rows:
        r['rate'] = float(r['rate'])
        r['tot'] = abs(float(r['dM_total_mNm']))
        for c in CH:
            r[c] = abs(float(r[f'dM_{c}_mNm']))

    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.30,
                          height_ratios=[1, 1.25])
    ax_rho = fig.add_subplot(gs[0, 0])
    ax_w = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[0, 2])
    got = panel_a(ax_rho, ax_w, ax_bar)

    # The chain's own answer for this run, taken from the budget CSV so
    # that the span projection -- which panel 1 deliberately shows without
    # -- is the same one every other number here uses.
    key = (SHOW[0], SHOW[1], SHOW[2])
    this = [r for r in rows if (r['case'], r['axis']) == key[:2]
            and r['rate'] == key[2] and r['dir'] == 'pos']
    val = abs(float(this[0]['dM_total_mNm'])) if this else float('nan')
    raw = 1e3 * got
    ax_bar.set_xscale('log')
    ax_bar.set_xlim(1e-3, 60)
    ax_bar.set_ylim(-1, 2)
    for y, v, c, lab in ((1.2, raw, '0.60', r'$\langle w,\rho\rangle$, no projection'),
                         (0.4, val, COL['gravity'], r'after the projection')):
        ax_bar.plot([1e-3, v], [y, y], '-', lw=9, color=c, solid_capstyle='butt')
        ax_bar.text(v * 1.25, y, f'{v:.3f}', fontsize=8, va='center')
        ax_bar.text(1.3e-3, y + 0.30, lab, fontsize=7.5, va='bottom',
                    color='0.25')
    ax_bar.axvline(BOUND_MNM, color='k', lw=2.2)
    ax_bar.text(BOUND_MNM * 1.1, 1.75, '(109)\n12.04', fontsize=8,
                va='top')
    ax_bar.set_yticks([])
    ax_bar.set_xlabel(r'$|\Delta M_{\rm crit}|$ [mN$\cdot$m]', fontsize=8.5)
    ax_bar.set_title(r'3. what reaches the threshold', fontsize=9.5)
    ax_bar.grid(alpha=0.25, which='both', lw=0.4, axis='x')
    ax_bar.tick_params(labelsize=7.5)

    ax = fig.add_subplot(gs[1, :])
    rates = sorted({r['rate'] for r in rows})
    rng = np.random.default_rng(0)
    for c in CH:
        xs = [r['rate'] * (1 + 0.04 * rng.standard_normal()) for r in rows]
        ax.plot(xs, [max(r[c], 1e-4) for r in rows], 'o', ms=3.5, alpha=0.45,
                color=COL[c], label=c)
    xs = [r['rate'] * (1 + 0.04 * rng.standard_normal()) for r in rows]
    ax.plot(xs, [max(r['tot'], 1e-4) for r in rows], 'D', ms=5,
            mfc='none', mec='k', mew=1.0, label='total')
    ax.axhline(BOUND_MNM, color='k', lw=2.2)
    ax.text(rates[0], BOUND_MNM * 1.15, '(109), roll: 12.04 mN$\\cdot$m',
            fontsize=9)
    ax.axhline(BOUND_PITCH, color='0.45', lw=1.6, ls='--')
    ax.text(rates[0], BOUND_PITCH * 0.72, 'pitch: 9.69', fontsize=8.5,
            color='0.35')
    med = [np.median([r['tot'] for r in rows if r['rate'] == q])
           for q in rates]
    ax.plot(rates, med, 's-', lw=2.0, ms=7, color='k', label='total, median')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(rates)
    ax.set_xticklabels([f'{q:g}' for q in rates])
    ax.set_xlabel(r'$\dot M$ [N$\cdot$m/s]', fontsize=9.5)
    ax.set_ylabel(r'$|\Delta M_{\rm crit}|$ per run [mN$\cdot$m]',
                  fontsize=9.5)
    ax.set_title('every run, every channel, against the bound the section '
                 'claims', fontsize=10.5)
    ax.legend(fontsize=8, ncol=5, loc='lower left')
    ax.grid(alpha=0.3, which='both', lw=0.4)
    ax.tick_params(labelsize=8.5)

    fig.suptitle('(109) validated forwards: perturbation to weight to '
                 'displacement. The fit residual appears nowhere.',
                 fontsize=12, y=0.965)
    fig.savefig(out, dpi=140, bbox_inches='tight')

    tot = np.array([r['tot'] for r in rows])
    print(f"  {len(rows)} runs.  |dM_crit| median {np.median(tot):.4f},"
          f" p90 {np.percentile(tot, 90):.4f}, max {tot.max():.4f} mN.m")
    print(f"  bound 12.04 mN.m -> margin {12.04 / np.median(tot):.0f}x median,"
          f" {12.04 / tot.max():.0f}x at worst")
    print(f"  runs inside: {100 * np.mean(tot < BOUND_MNM):.0f}%")
    print(f"  wrote {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
