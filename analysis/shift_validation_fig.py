#!/usr/bin/env python3
"""Critical-moment shift validation figure.

Exact nonlinear contact dynamics (true onset known) at each campaign
ramp rate. Orange: the shifted-moment bound of (104). Blue: the
detected-minus-theoretical critical moment through the deployed
constrained readout (Algorithm 1, sub-sample refined), mean and 95%
interval over noise realisations. Grey squares: the free fit, whose
onset lives on the sample grid -- at the fast ramps its shift is
exactly one grid step Mdot*Ts and exceeds the bound (red rings),
a sampling artefact the constrained readout removes.

Usage
-----
  PYTHONPATH=<stubs> python analysis/shift_validation_fig.py [--outdir DIR]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import witness_verify as wv                           # noqa: E402
import critical_value_getter_piecewise as cvp         # noqa: E402

RATES = (0.10, 0.20, 0.30, 0.45, 0.65, 0.90, 1.20)
SIG, BIA, NSEED = 2.45e-4, 2e-3, 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=Path, default=Path('docs'))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rho = wv.R_PHI * 0.5 * wv.W * wv.L_ARM * wv.PHI_MAX ** 2
    ceil, mean, lo, hi, free = [], [], [], [], []
    for mdot in RATES:
        wv.SIGMA, wv.BIAS = 0.0, 0.0
        t, y0, tau, e_om, x = wv.simulate(mdot)       # clean record
        dms = []
        for s in range(NSEED):
            rng = np.random.default_rng(100 + s)
            y = y0 + BIA + rng.normal(0, SIG, len(t))
            pw = cvp.cosh_onset_fit(t, y, np.zeros_like(t),
                                    onset_guess=None, c2_fixed=wv.C2,
                                    moment_floor=0.0, ramp_gain=wv.K,
                                    ramp_rate=mdot)
            dms.append(1e3 * mdot * abs(pw['onset_t']))
        dms = np.array(dms)
        mean.append(dms.mean())
        lo.append(np.percentile(dms, 2.5))
        hi.append(np.percentile(dms, 97.5))
        rng = np.random.default_rng(100)
        y = y0 + BIA + rng.normal(0, SIG, len(t))
        pwf = cvp.cosh_onset_fit(t, y, np.zeros_like(t),
                                 onset_guess=None, c2_fixed=None,
                                 moment_floor=0.0)
        free.append(1e3 * mdot * abs(t[pwf['onset_idx']]))
        u = rho * wv.C2 / mdot
        ceil.append(1e3 * (mdot / wv.C2) * np.arctanh(min(u, 0.999)))
        print(f'{mdot:.2f}: mean {dms.mean():.3f} '
              f'[{lo[-1]:.3f},{hi[-1]:.3f}]  free {free[-1]:.2f}  '
              f'ceil {ceil[-1]:.2f}')

    mean, lo, hi = map(np.array, (mean, lo, hi))
    ix = range(len(RATES))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(ix, ceil, '-o', lw=2.0, ms=4.5, color='#D55E00',
            label='shifted moment (theory)', zorder=4)
    ax.errorbar(ix, mean, yerr=[mean - lo, hi - mean], fmt='o-',
                lw=1.8, ms=5, color='#0072B2', ecolor='#0072B2',
                elinewidth=1.4, capsize=4.5, capthick=1.4, zorder=3,
                label='detected minus theoretical critical moment '
                      '(mean, 95% interval)')
    ax.plot(ix, free, 's', ms=6, mfc='none', mew=1.5, color='0.45',
            label='free fit (grid-quantised)', zorder=3)
    bad = [i for i, (f, c) in enumerate(zip(free, ceil)) if f > c]
    if bad:
        ax.plot([list(ix)[i] for i in bad], [free[i] for i in bad],
                's', ms=8, mfc='none', mew=1.6, color='#CC0000',
                label='exceeds (= one grid step)', zorder=5)
    ax.set_xticks(list(ix))
    ax.set_xticklabels([f'{v:g}' for v in RATES])
    ax.set_xlabel(r'ramp rate $\dot M$ [N·m/s]', fontsize=9.5)
    ax.set_ylabel(r'$\dot M\,|\delta t_c|$ [mN·m]', fontsize=9.5)
    ax.grid(alpha=0.4, lw=0.8, color='0.6')
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.2, loc='upper left', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(args.outdir / 'fig_shift_validation.png',
                bbox_inches='tight', dpi=600)
    print(f'written to {args.outdir}/fig_shift_validation.png')


if __name__ == '__main__':
    main()
