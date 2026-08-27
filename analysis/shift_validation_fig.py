#!/usr/bin/env python3
"""Critical-moment shift validation figure.

Exact nonlinear contact dynamics (true onset known) at each campaign
ramp rate, at the box-worst rho_bar configuration (pitch axis, the
offset rectangle's far corner, loaded mass). Orange: the shifted-moment bound of (104). Blue: the
detected-minus-theoretical critical moment through the calibration-free
PNLS fit (sub-sample refined onset), mean and 95% interval over noise
realisations -- the same estimator the RMSE-bound validation uses, so
both theory checks run on one calibration-free fit.

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

# box-worst rho_bar configuration: pitch axis, the offset rectangle's
# +25 mm corner, loaded mass -- the largest forcing bound the design
# box admits, so the validation runs where the bound is most stressed
LP = 0.140
wv.M_KG, wv.L_ARM, wv.Z = 3.220, LP + 0.025, 0.272
wv.J_CAD = 0.050564
wv.W = wv.M_KG * 9.81
wv.J_P = wv.J_CAD + wv.M_KG * (wv.Z ** 2 + LP ** 2)
wv.C2 = float(np.sqrt(wv.W * wv.Z / wv.J_P))
wv.K = 1.0 / (wv.W * wv.Z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=Path, default=Path('docs'))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rho = wv.R_PHI * 0.5 * wv.W * wv.L_ARM * wv.PHI_MAX ** 2
    ceil, mean, lo, hi = [], [], [], []
    for mdot in RATES:
        wv.SIGMA, wv.BIAS = 0.0, 0.0
        t, y0, tau, e_om, x = wv.simulate(mdot)       # clean record
        dms = []
        for s in range(NSEED):
            rng = np.random.default_rng(100 + s)
            y = y0 + BIA + rng.normal(0, SIG, len(t))
            pw = cvp.cosh_onset_fit(t, y, np.zeros_like(t),
                                    onset_guess=None, c2_fixed=None,
                                    moment_floor=0.0)
            dms.append(1e3 * mdot * abs(pw['onset_t']))
        dms = np.array(dms)
        mean.append(dms.mean())
        lo.append(np.percentile(dms, 2.5))
        hi.append(np.percentile(dms, 97.5))
        u = rho * wv.C2 / mdot
        ceil.append(1e3 * (mdot / wv.C2) * np.arctanh(min(u, 0.999)))
        print(f'{mdot:.2f}: mean {dms.mean():.3f} '
              f'[{lo[-1]:.3f},{hi[-1]:.3f}]  ceil {ceil[-1]:.2f}')

    mean, lo, hi = map(np.array, (mean, lo, hi))
    ix = range(len(RATES))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(ix, ceil, '-o', lw=2.0, ms=4.5, color='#D55E00',
            label='shifted moment bound (theory)', zorder=4)
    ax.errorbar(ix, mean, yerr=[mean - lo, hi - mean], fmt='o-',
                lw=1.8, ms=5, color='#0072B2', ecolor='#0072B2',
                elinewidth=1.4, capsize=4.5, capthick=1.4, zorder=3,
                label=r'critical-moment error $|M_{\mathrm{crit,est}}-M_{\mathrm{crit,ideal}}|$ (mean, 95% interval)')
    bad = [i for i, (h, c) in enumerate(zip(hi, ceil)) if h > c]
    if bad:
        ax.plot([list(ix)[i] for i in bad], [hi[i] for i in bad],
                'o', ms=8, mfc='none', mew=1.6, color='#CC0000',
                label='95% upper cap exceeds', zorder=5)
    ax.set_xticks(list(ix))
    ax.set_xticklabels([f'{v:g}' for v in RATES])
    ax.set_xlabel(r'ramp rate $\dot M$ [N·m/s]', fontsize=9.5)
    ax.set_ylabel(r'$\dot M\,|\delta t_c|$ [mN·m]', fontsize=9.5)
    ax.grid(alpha=0.4, lw=0.8, color='0.6')
    ax.set_axisbelow(True)
    # headroom above the flat theory curve so the legend sits clear
    ax.set_ylim(0, 1.45 * max(ceil))
    ax.legend(fontsize=8.2, loc='upper left', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(args.outdir / 'fig_shift_validation.png',
                bbox_inches='tight', dpi=600)
    print(f'written to {args.outdir}/fig_shift_validation.png')


if __name__ == '__main__':
    main()
