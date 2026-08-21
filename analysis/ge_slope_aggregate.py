#!/usr/bin/env python3
"""The per-run GE slope, aggregated: what the campaign can and cannot
say about the model's attitude dependence.

For every run the trusted-span trace (deployed differentiator, w = 9
order 2, half-width trim, relative attitude, offset-corrected J_P) is
reduced to ONE number, the linear slope of dM_GE against tilt, and the
140 numbers are compared with the interference model's own slope.

Three aggregate statements come out, in decreasing strength:

1. SIGN: the model predicts the GE moment FALLS with tilt (tipping
   about one foot lifts the far rotors, which carry the long arms).
   The per-run sign census tests exactly this prediction.

2. MAGNITUDE: the fitted slopes are dominated by the excursion-
   correlated artefact (r = +0.5 between slope and the run's range),
   so the raw median cannot be read as the GE attitude dependence.

3. EXTRAPOLATION: if the artefact scales like 1/range (an endpoint
   effect diluted by a longer lever arm), regressing slope on
   1/range and reading the intercept at 1/range -> 0 estimates the
   range-free slope.  The bootstrap CI on that intercept is the
   honest statement of what the campaign leaves for the model.

Usage: PYTHONPATH=<stubs> python analysis/ge_slope_aggregate.py [out.png]
"""
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from analysis.ge_trusted_span import collect, K_TRIM, W_SG  # noqa: E402


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'ge_slope_aggregate.png'
    rows = collect()

    sd, sm, rng = [], [], []
    for r in rows:
        ph, gd, gm = r['trace']
        e = len(ph) - K_TRIM
        if e < 15:
            continue
        sd.append(np.polyfit(ph[:e], gd[:e], 1)[0])
        sm.append(np.polyfit(ph[:e], gm[:e], 1)[0])
        rng.append(ph[e - 1] - ph[0])
    sd, sm, rng = map(np.array, (sd, sm, rng))
    n = len(sd)

    neg = int(np.sum(sd < 0))
    print(f"\n  {n} runs (w={W_SG}, trim {K_TRIM})")
    print(f"  sign census: {neg}/{n} runs slope < 0 "
          f"(model: negative on {int(np.sum(sm < 0))}/{n})")
    print(f"  dyn slope   : median {np.median(sd):7.1f}  "
          f"IQR [{np.percentile(sd,25):7.1f},{np.percentile(sd,75):7.1f}]")
    print(f"  model slope : median {np.median(sm):7.2f}  "
          f"range {sm.min():.2f}..{sm.max():.2f}")

    # 1/range extrapolation with a bootstrap CI on the intercept
    X = np.column_stack([np.ones(n), 1.0 / rng])
    co, *_ = np.linalg.lstsq(X, sd, rcond=None)
    bs = []
    rs = np.random.default_rng(0)
    for _ in range(4000):
        i = rs.integers(0, n, n)
        c, *_ = np.linalg.lstsq(X[i], sd[i], rcond=None)
        bs.append(c[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  slope = a + b/range fit: a = {co[0]:.1f} "
          f"[{lo:.1f}, {hi:.1f}] 95% CI,  b = {co[1]:.1f}")
    print(f"  model median {np.median(sm):.2f} "
          f"{'INSIDE' if lo <= np.median(sm) <= hi else 'outside'} the CI")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.9))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.795, bottom=0.115,
                        wspace=0.24)

    bins = np.linspace(min(sd.min(), -180), 20, 41)
    a1.hist(np.clip(sd, bins[0], bins[-1]), bins=bins, color='#c0392b',
            alpha=0.8, label=f'dynamic inversion ({n} runs)')
    a1.axvspan(sm.min(), sm.max(), color='#e08214', alpha=0.55,
               label=f'model slope range ({sm.min():.1f}..{sm.max():.1f})')
    a1.axvline(0, color='0.4', lw=0.9)
    a1.axvline(float(np.median(sd)), color='k', lw=1.2, ls='--',
               label=f'median {np.median(sd):.1f}')
    a1.set_xlabel(r'per-run slope of $\Delta M_{GE}$ vs tilt '
                  r'[mN$\cdot$m/deg]', fontsize=10)
    a1.set_ylabel('runs', fontsize=10)
    a1.set_title(f'(a) sign census: {neg}/{n} runs negative, as the model\n'
                 'predicts -- magnitude carries the excursion artefact',
                 fontsize=11)
    a1.legend(fontsize=8.5, loc='upper left')
    a1.grid(alpha=0.22, lw=0.4, axis='y')

    a2.scatter(1.0 / rng, sd, s=16, c='#c0392b', alpha=0.55, lw=0,
               label='runs')
    xg = np.linspace(0, (1.0 / rng).max() * 1.05, 50)
    a2.plot(xg, co[0] + co[1] * xg, 'k-', lw=1.4,
            label=f'fit  $a + b/\\mathrm{{range}}$,  '
                  f'$a = {co[0]:.1f}$ [{lo:.1f}, {hi:.1f}]')
    a2.axhspan(sm.min(), sm.max(), color='#e08214', alpha=0.55,
               label='model slope range')
    a2.axhline(0, color='0.4', lw=0.9)
    a2.set_xlabel(r'1 / excursion range [1/deg]', fontsize=10)
    a2.set_ylabel(r'per-run slope [mN$\cdot$m/deg]', fontsize=10)
    a2.set_title('(b) the artefact scales with 1/range; the intercept at\n'
                 r'$1/\mathrm{range} \to 0$ is the range-free estimate',
                 fontsize=11)
    a2.legend(fontsize=8.5, loc='lower left')
    a2.grid(alpha=0.22, lw=0.4)

    fig.suptitle('Per-run GE slope, aggregated against the interference '
                 f'model ($w={W_SG}$, order 2, trim {K_TRIM}, relative '
                 'attitude)', fontsize=12, y=0.965)
    fig.savefig(out, dpi=150)
    print(f"\n  wrote {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
