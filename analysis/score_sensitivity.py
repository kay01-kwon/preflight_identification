#!/usr/bin/env python3
"""Sensitivity of the ramp-invariance score around the calibration.

For every hardware dataset (case x axis), the consistency score of
eq. (score) -- the sum over tip directions of the CV of the
constrained-readout critical moments across the seven ramp rates --
is re-evaluated on one-dimensional sweeps around the deployed
constants: C2 scaled by 0.75..1.25 with K fixed, and K scaled by
0.6..1.4 with C2 fixed (the stage-2 search ranges). Each panel
plots the score relative to its value at the calibrated point, one
curve per dataset, so the depth and width of the calibration valley
is read directly.

In addition, a two-dimensional heatmap of the score over the full
(C2, K) search rectangle is drawn for one representative dataset
(default E2/My, the dataset the fit-comparison figure uses), with
the calibrated point marked (exp_score_heatmap.png).

Usage
-----
  PYTHONPATH=<stubs> python analysis/score_sensitivity.py
      [--outdir DIR] [--dpi N] [--heatmap-dataset case_02/My]
      [--no-slices]
"""
import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                        # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
import critical_value_getter_piecewise as cvp          # noqa: E402
from utils.extractor import load_excitation_dataset    # noqa: E402
from analysis.pnls_constants import PNLS_CONSTANTS     # noqa: E402

C2_FACT = np.linspace(0.75, 1.25, 11)
K_FACT = np.linspace(0.60, 1.40, 11)
COLC = {'case_01': '#0072B2', 'case_02': '#E69F00',
        'case_03': '#009E73', 'case_04': '#D55E00',
        'case_05': '#CC79A7'}
LSA = {'Mx': '-', 'My': '--'}


def score(bags, axis, c2, k):
    """Eq. (score): sum over directions of the rate CV."""
    by = {'pos': [], 'neg': []}
    for bag in bags:
        if cvp.commanded_ramp_rate(bag.name) is None:
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                crit, _ = cvp.extract_piecewise(bag, axis, model='cosh',
                                                cosh_c2=c2, ramp_gain=k)
        except Exception:
            continue
        d = 'pos' if bag.name.startswith('pos') else 'neg'
        by[d].append(crit.onset_moment)
    s = 0.0
    for d in ('pos', 'neg'):
        v = np.array(by[d])
        s += float(v.std(ddof=1) / abs(v.mean()))
    return s


def stage1_centre(bags, axis):
    """Stage-1 medians (C2_0, K_0) from the free per-run fits."""
    c2s, ks = [], []
    for bag in bags:
        mdot = cvp.commanded_ramp_rate(bag.name)
        if mdot is None:
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                crit, fit = cvp.extract_piecewise(bag, axis,
                                                  model='cosh',
                                                  cosh_c2=None,
                                                  ramp_gain=None)
            c1, c2, _ = fit['params']
            if abs(mdot) > 1e-6 and np.isfinite(c1) and np.isfinite(c2):
                c2s.append(abs(c2))
                ks.append(abs(c1) / abs(mdot))
        except Exception:
            continue
    return float(np.median(c2s)), float(np.median(ks))


def heatmap(root, ds, outdir, dpi, n=17):
    """Score over the Alg.-3 search rectangle: the grid is laid on
    the stage-1 centre (C2_0, K_0) exactly as Fig. 6, with the
    deployed stage-2 constants (C2*, K*) marked inside it."""
    case, simax = ds.split('/')
    axis = 'x' if simax == 'Mx' else 'y'
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(root / case / simax)
    c20, k0 = stage1_centre(bags, axis)
    c2s, ks = PNLS_CONSTANTS[(case, simax)]
    print(f'stage-1 centre C2_0 {c20:.3f}, K_0 {k0:.4f}; deployed '
          f'C2* {c2s:.3f} (x{c2s / c20:.3f}), K* {ks:.4f} '
          f'(x{ks / k0:.3f})', flush=True)
    fc2 = np.linspace(0.75, 1.25, n)
    fk = np.linspace(0.60, 1.40, n)
    Z = np.array([[score(bags, axis, f2 * c20, f1 * k0) for f1 in fk]
                  for f2 in fc2])
    s0 = score(bags, axis, c2s, ks)
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    im = ax.pcolormesh(fk, fc2, Z / s0, shading='nearest',
                       cmap='viridis')
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r'score / score$^{\ast}$', fontsize=10)
    ax.plot(ks / k0, c2s / c20, '*', ms=14, color='white', mec='k',
            mew=0.8, label=r'calibrated $(C_2^{\ast}, K^{\ast})$')
    ax.set_xlabel(r'$K / K_0$', fontsize=10)
    ax.set_ylabel(r'$C_2 / C_{2,0}$', fontsize=10)
    ax.set_title(f'{case.replace("case_0", "E")}/{simax}', loc='left',
                 fontsize=10)
    ax.legend(fontsize=8.5, loc='upper left', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(outdir / 'exp_score_heatmap.png', dpi=dpi,
                bbox_inches='tight')
    plt.close(fig)
    jmin = np.unravel_index(np.argmin(Z), Z.shape)
    print(f'heatmap: score* {s0:.4f}, sweep min {Z.min():.4f} at '
          f'C2 x{fc2[jmin[0]]:.2f}, K x{fk[jmin[1]]:.2f} of the '
          f'stage-1 centre; max/score* {Z.max() / s0:.2f}',
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=Path, default=Path('docs'))
    ap.add_argument('--dpi', type=int, default=600)
    ap.add_argument('--heatmap-dataset', default='case_02/My')
    ap.add_argument('--no-slices', action='store_true',
                    help='skip the per-dataset 1-D sweep figure')
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parents[1] / 'DataSet' / 'exp'
    heatmap(root, args.heatmap_dataset, args.outdir, args.dpi)
    if args.no_slices:
        return
    curves = {}                    # (case, simax) -> (sc2, sk, s0)
    for d in sorted(root.glob('case_*/M[xy]')):
        axis = 'x' if d.name == 'Mx' else 'y'
        with contextlib.redirect_stdout(io.StringIO()):
            bags = load_excitation_dataset(d)
        c2s, ks = PNLS_CONSTANTS[(d.parent.name, d.name)]
        s0 = score(bags, axis, c2s, ks)
        sc2 = [score(bags, axis, f * c2s, ks) for f in C2_FACT]
        sk = [score(bags, axis, c2s, f * ks) for f in K_FACT]
        curves[(d.parent.name, d.name)] = (np.array(sc2),
                                           np.array(sk), s0)
        print(f'{d.parent.name}/{d.name}: score* {s0:.4f}, '
              f'C2 sweep {min(sc2):.4f}..{max(sc2):.4f}, '
              f'K sweep {min(sk):.4f}..{max(sk):.4f}', flush=True)

    fig, axes = plt.subplots(2, 1, figsize=(6.8, 6.6))
    for ax, fact, idx, xlab in (
            (axes[0], C2_FACT, 0, r'$C_2 / C_2^{\ast}$'),
            (axes[1], K_FACT, 1, r'$K / K^{\ast}$')):
        for (case, simax), tup in curves.items():
            ax.plot(fact, tup[idx] / tup[2], LSA[simax], lw=1.4,
                    color=COLC[case], alpha=0.85,
                    label=(f'{case.replace("case_0", "E")}/{simax}'
                           if True else None))
        ax.axvline(1.0, color='0.35', lw=1.0, ls=':')
        ax.axhline(1.0, color='0.35', lw=1.0, ls=':')
        ax.set_xlabel(xlab, fontsize=10)
        ax.set_ylabel(r'score / score$^{\ast}$', fontsize=10)
        ax.grid(alpha=0.4, lw=0.8, color='0.6')
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=6.8, ncol=2, loc='upper center',
                   framealpha=0.9)
    fig.tight_layout()
    fig.savefig(args.outdir / 'exp_score_sensitivity.png',
                dpi=args.dpi, bbox_inches='tight')
    print(f'written to {args.outdir}/exp_score_sensitivity.png')


if __name__ == '__main__':
    main()
