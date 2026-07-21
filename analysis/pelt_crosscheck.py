#!/usr/bin/env python3
"""
Change-point benchmark for the onset detection  (review defense)
================================================================
Independent, method-agnostic validation of the proposed model-based onset.

The identification locates the tip-over onset with the closed-form solution
of the unstable dynamics (``cosh`` model). This script benchmarks that onset
against classic change-point / onset detectors:

  * LLR         — variance-change generalized log-likelihood ratio
                  (single change-point; the classic likelihood-ratio test)
  * PELT normal — Binary Segmentation, Gaussian (mean+variance) cost (ruptures)
  * PELT rbf    — Binary Segmentation, RBF-kernel cost (nonparametric)
  * CUSUM       — tabular cumulative-sum sequential detector

Agreement across these independent families demonstrates the critical
moment is robust to the detection methodology. Two systematic differences
are informative and worth reporting:

  * the statistical detectors sit at slightly larger |M| (detection lag —
    the distribution change must accumulate enough post-onset samples), and
  * CUSUM needs an allowance k≈2σ to reject the pre-onset vibration, whereas
    the model-based (cosh) fit needs no such tuning.

Requires the optional package ``ruptures``:  pip install ruptures

Usage
-----
python analysis/pelt_crosscheck.py DataSet/exp/case_05/My
python analysis/pelt_crosscheck.py DataSet/exp/case_05/My --axis y --save-fig
"""

import os
import sys

# Make the repo root importable when run as `python analysis/pelt_crosscheck.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils.extractor import load_excitation_dataset
from critical_value_getter_piecewise import (
    extract_piecewise,
    detect_excitation_window,
    detect_axis,
    bag_name_to_title,
    estimate_pivot_from_mocap,
    compute_mass_and_offset,
)

# Proposed model = 'cosh'. Classic detectors benchmarked against it:
CLASSIC = ['llr', 'pelt_normal', 'pelt_rbf', 'cusum']
ALL_METHODS = ['cosh'] + CLASSIC
_LABEL = {'cosh': 'cosh (proposed)', 'llr': 'LLR', 'pelt_normal': 'PELT normal',
          'pelt_rbf': 'PELT rbf', 'cusum': 'CUSUM'}
_SHORT = {'cosh': 'cosh', 'llr': 'LLR', 'pelt_normal': 'PELTnorm',
          'pelt_rbf': 'PELTrbf', 'cusum': 'CUSUM'}
_COLOR = {'cosh': 'tab:blue', 'llr': 'tab:green', 'pelt_normal': 'tab:orange',
          'pelt_rbf': 'tab:purple', 'cusum': 'tab:brown'}


# ═════════════════════════════════════════════════════════════
#  Classic change-point / onset detectors
# ═════════════════════════════════════════════════════════════

def pelt_onset_index(omega_win, cost, min_size=5):
    """Single change-point via Binary Segmentation with the given cost."""
    try:
        import ruptures as rpt  # optional dependency
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("This benchmark needs 'ruptures'.  pip install ruptures") from exc
    bkps = rpt.Binseg(model=cost, min_size=min_size).fit(
        np.asarray(omega_win).reshape(-1, 1)).predict(n_bkps=1)
    return int(min(bkps[0], len(omega_win) - 1))


def llr_onset_index(omega_win, min_seg=5):
    """
    Variance-change generalized log-likelihood ratio, single change-point:
        S(j) = N·log σ²_tot − j·log σ²_L − (N−j)·log σ²_R,   onset = argmax S.
    """
    x = np.abs(np.asarray(omega_win, dtype=float))
    N = len(x)
    tot = np.var(x)
    if tot < 1e-15 or N < 2 * min_seg:
        return N // 2
    cs = np.cumsum(x)
    cs2 = np.cumsum(x ** 2)
    best, bj = -np.inf, N // 2
    for j in range(min_seg, N - min_seg):
        ml = cs[j - 1] / j
        vl = max(cs2[j - 1] / j - ml ** 2, 1e-15)
        nr = N - j
        mr = (cs[-1] - cs[j - 1]) / nr
        vr = max((cs2[-1] - cs2[j - 1]) / nr - mr ** 2, 1e-15)
        s = N * np.log(tot) - j * np.log(vl) - nr * np.log(vr)
        if s > best:
            best, bj = s, j
    return int(bj)


def cusum_onset_index(omega_win, guess, direction, k=2.0, h=5.0):
    """
    Tabular CUSUM alarm (first threshold crossing) = onset.

    The increment is standardized by the pre-onset noise scale; the allowance
    k (in σ units) rejects the pre-onset vibration — k≈2 ignores excursions up
    to 2σ, which is necessary so the long flat baseline does not false-alarm.
    """
    x = np.asarray(omega_win, dtype=float)
    base = x[:max(guess, 5)]
    mu0 = float(np.median(base))
    sigma = float(np.std(base)) + 1e-9
    z = direction * (x - mu0) / sigma
    S = 0.0
    for t in range(len(x)):
        S = max(0.0, S + z[t] - k)
        if S > h:
            return int(t)
    return int(len(x) - 1)


def classic_onset_index(name, omega_win, guess, direction):
    if name == 'llr':
        return llr_onset_index(omega_win)
    if name.startswith('pelt_'):
        return pelt_onset_index(omega_win, name.split('_', 1)[1])
    if name == 'cusum':
        return cusum_onset_index(omega_win, guess, direction)
    raise ValueError(f"unknown detector '{name}'")


def _window(bag, axis):
    """Signals + excitation window + local onset guess + tip-over direction."""
    base, _ = extract_piecewise(bag, axis, model='piecewise')  # fast; for signals/seed
    i0, i1 = detect_excitation_window(base.moment)
    win = slice(i0, i1 + 1)
    guess = base.onset_idx - i0
    direction = 1.0 if 'pos' in bag.name.lower() else -1.0
    return base, i0, i1, win, guess, direction


# ═════════════════════════════════════════════════════════════
#  Onset-level benchmark
# ═════════════════════════════════════════════════════════════

def crosscheck(bags, axis):
    """cosh onset moment vs every classic detector, per bag."""
    rows, series = [], []
    for bag in bags:
        crit, _ = extract_piecewise(bag, axis, model='cosh')  # proposed
        base, i0, i1, win, guess, direction = _window(bag, axis)
        t_win, omega_win, M_win = base.t[win], base.omega[win], base.moment[win]

        det = {'cosh': (crit.onset_time, crit.onset_moment)}
        for name in CLASSIC:
            j = classic_onset_index(name, omega_win, guess, direction)
            det[name] = (t_win[j], float(M_win[j]))

        row = {'bag': crit.bag_name}
        for name in ALL_METHODS:
            row[f'M_{name}'] = det[name][1]
        rows.append(row)
        series.append(dict(
            name=crit.bag_name, t_win=t_win, omega_win=omega_win, M_win=M_win,
            onsets={_LABEL[n]: (det[n][0], det[n][1], _COLOR[n],
                                '-' if n == 'cosh' else '-.') for n in ALL_METHODS},
        ))
    return rows, series


# ═════════════════════════════════════════════════════════════
#  Downstream: propagate each detector to CoM / moment offset
# ═════════════════════════════════════════════════════════════

def _crit_for_method(bag, axis, name):
    """CriticalValueResult for a bag under the given detector."""
    if name == 'cosh':
        crit, _ = extract_piecewise(bag, axis, model='cosh')
        return crit
    base, i0, i1, win, guess, direction = _window(bag, axis)
    j = i0 + classic_onset_index(name, base.omega[win], guess, direction)
    return replace(base, onset_idx=j, onset_time=float(base.t[j]),
                   onset_thrust=float(base.f_col[j]),
                   onset_moment=float(base.moment[j]),
                   onset_omega=float(base.omega[j]))


def downstream_crosscheck(bags, axis, known_mass=None):
    out = []
    for name in ALL_METHODS:
        crits = [_crit_for_method(b, axis, name) for b in bags]
        pivots = [estimate_pivot_from_mocap(b, c.onset_time, axis)
                  for b, c in zip(bags, crits)]
        e = compute_mass_and_offset(crits, pivots, axis, known_mass=known_mass)
        out.append(dict(
            method=name,
            M_ff=e['pair3_ff_mean'], M_ff_std=e['pair3_ff_std'],
            offset_mm=e['pair3_offset_mean'] * 1e3,
            offset_mm_std=e['pair3_offset_std'] * 1e3,
            W_off=e['pair3_Woffset_mean'], mass=e['pair3_mass_mean'],
        ))
    return out


# ═════════════════════════════════════════════════════════════
#  Output
# ═════════════════════════════════════════════════════════════

def print_and_save_table(rows, axis, output_dir):
    cols = [f'M_{n}' for n in ALL_METHODS]
    hdr = f"{'bag':<14}" + "".join(f"{_SHORT[n]:>12}" for n in ALL_METHODS)
    print(hdr)
    print("-" * len(hdr))
    dev = {n: [] for n in CLASSIC}
    for r in rows:
        print(f"{r['bag']:<14}" + "".join(f"{r[c]:>+12.4f}" for c in cols))
        base = r['M_cosh']
        for n in CLASSIC:
            dev[n].append((abs(r[f'M_{n}']) - abs(base)) / abs(base) * 100 if base else np.nan)
    print("-" * len(hdr))
    print("mean |Δ|% vs cosh:  " + "  ".join(
        f"{_SHORT[n]}={np.nanmean(dev[n]):+.1f}%" for n in CLASSIC))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / f"onset_benchmark_{axis}.csv"
    with open(p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bag'] + cols)
        for r in rows:
            w.writerow([r['bag']] + [f"{r[c]:.6f}" for c in cols])
    print(f"\nOnset table → {p}")
    return p


def print_and_save_downstream(down, axis, output_dir):
    def _spread(vals):
        v = np.array([x for x in vals if not np.isnan(x)])
        return (v.max() - v.min()) / abs(np.mean(v)) * 100 if len(v) else np.nan

    print(f"\n{'method':16s} | {'M_ff[Nm]':>17} | {'CoM off[mm]':>14} | {'mass[kg]':>8}")
    print("-" * 66)
    for d in down:
        print(f"{_LABEL[d['method']]:16s} | {d['M_ff']:+8.4f} ± {d['M_ff_std']:.4f} | "
              f"{d['offset_mm']:+7.2f} ± {d['offset_mm_std']:4.2f} | {d['mass']:8.3f}")
    print("-" * 66)
    print(f"spread across methods:  M_ff {_spread([d['M_ff'] for d in down]):.1f}%  "
          f"CoM {_spread([d['offset_mm'] for d in down]):.1f}%  "
          f"mass {_spread([d['mass'] for d in down]):.1f}%")
    print("(a large spread flags an outlier detector — inspect per-method rows.)")

    p = Path(output_dir) / f"onset_benchmark_downstream_{axis}.csv"
    with open(p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'M_ff_Nm', 'M_ff_std', 'CoM_offset_mm',
                    'CoM_offset_std_mm', 'W_offset_Nm', 'mass_kg'])
        for d in down:
            w.writerow([d['method'], f"{d['M_ff']:.6f}", f"{d['M_ff_std']:.6f}",
                        f"{d['offset_mm']:.4f}", f"{d['offset_mm_std']:.4f}",
                        f"{d['W_off']:.6f}", f"{d['mass']:.4f}"])
    print(f"Downstream table → {p}")
    return p


def plot_crosscheck(series, axis, save_dir=None, show=True):
    n = len(series)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)
    for idx, s in enumerate(series):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        tw, ww, Mw = s['t_win'], s['omega_win'], s['M_win']
        ax.plot(tw, ww, 'k-', lw=0.8, alpha=0.85)
        js = []
        for lab, (t_on, M_on, clr, ls) in s['onsets'].items():
            lw = 2.2 if 'proposed' in lab else 1.5
            ax.axvline(t_on, color=clr, ls=ls, lw=lw, label=f'{lab}: {M_on:+.3f}')
            js.append(t_on)
        ax2 = ax.twinx()
        ax2.plot(tw, Mw, 'tab:gray', lw=1.0, alpha=0.35)
        ax2.set_ylabel(f'$M_{axis}$ [N·m]', color='gray')
        lo, hi = min(js), max(js)
        pad = 0.06 * (tw[-1] - tw[0])
        ax.set_xlim(max(tw[0], lo - pad), min(tw[-1], hi + pad))
        ax.set_title(bag_name_to_title(s['name']), fontsize=12)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(rf'$\omega_{axis}$ [rad/s]')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower left' if 'neg' in s['name'].lower() else 'upper left')
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)
    fig.suptitle('Onset benchmark: cosh (proposed) vs classic change-point '
                 'detectors (LLR / PELT / CUSUM)', fontsize=14)
    fig.tight_layout()
    if save_dir:
        p = Path(save_dir) / f"onset_benchmark_{axis}.png"
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f"Figure → {p}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_downstream(down, axis, save_dir=None, show=True):
    labels = [_LABEL[d['method']].replace(' (proposed)', '\n(proposed)').replace(' ', '\n')
              for d in down]
    x = np.arange(len(down))
    colors = [_COLOR[d['method']] for d in down]
    off_lbl = 'x_{off}' if axis == 'y' else 'y_{off}'
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    ff = [d['M_ff'] for d in down]
    a1.bar(x, ff, yerr=[d['M_ff_std'] for d in down], capsize=4,
           color=colors, alpha=0.85, edgecolor='k')
    a1.axhline(np.mean(ff), color='gray', ls='--', lw=1, label=f'mean={np.mean(ff):+.4f}')
    a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=8)
    a1.set_ylabel(r'Moment offset $M_{ff}=0.5(M_p+M_n)$ [N·m]')
    a1.set_title('Feedforward moment offset (pivot-free)')
    a1.legend(); a1.grid(axis='y', alpha=0.3)
    off = [d['offset_mm'] for d in down]
    a2.bar(x, off, yerr=[d['offset_mm_std'] for d in down], capsize=4,
           color=colors, alpha=0.85, edgecolor='k')
    a2.axhline(np.mean(off), color='gray', ls='--', lw=1, label=f'mean={np.mean(off):+.2f}mm')
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=8)
    a2.set_ylabel(rf'CoM offset ${off_lbl}$ [mm]')
    a2.set_title('CoM offset (pivot-based)')
    a2.legend(); a2.grid(axis='y', alpha=0.3)
    fig.suptitle('Downstream robustness: final CoM / moment offset vs detector', fontsize=13)
    fig.tight_layout()
    if save_dir:
        p = Path(save_dir) / f"onset_benchmark_downstream_{axis}.png"
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f"Downstream figure → {p}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark the cosh onset against classic change-point detectors.")
    p.add_argument('data_dir', type=str)
    p.add_argument('--axis', type=str, default=None, choices=['x', 'y'])
    p.add_argument('--mass', type=float, default=None,
                   help='Known mass [kg] for the downstream estimation.')
    p.add_argument('--output-dir', type=str, default=None)
    p.add_argument('--no-plot', action='store_true')
    p.add_argument('--save-fig', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir

    bags = load_excitation_dataset(dataset_dir)
    print(f"Loaded {len(bags)} bags: {[b.name for b in bags]}\n")
    axis = args.axis if args.axis else detect_axis(dataset_dir, bags)
    print(f"Axis        : {axis} ({'roll' if axis == 'x' else 'pitch'})")
    print(f"Benchmark   : cosh (proposed) vs {CLASSIC}\n")

    print("── Onset moment (M_crit) ──")
    rows, series = crosscheck(bags, axis)
    print_and_save_table(rows, axis, output_dir)

    print("\n── Downstream (CoM / moment offset) ──")
    down = downstream_crosscheck(bags, axis, known_mass=args.mass)
    print_and_save_downstream(down, axis, output_dir)

    save_dir = output_dir if args.save_fig else None
    show = not args.no_plot
    if show or save_dir:
        plot_crosscheck(series, axis, save_dir=save_dir, show=show)
        plot_downstream(down, axis, save_dir=save_dir, show=show)


if __name__ == "__main__":
    main()
