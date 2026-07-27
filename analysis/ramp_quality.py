#!/usr/bin/env python3
"""
Hardware ramp-execution quality vs the commanded moment ramp rate
=================================================================
Grades every run's executed moment ramp against its commanded rate, using the
SAME window and the SAME least-squares slope that the identification consumes
for the amplitude constraint C₁ = K·Ṁ — so the metric scores exactly the
quantity the closed-form model depends on. No onset model is involved
(signal-level only), so the assessment is fast and free of circularity.

Per run:
  * slope error  ε = (|Ṁ_meas| − Ṁ_cmd)/Ṁ_cmd  [%]   (full excitation window)
  * linearity RMSE — residual of M(t) about its own best-fit line [N·m]

Run-level gates (both onset-free), the same pair applied automatically inside
extract_piecewise_batch:
  * |ε| ≤ 3% — conservative by construction: a 3% Ṁ error perturbs C₁ by 3%,
    ~7× below the ±20% level the sensitivity analysis shows harmless (<0.9 mm);
  * N_full ≥ 38 — realized window sample count; a necessary condition
    independent of where the onset falls (30 pre-onset + 8 post-onset search
    minimum), the hardware counterpart of the a priori screening bound
    N̂_pre = M_crit^min/(Ṁ·T_s) ≥ 30.

Outputs (in --output-dir):
  ramp_quality_runs.csv      per-run metrics
  ramp_quality_by_rate.csv   per-rate aggregate over PASSING runs
  ramp_quality_excluded.csv  gated-out runs
  ramp_quality_vs_rate.png   (with --save-fig) ε and linearity vs rate

Usage
-----
python analysis/ramp_quality.py DataSet/exp                      # all cases/axes
python analysis/ramp_quality.py DataSet/exp --cases case_05      # one case
python analysis/ramp_quality.py DataSet/exp --gate 3 --save-fig
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import contextlib
import csv
import io
from collections import defaultdict
from pathlib import Path

import numpy as np

from utils.extractor import load_excitation_dataset
from critical_value_getter_piecewise import (
    prepare_signals,
    detect_excitation_window,
    commanded_ramp_rate,
)


def assess_dir(data_dir: Path, axis: str) -> list[dict]:
    """Per-run full-window ramp metrics for one case/axis directory."""
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(data_dir)
    rows = []
    for bag in bags:
        cmd = commanded_ramp_rate(bag.name)
        if not cmd:
            continue
        sig = prepare_signals(bag, axis)
        i0, i1 = detect_excitation_window(sig['moment'])
        win = slice(i0, i1 + 1)
        t, M = sig['t'][win], sig['moment'][win]
        if len(t) < 12:
            continue
        a, b = np.polyfit(t, M, 1)
        rows.append(dict(
            dataset=str(data_dir), axis=axis, bag=bag.name,
            cmd_rate=cmd, actual_rate=abs(float(a)),
            slope_error_pct=(abs(float(a)) - cmd) / cmd * 100.0,
            linearity_rmse=float(np.std(M - (a * t + b))),
            n_win=len(t),
        ))
    return rows


def main():
    p = argparse.ArgumentParser(
        description="Ramp-execution quality vs commanded rate, with run gate.")
    p.add_argument('root', help="dataset root (e.g. DataSet/exp) or a single "
                                "case/axis directory (…/case_05/My)")
    p.add_argument('--cases', nargs='+', default=None,
                   help="restrict to these case names (default: all case_*)")
    p.add_argument('--gate', type=float, default=3.0,
                   help="run-level |slope error| gate in %% (default 3)")
    p.add_argument('--nmin', type=int, default=38,
                   help="minimum realized window sample count N_full "
                        "(default 38 = 30 pre + 8 post onset-search minimum)")
    p.add_argument('--output-dir', default=None)
    p.add_argument('--save-fig', action='store_true')
    args = p.parse_args()

    root = Path(args.root)
    if (root / 'metadata.yaml').exists() or any(root.glob('*/metadata.yaml')):
        dirs = [root]      # a single case/axis directory was given
    else:
        dirs = sorted(d for d in root.glob('case_*/M[xy]') if d.is_dir())
        if args.cases:
            dirs = [d for d in dirs if d.parent.name in args.cases]
    if not dirs:
        raise SystemExit(f"no datasets found under {root}")
    out = Path(args.output_dir) if args.output_dir else \
        (root if root.is_dir() else root.parent)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in dirs:
        axis = 'x' if d.name == 'Mx' else 'y'
        rows += assess_dir(d, axis)
        print(f"  assessed {d} ({len(rows)} runs so far)")

    with open(out / 'ramp_quality_runs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def ok(r):
        return abs(r['slope_error_pct']) <= args.gate and r['n_win'] >= args.nmin
    passing = [r for r in rows if ok(r)]
    excluded = [r for r in rows if not ok(r)]

    by = defaultdict(list)
    for r in passing:
        by[r['cmd_rate']].append(r)
    n_tot = defaultdict(int)
    for r in rows:
        n_tot[r['cmd_rate']] += 1

    print(f"\n── ramp quality by commanded rate (gate |ε| ≤ {args.gate:.0f}%, N_full ≥ {args.nmin}) ──")
    hdr = (f"{'rate':>5} {'pass/total':>10} | {'mean|ε|%':>9} {'max|ε|%':>8} "
           f"{'meanε%':>8} | {'linRMSE':>9}")
    print(hdr)
    print("-" * len(hdr))
    agg = []
    for rate in sorted(n_tot):
        g = by.get(rate, [])
        se = np.array([r['slope_error_pct'] for r in g]) if g else np.array([np.nan])
        lr = np.array([r['linearity_rmse'] for r in g]) if g else np.array([np.nan])
        print(f"{rate:5.2f} {len(g):>5}/{n_tot[rate]:<4} | "
              f"{np.nanmean(np.abs(se)):9.2f} {np.nanmax(np.abs(se)):8.2f} "
              f"{np.nanmean(se):+8.2f} | {np.nanmean(lr):9.5f}")
        agg.append([f"{rate:.2f}", len(g), n_tot[rate],
                    f"{np.nanmean(np.abs(se)):.3f}", f"{np.nanmax(np.abs(se)):.3f}",
                    f"{np.nanmean(se):+.3f}", f"{np.nanmean(lr):.5f}"])

    with open(out / 'ramp_quality_by_rate.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cmd_rate_Nm_s', 'n_pass', 'n_total',
                    'mean_abs_slope_error_pct', 'max_abs_slope_error_pct',
                    'mean_slope_error_pct', 'mean_linearity_rmse_Nm'])
        w.writerows(agg)

    with open(out / 'ramp_quality_excluded.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(excluded, key=lambda r: -abs(r['slope_error_pct'])))
    print(f"\n{len(passing)}/{len(rows)} runs pass; "
          f"{len(excluded)} excluded → ramp_quality_excluded.csv")
    print(f"Tables → {out}")

    if args.save_fig:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        rates = np.array([r['cmd_rate'] for r in rows])
        eps = np.array([r['slope_error_pct'] for r in rows])
        lin = np.array([r['linearity_rmse'] for r in rows]) * 1e3
        bad = np.abs(eps) > args.gate
        jit = rates + np.random.default_rng(0).uniform(-0.012, 0.012, len(rates))
        ur = sorted(set(rates))
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
        a1.axhspan(-args.gate, args.gate, color='tab:green', alpha=0.10,
                   label=f'gate ±{args.gate:.0f}%')
        a1.scatter(jit[~bad], eps[~bad], s=22, c='tab:blue', alpha=0.7,
                   label=f'pass ({int((~bad).sum())})')
        a1.scatter(jit[bad], eps[bad], s=40, c='tab:red', marker='x',
                   label=f'gated out ({int(bad.sum())})')
        a1.plot(ur, [np.mean(np.abs(eps[(rates == r) & ~bad])) for r in ur],
                'k.-', lw=1.5, ms=8, label='mean |ε| (after gate)')
        a1.axhline(0, color='gray', lw=0.6)
        a1.set_xlabel(r'commanded ramp rate $\dot M$ [N·m/s]')
        a1.set_ylabel('full-window slope error ε [%]')
        a1.set_title('(a) Ramp-rate tracking error')
        a1.legend(fontsize=8)
        a1.grid(alpha=0.3)
        a2.scatter(jit[~bad], lin[~bad], s=22, c='tab:blue', alpha=0.7)
        a2.scatter(jit[bad], lin[bad], s=40, c='tab:red', marker='x')
        a2.plot(ur, [np.mean(lin[(rates == r) & ~bad]) for r in ur],
                'k.-', lw=1.5, ms=8, label='mean (after gate)')
        a2.set_xlabel(r'commanded ramp rate $\dot M$ [N·m/s]')
        a2.set_ylabel('linearity RMSE [mN·m]')
        a2.set_title('(b) Ramp linearity')
        a2.legend(fontsize=8)
        a2.grid(alpha=0.3)
        fig.suptitle(f'Hardware ramp execution quality ({len(rows)} runs)',
                     y=1.02)
        fig.tight_layout()
        fp = out / 'ramp_quality_vs_rate.png'
        fig.savefig(fp, dpi=300, bbox_inches='tight')
        print(f"Figure → {fp}")


if __name__ == '__main__':
    main()
