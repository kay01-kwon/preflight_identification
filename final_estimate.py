#!/usr/bin/env python3
"""
Mass & CoM Offset Estimator
============================

Usage
-----
python3 final_estimate.py results/sim
python3 final_estimate.py results/sim -p
python3 final_estimate.py results/sim --save-fig --output-dir figures/
python3 final_estimate.py results/sim --no-plot
"""

import argparse
from pathlib import Path
from analysis.critical_value_loader import CriticalValueLoader
from analysis.com_estimator import CoMEstimator, X_P_CG, Y_P_CG


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load critical value CSVs → estimate mass & CoM offset"
    )
    parser.add_argument(
        'csv_dirs', nargs='+',
        help='Directories containing critical_values_*.csv'
    )
    parser.add_argument(
        '--x-pivot', type=float, default=X_P_CG,
        help=f'Pitch pivot half-span [m] (default {X_P_CG})'
    )
    parser.add_argument(
        '--y-pivot', type=float, default=Y_P_CG,
        help=f'Roll pivot half-span [m] (default {Y_P_CG})'
    )
    parser.add_argument(
        '-p', '--print-data', action='store_true',
        help='Print raw critical value data table'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Directory to save figures (default: csv_dirs[0])'
    )
    parser.add_argument(
        '--save-fig', action='store_true',
        help='Save figures as PNG'
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='Skip showing plots'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load CSVs ──
    loader = CriticalValueLoader(args.csv_dirs)
    print(f"Loaded {len(loader.all_records)} critical value records")

    if args.print_data:
        loader.print_summary()

    # ── Estimate ──
    estimator = CoMEstimator(x_p=args.x_pivot, y_p=args.y_pivot)
    result = estimator.estimate(loader)
    CoMEstimator.print_result(result)

    # ── Plot ──
    save_dir = None
    if args.save_fig:
        save_dir = Path(args.output_dir) if args.output_dir else Path(args.csv_dirs[0])

    show = not args.no_plot

    if show or save_dir:
        CoMEstimator.plot_result(result, save_dir=save_dir, show=show)


if __name__ == "__main__":
    main()