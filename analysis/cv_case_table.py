#!/usr/bin/env python3
"""Per-case ramp-rate dispersion (CV) of the detected critical moment.

For every estimator and hardware case E1-E5, the coefficient of
variation std/|mean| of the detected critical moment within each of
the case's four direction groups (2 axes x 2 tip directions, ~7 ramp
rates each), reported as the median over the four groups with the
worst group in parentheses. Emits docs/tab_cv_case.tex.

Usage
-----
  python analysis/cv_case_table.py <scratch>/nls_comparison_runs.csv
                                   [--outdir DIR]
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = [('cosh', 'COSH'), ('cosh_cad', 'COSH-CAD'), ('nls', 'PNLS'),
           ('pelt_normal', 'CPD-N'), ('pelt_rbf', 'CPD-R'),
           ('cusum', 'CUSUM')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', type=Path)
    ap.add_argument('--outdir', type=Path, default=Path('docs'))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for key, _ in METHODS:
            if r.get(f'mcrit_{key}'):
                groups[(r['case'], r['axis'], r['dir'])][key].append(
                    float(r[f'mcrit_{key}']))

    cases = sorted({k[0] for k in groups})
    out = args.outdir / 'tab_cv_case.tex'
    with open(out, 'w') as fh:
        fh.write(
            '% Per-case ramp-rate dispersion (CV, %) of the detected\n'
            '% critical moment: median over the four direction groups\n'
            '% of each case, worst group in parentheses.  Generated\n'
            '% by analysis/cv_case_table.py.\n'
            '\\begin{table}[t]\n'
            '\\caption{Ramp-rate dispersion of the detected critical\n'
            'moment per case: coefficient of variation [\\%] within a\n'
            'direction group (about seven ramp rates each), median\n'
            'over the four groups of the case, worst group in\n'
            'parentheses.}\n'
            '\\label{tab:cv_case}\n'
            '\\centering\\small\n'
            '\\begin{tabular}{@{}l' + 'c' * len(METHODS) + '@{}}\n'
            '\\toprule\n'
            'Case & ' + ' & '.join(lab for _, lab in METHODS)
            + '\\\\\n\\midrule\n')
        allcv = {key: [] for key, _ in METHODS}
        for case in cases:
            e = case.replace('case_0', 'E')
            cells = []
            for key, _ in METHODS:
                cvs = []
                for (c, ax, d), g in groups.items():
                    v = np.array(g[key])
                    if c == case and len(v) > 1:
                        cvs.append(100 * v.std(ddof=1) / abs(v.mean()))
                allcv[key].extend(cvs)
                cells.append(f'${np.median(cvs):.1f}$ '
                             f'$({max(cvs):.1f})$')
            fh.write(f'{e} & ' + ' & '.join(cells) + '\\\\\n')
            print(e, ' | '.join(cells))
        fh.write('\\midrule\nall & ' + ' & '.join(
            f'${np.median(allcv[key]):.1f}$ $({max(allcv[key]):.1f})$'
            for key, _ in METHODS) + '\\\\\n')
        print('all', ' | '.join(
            f'{np.median(allcv[key]):.1f} ({max(allcv[key]):.1f})'
            for key, _ in METHODS))
        fh.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    print(f'written to {out}')


if __name__ == '__main__':
    main()
