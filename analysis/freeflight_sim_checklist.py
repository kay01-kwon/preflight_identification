#!/usr/bin/env python3
"""Tickable run checklist for the simulated free-flight validation.

The excitation campaign identified the offset of every feasible case;
the free-flight campaign is the closing of the loop -- take off with
and without the feedforward built from that IDENTIFIED offset (not the
injected truth: using the truth would validate the simulator, not the
procedure). Only the cases the identification succeeded on fly: S10
and S12 sit outside the identifiable rectangle and tip over under the
excitation collective, so there is no estimate to compensate with.

Per case, the hardware protocol is mirrored: three take-offs without
compensation, three with, same logging as the excitation runs. The
identified offsets entered as feedforward are frozen in this file so
the checklist and the controller configuration cannot drift apart.

Outputs freeflight_sim_checklist.{md,tex}; compile the tex twice for
the PDF.

Usage
-----
  python analysis/freeflight_sim_checklist.py [--out DIR]
"""
import argparse
from pathlib import Path

G = 9.81

# id -> (x_true, y_true, x_hat, y_hat [mm], mass [kg]); the hats are the
# identified offsets of the R3 excitation campaign and are the numbers
# the feedforward must be built from
CASES = {
    'S1':  (-6.0,   0.0,  -6.01,  -0.11, 3.066),
    'S2':  ( 0.0,  10.0,  -0.01,  10.00, 3.066),
    'S3':  (10.0,  -5.0,  10.02,  -5.05, 3.066),
    'S4':  (20.0,  20.0,  20.19,  20.19, 3.066),
    'S5':  (20.0, -20.0,  20.22, -20.16, 3.066),
    'S6':  (-20.0, 20.0, -20.26,  20.09, 3.066),
    'S7':  (-20.0,-20.0, -20.21, -20.24, 3.066),
    'S8':  (25.0,  25.0,  25.22,  25.20, 3.066),
    'S9':  (32.0,  32.0,  32.37,  32.09, 3.066),
    'S11': (38.0,  14.0,  38.13,  14.22, 3.066),
    'S13': (25.0,  25.0,  25.35,  25.23, 3.220),
}
VARIANTS = ('wo_ff', 'w_ff')
N_TRIAL = 3


def m_ff(case):
    """Feedforward moments from the identified offset: the moment the
    offset exerts, to be cancelled by the allocator."""
    _, _, xh, yh, m = CASES[case]
    W = m * G
    return W * yh * 1e-3, W * xh * 1e-3          # (about x from y, about y from x)


def emit_md(out):
    L = ['# Simulated free-flight validation -- run checklist', '']
    L += ['Only the identifiable cases fly (S10, S12 tip under the '
          'excitation collective and are excluded by construction).',
          '',
          'Protocol per trial: set the case offset and mass in the model, '
          'configure the feedforward from the IDENTIFIED offset below '
          '(w_ff runs only), arm, take off to the standard hover point, '
          'hold at least 5 s, stop the log. Same topics as the excitation '
          'runs. Bag name: `<variant>_<trial>` under `S<k>/`.',
          '']
    n_tot = 0
    for c, (xt, yt, xh, yh, m) in CASES.items():
        mfx, mfy = m_ff(c)
        L += [f'### {c} --- truth ({xt:+.0f}, {yt:+.0f}) mm, '
              f'mass {m:.3f} kg']
        L += [f'    feedforward from identified ({xh:+.2f}, {yh:+.2f}) mm '
              f'->  W*y = {mfx:+.3f} N.m, W*x = {mfy:+.3f} N.m', '']
        for v in VARIANTS:
            row = '  '.join(f'[ ] {v}_{k+1}' for k in range(N_TRIAL))
            L += [f'- {row}']
            n_tot += N_TRIAL
        L += ['']
    L += [f'**{n_tot} flights** = {len(CASES)} cases x {len(VARIANTS)} '
          f'variants x {N_TRIAL} trials.', '',
          'After the campaign, pack with:', '',
          '    python analysis/freeflight_pack.py <bag_root> '
          'SimDataSet/free_flight', '']
    (out / 'freeflight_sim_checklist.md').write_text('\n'.join(L))


def emit_tex(out):
    T = [r'\documentclass[10pt]{article}',
         r'\usepackage[margin=2.2cm]{geometry}',
         r'\usepackage{amssymb,array,booktabs}',
         r'\newcommand{\cb}{$\square$}',
         r'\setlength{\parindent}{0pt}',
         r'\begin{document}',
         r'\begin{center}\Large\bfseries '
         r'Simulated free-flight validation --- run checklist'
         r'\end{center}',
         r'\medskip',
         r'Only the identifiable cases fly; S10 and S12 tip under the '
         r'excitation collective and are excluded by construction. '
         r'Per trial: set the case offset and mass, configure the '
         r'feedforward from the \emph{identified} offset (w\_ff only), '
         r'arm, take off to the standard hover point, hold $\geq 5$\,s, '
         r'stop the log. Same topics as the excitation runs; bag name '
         r'\texttt{<variant>\_<trial>} under \texttt{S<k>/}.',
         r'\medskip', '',
         r'\renewcommand{\arraystretch}{1.45}',
         r'\begin{tabular}{l l r r c c}',
         r'\toprule',
         r'case & truth [mm] & identified [mm] & $m$ [kg] & '
         r'wo\_ff $\times$3 & w\_ff $\times$3 \\',
         r'\midrule']
    for c, (xt, yt, xh, yh, m) in CASES.items():
        cbs = r'\cb\ \cb\ \cb'
        T += [f'{c} & $({xt:+.0f},\\,{yt:+.0f})$ & '
              f'$({xh:+.2f},\\,{yh:+.2f})$ & {m:.3f} & {cbs} & {cbs} \\\\']
    T += [r'\bottomrule', r'\end{tabular}', r'\medskip', '',
          f'{len(CASES) * len(VARIANTS) * N_TRIAL} flights = '
          f'{len(CASES)} cases $\\times$ {len(VARIANTS)} variants '
          f'$\\times$ {N_TRIAL} trials. The identified offsets are the '
          r'R3 excitation-campaign estimates, frozen in '
          r'\texttt{analysis/freeflight\_sim\_checklist.py}; the '
          r'feedforward moments follow as $W\hat{y}$ and $W\hat{x}$.',
          r'\end{document}']
    (out / 'freeflight_sim_checklist.tex').write_text('\n'.join(T))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path,
                   default=Path(__file__).resolve().parents[1] / 'DataSet')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    emit_md(a.out)
    emit_tex(a.out)
    print(f'written to {a.out}/freeflight_sim_checklist.{{md,tex}}')


if __name__ == '__main__':
    main()
