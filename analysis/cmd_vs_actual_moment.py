#!/usr/bin/env python3
"""Commanded vs achieved excitation moment, at the point that matters.

WHY. The identification reads the critical moment off the *achieved*
moment, reconstructed from the measured rotor speeds. The allocator,
however, commands a moment, and the rotor answers late. Whether that
lag matters depends on where it is evaluated: at the detected onset a
method that read the *commanded* ramp would report the moment the
rotors had not yet delivered. Comparing both signals at the onset the
pipeline itself detects measures that error directly -- the
experimental face of the R0/R3 rotor-dynamics ablation.

WHAT IS COMPUTED, per excitation run:

    M_act  : moment from the measured speeds, exactly the pipeline's
             signal (CriticalValueExtractor._prepare_signals)
    M_cmd  : the same hex allocation over the speed the command asks
             for, cmd * 9800/8191 (HexaCmdData.rpm), on the same clock

    onset  : the GLR change point the pipeline detects on |omega|

and from them, at the onset:

    gap@onset  M_cmd - M_act at the detected onset [N.m], signed
               along the ramp (+ = command ahead of the response)
    gap/M      the same, relative to the critical moment read there
    tau_eff    gap / r, the delay the gap corresponds to on a ramp of
               rate r [ms]

Cross-correlation is deliberately NOT used for the delay: on a ramp
two lines of equal slope correlate best at zero shift whatever their
offset, so the correlator is blind to exactly the quantity of
interest. On a pure ramp the offset IS the delay, gap = r * tau, and
dividing by the known rate recovers it; the consistency check is that
tau_eff comes out rate-independent, which it does.

The window END (max |M_act|) is deliberately not compared: it falls in
the tip-over and motor-cut transient, where the commanded moment has
already collapsed and the difference measures the shutdown, not the
rotor lag the ablation is after.

Usage
-----
  PYTHONPATH=<stubs> python analysis/cmd_vs_actual_moment.py \
      [--data DIR] [--outdir DIR]
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.extractor import RosBagExtractor, BagData   # noqa: E402
from utils import math_tools                           # noqa: E402
from analysis.critical_value_extractor import (        # noqa: E402
    CriticalValueExtractor)

C_T = 1.3175e-7                # N/rpm^2, table 9
ARM = 0.265                    # m

# bag name -> ramp rate [N.m/s]: pos_Mx_03 = 0.3, neg_My_045 = 0.45 ...
_RATE = re.compile(r'_(\d+)$')


def ramp_rate(name):
    d = _RATE.search(name).group(1)
    return float(d) / (10.0 if len(d) <= 2 else 100.0)


def one_run(bag_dir, axis):
    """Pipeline onset plus (M_cmd, M_act) on the pipeline's clock."""
    ext = RosBagExtractor(bag_dir)
    odom = ext.get_odometry('/mavros/local_position/odom')
    rpm = ext.get_hexa_rpm('/uav/actual_rpm')
    cmd = ext.get_hexa_cmd('/uav/cmd_raw')

    bag = BagData(name=bag_dir.name, odom=odom, rpm=rpm, cmd=cmd)
    res = CriticalValueExtractor(C_T=C_T, arm_length=ARM).extract(bag, axis)

    # the commanded moment, through the identical allocation, onto the
    # identical (odom) timeline the pipeline's moment lives on
    axis_idx = 0 if axis == 'x' else 1
    M_cmd_raw = math_tools.rpm_to_moments_vectorized(
        C_T, cmd.rpm, arm_length=ARM)[:, axis_idx]
    M_cmd = np.interp(res.t, cmd.t - odom.t[0], M_cmd_raw)

    i0, _ = CriticalValueExtractor._detect_excitation_window(
        res.moment, res.omega)
    return res, M_cmd, i0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('DataSet/exp'))
    p.add_argument('--outdir', type=Path,
                   default=Path(__file__).resolve().parents[1] / 'docs')
    a = p.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    bags = sorted(d for d in a.data.glob('case_0*/M*/*')
                  if (d / 'metadata.yaml').exists())
    rows, keep = [], {}
    for b in bags:
        axis = 'x' if b.parent.name == 'Mx' else 'y'
        res, M_cmd, i0 = one_run(b, axis)
        r = ramp_rate(b.name)
        j = res.onset_idx
        # signed along the ramp: + = the command leads the response
        sgn = 1.0 if b.name.startswith('pos') else -1.0
        gap = sgn * (M_cmd[j] - res.moment[j])
        rows.append(dict(
            case=b.parents[1].name, axis=b.parent.name, run=b.name,
            rate=r, gap_onset=float(gap),
            gap_rel=float(gap / abs(res.onset_moment)),
            tau_ms=1e3 * gap / r, M_crit=float(res.onset_moment),
        ))
        if b.name in ('pos_Mx_01', 'pos_Mx_120') \
                and b.parents[1].name == 'case_02':
            keep[b.name] = (res, M_cmd, i0)
        print(f'  {b.parents[1].name}/{b.parent.name}/{b.name:<12}'
              f' r={r:.2f}  gap@onset {gap:+.4f} N.m'
              f' ({100*gap/abs(res.onset_moment):+.1f}%)'
              f'  tau {1e3*gap/r:4.0f} ms', flush=True)

    with open(a.outdir / 'cmd_vs_actual_moment.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ── summary per ramp rate ───────────────────────────────────────
    # medians: in a handful of runs the GLR onset lands in the cut-off
    # transient where the command has already collapsed, and the gap
    # there measures the shutdown rather than the rotor. Those runs
    # are flagged below rather than silently absorbed into a mean.
    flag = [r for r in rows if abs(r['gap_rel']) > 0.5]
    by = defaultdict(list)
    for r in rows:
        by[r['rate']].append(r)
    print(f'\n{"rate":>6}{"n":>4}{"gap@onset [N.m]":>17}'
          f'{"gap/M_crit":>12}{"tau_eff [ms]":>14}   (medians)')
    for rate in sorted(by):
        g = by[rate]
        f = lambda k: np.array([x[k] for x in g])       # noqa: E731
        print(f'{rate:>6.2f}{len(g):>4}'
              f'{np.median(f("gap_onset")):>17.4f}'
              f'{100 * np.median(f("gap_rel")):>11.1f}%'
              f'{np.median(f("tau_ms")):>14.0f}')
    allg = np.array([r['gap_rel'] for r in rows])
    tau = np.array([r['tau_ms'] for r in rows])
    print(f'\nall runs: gap/M_crit median {100*np.median(allg):+.2f}%  '
          f'p95 {100*np.percentile(allg, 95):+.2f}%   '
          f'tau_eff median {np.median(tau):.0f} ms')
    if flag:
        print(f'{len(flag)} run(s) with |gap| > 50% of M_crit -- onset '
              f'detected inside the cut-off transient:')
        for r in flag:
            print(f'    {r["case"]}/{r["axis"]}/{r["run"]}'
                  f'  gap {r["gap_onset"]:+.3f} N.m'
                  f'  ({100*r["gap_rel"]:+.0f}%)')

    # ── figure: slowest and fastest ramp of one case ────────────────
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    for ax, name in zip(axes, ('pos_Mx_01', 'pos_Mx_120')):
        res, M_cmd, i0 = keep[name]
        j = res.onset_idx
        pad = int(0.15 * (j - i0)) + 5
        s = slice(max(i0 - pad, 0), min(j + 3 * pad, len(res.t)))
        ax.plot(res.t[s], M_cmd[s], color='#D55E00', lw=1.4,
                label='commanded')
        ax.plot(res.t[s], res.moment[s], color='#0072B2', lw=1.4,
                label='achieved')
        ax.axvline(res.t[j], color='0.3', lw=1.0, ls='--')
        ax.annotate('detected onset', (res.t[j], ax.get_ylim()[0]),
                    xytext=(4, 14), textcoords='offset points',
                    fontsize=8.5, color='0.25', rotation=90, va='bottom')
        g = [r for r in rows
             if r['run'] == name and r['case'] == 'case_02'][0]
        # outside mathtext matplotlib renders '_' literally; escaping
        # it would put the backslash on the page
        ax.set_title(f'case_02 {name}   '
                     f'({g["rate"]:.2f} N$\\cdot$m/s, '
                     f'$\\tau_{{eff}}$ {g["tau_ms"]:.0f} ms)',
                     fontsize=9.5, loc='left')
        ax.set_xlabel('t [s]', fontsize=9.5)
        ax.grid(alpha=0.45, lw=0.8, color='0.6')
        ax.set_axisbelow(True)
    axes[0].set_ylabel('$M_x$ [N$\\cdot$m]', fontsize=9.5)
    axes[0].legend(fontsize=8.5, loc='upper left', framealpha=0.9)
    fig.tight_layout()
    for ext, kw in (('pdf', {}), ('png', dict(dpi=200))):
        fig.savefig(a.outdir / f'fig_cmd_vs_actual.{ext}',
                    bbox_inches='tight', **kw)
    print(f'\nwritten to {a.outdir / "fig_cmd_vs_actual.pdf"}')


if __name__ == '__main__':
    main()
