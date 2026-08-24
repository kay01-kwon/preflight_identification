#!/usr/bin/env python3
"""Generate the simulation sweep, with each case checked before it is run.

WHY THE SWEEP IS WIDENED. The three cases of Table 8 --- (-6, 0),
(0, 10) and (10, -5) mm --- reach 11 mm at most and two of them load a
single axis, so they neither cover the +/-20 mm design box the paper
declares in Table 2 nor exercise a genuinely asymmetric load.

WHERE THE METHOD ACTUALLY ENDS, which is what a robustness sweep should
find rather than avoid. From the critical moments,

    M_x,+ = (W-f) l_r + W p_y        M_x,- = -(W-f) l_l + W p_y
    M_y,+ = (W-f) l_f - W p_x        M_y,- = -(W-f) l_b - W p_x

the negative-direction moment vanishes at p = (1 - f/W) l, which is
33 mm on roll and 42 mm on pitch at the f/W = 0.70 used here. Past that
the corresponding contact carries no load once the excitation
collective is applied, that direction has no critical moment, and the
pairing has nothing to pair. The bound belongs to the excitation and
not to the airframe: with no collective the vehicle is statically
stable out to the landing-gear half-width itself, 110 and 140 mm. The
allocator saturates only later, at 46 and 49 mm, so within this
procedure it is the balance and not the actuation that binds.

The validity region is therefore a rectangle in the offset plane, and
a sweep that walks outward on more than one bearing crosses a
different side of it each time -- which is why the failures below are
placed on both axes rather than on one.

THE DESIGN. Four groups, each answering a different question:
  small      the three published cases, retained so tables 13-14 stay
             valid and the small-offset regime is still covered
  corner     both axes at the limit of the declared box at once -- the
             asymmetric load the review asks for
  diagonal   both components grown together until the roll side of the
             validity rectangle is crossed
  shallow    grown on a shallow bearing so the pitch side is crossed
             instead, so the failure is not an artefact of one axis
  loaded     the same diagonal offset at the loaded mass

Each case is emitted with the critical moments it should produce and a
verdict --- ok, or the reason it is expected to fail --- so a run that
disagrees with its prediction is visible immediately rather than after
the fact.

Usage
-----
  python analysis/sim_case_list.py [--out DIR]
"""
import argparse
import csv
from pathlib import Path

import numpy as np

G = 9.81
M_UNLOADED, M_LOADED = 3.066, 3.220
THRUST_FRAC = 0.70               # collective held during excitation
L_ROLL, L_PITCH = 0.110, 0.140   # pivot half-widths, table 2
M_MAX = dict(roll=2.371, pitch=2.737)   # allocator caps, figs 12-13
Z_COM = 0.261                    # CAD, for the simulator model


def bearing(mag, deg):
    r = np.radians(deg)
    return round(mag * np.cos(r), 1), round(mag * np.sin(r), 1)


CASES = (
    # the published three, retained so tables 13-14 stay valid and the
    # small-offset regime is still covered
    [dict(group='small', x=x, y=y) for x, y in
     ((-6.0, 0.0), (0.0, 10.0), (10.0, -5.0))]
    # both axes at the limit of the declared box, simultaneously
    + [dict(group='corner', x=x, y=y) for x, y in
       ((20.0, 20.0), (20.0, -20.0))]
    # walked outward at 45 deg: both components grow together until the
    # roll side of the validity rectangle is crossed
    + [dict(group='diagonal', x=v, y=v) for v in (25.0, 32.0, 38.0)]
    # walked outward on a shallow bearing, so the pitch side is crossed
    # instead -- the failure is not an artefact of one axis
    + [dict(group='shallow', x=x, y=y) for x, y in
       ((38.0, 14.0), (46.0, 17.0))]
    + [dict(group='loaded', x=25.0, y=25.0, mass=M_LOADED)]
)


def predict(c):
    """Critical moments and the verdict for one case."""
    m = c.get('mass', M_UNLOADED)
    W = m * G
    f = THRUST_FRAC * W
    d = (W - f)
    px, py = c['x'] * 1e-3, c['y'] * 1e-3
    M = dict(roll_pos=d * L_ROLL + W * py,
             roll_neg=-d * L_ROLL + W * py,
             pitch_pos=d * L_PITCH - W * px,
             pitch_neg=-d * L_PITCH - W * px)
    why = []
    for ax in ('roll', 'pitch'):
        # the pair only exists while the two directions keep their signs
        if M[f'{ax}_pos'] <= 0 or M[f'{ax}_neg'] >= 0:
            why.append(f'{ax}: contact unloaded under the excitation '
                       f'collective')
        elif max(abs(M[f'{ax}_pos']),
                 abs(M[f'{ax}_neg'])) > M_MAX[ax]:
            why.append(f'{ax}: excitation exceeds the allocator cap')
    return M, m, ('ok' if not why else '; '.join(why))


def figure(rows, outdir):
    """The sweep in the offset plane, which is where it reads at a glance.

    The identifiable region is a rectangle and not a disc -- the roll
    bound sits on the y component and the pitch bound on the x -- so
    coverage and the two crossings are geometric facts a reader can see,
    where a table asks them to compare columns of numbers.

    The landing-gear footprint, 110 by 140 mm, is left out of the axes
    deliberately: drawn to scale it would be seven times the area of
    everything else and compress the part that matters. It belongs in
    the caption.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    W = M_UNLOADED * G
    f = THRUST_FRAC * W
    ylim = (1 - f / W) * L_ROLL * 1e3
    xlim = (1 - f / W) * L_PITCH * 1e3

    def box(ax, bx, by, **kw):
        ax.plot([-bx, bx, bx, -bx, -bx], [-by, -by, by, by, -by], **kw)

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    box(ax, xlim, ylim, color='#0072B2', lw=1.8, zorder=2,
        label=f'identifiable, ${xlim:.0f}\\times{ylim:.0f}$ mm')
    box(ax, 20, 20, color='#D55E00', lw=1.4, ls='--', zorder=2,
        label='declared design box, $20\\times20$ mm')

    # the loaded case repeats an unloaded offset, so it needs its own
    # marker or the two sit on top of each other with their labels
    ok = [(r['x_off_mm'], r['y_off_mm']) for r in rows
          if r['verdict'] == 'ok' and r['group'] != 'loaded']
    ld = [(r['x_off_mm'], r['y_off_mm']) for r in rows
          if r['group'] == 'loaded']
    no = [(r['x_off_mm'], r['y_off_mm']) for r in rows
          if r['verdict'] != 'ok']
    ax.plot(*zip(*ok), 'o', ms=6.5, mfc='#4DAF4A', mec='0.2', mew=0.9,
            ls='', zorder=4, label='expected to apply')
    ax.plot(*zip(*ld), 's', ms=8.5, mfc='none', mec='#1B7837', mew=1.8,
            ls='', zorder=3, label=f'same offset, loaded ({M_LOADED} kg)')
    ax.plot(*zip(*no), 'X', ms=9.5, mfc='#C1121F', mec='0.2', mew=0.9,
            ls='', zorder=4, label='expected to fail')
    for k, r in enumerate(rows, 1):
        dx, dy = (5, 4) if r['group'] != 'loaded' else (6, -11)
        ax.annotate(f'S{k}', (r['x_off_mm'], r['y_off_mm']),
                    xytext=(dx, dy), textcoords='offset points',
                    fontsize=7.5, color='0.25', zorder=5)

    ax.set_xlim(-58, 58)
    ax.set_ylim(-46, 46)
    ax.set_aspect('equal')
    ax.axhline(0, color='0.8', lw=0.7, zorder=1)
    ax.axvline(0, color='0.8', lw=0.7, zorder=1)
    ax.grid(alpha=0.4, lw=0.7, color='0.65')
    ax.set_axisbelow(True)
    ax.set_xlabel('$x_{\\mathrm{off}}$ [mm]', fontsize=9.5)
    ax.set_ylabel('$y_{\\mathrm{off}}$ [mm]', fontsize=9.5)
    ax.legend(fontsize=7.6, loc='lower center', ncol=2,
              bbox_to_anchor=(0.5, -0.30), frameon=False)
    fig.tight_layout()
    for ext, kw in (('pdf', {}), ('png', dict(dpi=200))):
        fig.savefig(outdir / f'fig_sim_cases.{ext}',
                    bbox_inches='tight', **kw)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path,
                   default=Path(__file__).resolve().parents[1] / 'DataSet')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for k, c in enumerate(CASES, 1):
        M, m, verdict = predict(c)
        rows.append(dict(id=f'sim_{k:02d}', group=c['group'],
                         x_off_mm=c['x'], y_off_mm=c['y'],
                         mag_mm=round(float(np.hypot(c['x'], c['y'])), 1),
                         mass_kg=m, z_com_m=Z_COM,
                         **{k2: round(v, 3) for k2, v in M.items()},
                         verdict=verdict))

    with open(a.out / 'sim_cases.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # a YAML the simulator can read without a parser dependency here
    with open(a.out / 'sim_cases.yaml', 'w') as fh:
        fh.write('# Generated by analysis/sim_case_list.py -- do not edit.\n'
                 '# Offsets in mm from the geometric centre; the critical\n'
                 '# moments are what each case should produce, and the\n'
                 '# verdict is whether the procedure is expected to apply.\n'
                 'cases:\n')
        for r in rows:
            fh.write(f'  - id: {r["id"]}\n'
                     f'    group: {r["group"]}\n'
                     f'    x_off_mm: {r["x_off_mm"]}\n'
                     f'    y_off_mm: {r["y_off_mm"]}\n'
                     f'    mass_kg: {r["mass_kg"]}\n'
                     f'    z_com_m: {r["z_com_m"]}\n'
                     f'    expect:\n'
                     f'      roll:  [{r["roll_neg"]}, {r["roll_pos"]}]\n'
                     f'      pitch: [{r["pitch_neg"]}, {r["pitch_pos"]}]\n'
                     f'      verdict: "{r["verdict"]}"\n')

    # the paper table. Simulation cases are prefixed S so they cannot be
    # confused with the hardware cases 01-05, which the present text
    # numbers the same way.
    REGIME = {'small': 'small offset', 'corner': 'box corner',
              'diagonal': 'diagonal reach', 'shallow': 'shallow reach',
              'loaded': 'loaded'}
    with open(a.out / 'tab_sim_cases.tex', 'w') as fh:
        fh.write('% Generated by analysis/sim_case_list.py.\n'
                 '\\begin{tabular}{c l r r r c}\n'
                 '  \\toprule\n'
                 '  Case & regime & {$x_{\\mathrm{off}}$ (mm)} & '
                 '{$y_{\\mathrm{off}}$ (mm)} & {$|p_{\\mathrm{off}}|$ (mm)}'
                 ' & applies \\\\\n  \\midrule\n')
        prev = None
        for k, r in enumerate(rows, 1):
            if prev and r['group'] != prev:
                fh.write('  \\addlinespace\n')
            prev = r['group']
            ap = 'yes' if r['verdict'] == 'ok' else '\\textbf{no}'
            fh.write(f'  S{k} & {REGIME[r["group"]]} & '
                     f'${r["x_off_mm"]:g}$ & ${r["y_off_mm"]:g}$ & '
                     f'${r["mag_mm"]:.1f}$ & {ap} \\\\\n')
        fh.write('  \\bottomrule\n\\end{tabular}\n')

    print(f'{"id":<9}{"group":<9}{"x":>7}{"y":>7}{"|p|":>7}{"mass":>7}'
          f'{"roll -/+":>16}{"pitch -/+":>16}  verdict')
    for r in rows:
        print(f'{r["id"]:<9}{r["group"]:<9}{r["x_off_mm"]:>7.1f}'
              f'{r["y_off_mm"]:>7.1f}{r["mag_mm"]:>7.1f}{r["mass_kg"]:>7.3f}'
              f'{r["roll_neg"]:>8.2f}{r["roll_pos"]:>8.2f}'
              f'{r["pitch_neg"]:>8.2f}{r["pitch_pos"]:>8.2f}'
              f'  {r["verdict"]}')

    figure(rows, a.out)

    n_ok = sum(r['verdict'] == 'ok' for r in rows)
    print(f'\n{len(rows)} cases, {n_ok} expected to apply, '
          f'{len(rows) - n_ok} expected to fail by design')
    print(f'written to {a.out / "sim_cases.csv"} and sim_cases.yaml')


if __name__ == '__main__':
    main()
