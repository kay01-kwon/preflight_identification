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

    print(f'{"id":<9}{"group":<9}{"x":>7}{"y":>7}{"|p|":>7}{"mass":>7}'
          f'{"roll -/+":>16}{"pitch -/+":>16}  verdict')
    for r in rows:
        print(f'{r["id"]:<9}{r["group"]:<9}{r["x_off_mm"]:>7.1f}'
              f'{r["y_off_mm"]:>7.1f}{r["mag_mm"]:>7.1f}{r["mass_kg"]:>7.3f}'
              f'{r["roll_neg"]:>8.2f}{r["roll_pos"]:>8.2f}'
              f'{r["pitch_neg"]:>8.2f}{r["pitch_pos"]:>8.2f}'
              f'  {r["verdict"]}')

    n_ok = sum(r['verdict'] == 'ok' for r in rows)
    print(f'\n{len(rows)} cases, {n_ok} expected to apply, '
          f'{len(rows) - n_ok} expected to fail by design')
    print(f'written to {a.out / "sim_cases.csv"} and sim_cases.yaml')


if __name__ == '__main__':
    main()
