#!/usr/bin/env python3
"""Take-off transient metrics over the free-flight campaign, every trial.

WHY MAGNITUDES RATHER THAN AXES.  The delivered offset is a vector, and
in four of the five cases it points nowhere near an axis: (-11.45,
-2.90), (-9.90, -14.29), (3.14, -5.26), (2.40, 6.67), (-10.89, 10.91)
mm, i.e. 14, 55, -59, 70 and 135 degrees.  Reporting peak |phi| beside
peak |theta| therefore splits one physical excursion across two columns
along axes the physics does not prefer, and the split moves with
whatever yaw the vehicle happened to hold at lift-off.  The magnitudes
below are invariant to that:

    tilt      acos(R_33), the angle between the body and world verticals
              -- what tip-over risk actually means, not the larger of
              two Euler peaks
    rate      ||(w_x, w_y)||
    drift     ||p_xy(t) - p_xy(t_lo)||, the horizontal distance travelled
    speed     ||(v_x, v_y)||

Body and world frames differ by a rotation about the vertical, so the
horizontal magnitudes are the same in either -- the reason the velocity
column needs no frame convention.

The per-axis columns are written too, for the appendix: the direction
of the drift is the evidence that the mechanism is the claimed one, and
that is worth keeping even though it does not belong in the summary.

NO INCLUSION CRITERIA.  Every trial is reported.  Conditioning the
percentages on trials where the uncompensated baseline was already bad
selects on the outcome, and percentages taken over a small baseline
exaggerate; both are avoided by aggregating everything and quoting the
absolute change in degrees, deg/s and metres beside the percentage.

Definitions follow section VIII-A: t_lo is the first sample at which the
reconstructed collective thrust exceeds the nominal weight, t_70 the
first at which the vehicle has climbed to 70% of the 0.20 m target, and
every peak is taken over [t_lo, t_70].

Usage
-----
  python analysis/freeflight_metrics.py [--data DIR] [--out DIR]
                                        [--source {odom,pose}]

Writes freeflight_metrics_runs.csv (one row per trial, magnitudes and
per-axis together) and prints the summary table.
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

C_T = 1.3175e-7                     # N/rpm^2, table 9
G = 9.81
Z_REF = 0.20                        # m, commanded target altitude
CLIMB_FRAC = 0.70                   # t_70 is 70% of the way there
MASS = {'01': 3.066, '02': 3.220, '03': 3.220, '04': 3.220, '05': 3.220}
VARIANTS = ('wo_ff', 'ff_pivot_based', 'ff_pivot_free')
LABEL = {'wo_ff': 'none', 'ff_pivot_based': 'pivot-based',
         'ff_pivot_free': 'pivot-free'}
# the load-cell truth of table 10, for the drift-direction check
OFFSET = {'01': (-11.45, -2.90), '02': (-9.90, -14.29),
          '03': (3.14, -5.26), '04': (2.40, 6.67), '05': (-10.89, 10.91)}


def tilt_from_vertical(q):
    """Angle between the body and world vertical axes, from [w,x,y,z].

    R_33 = 1 - 2(x^2 + y^2) is the only entry needed, and it is free of
    yaw -- which is the point: the excursion is the same event however
    the vehicle was facing.
    """
    r33 = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    return np.degrees(np.arccos(np.clip(r33, -1.0, 1.0)))


def metrics(path, case, source):
    d = np.load(path)
    t = d['odom/t']
    W = MASS[case] * G

    # t_lo: the thrust reconstruction of (1) first carrying the weight
    f = C_T * np.sum(d['rpm/rpm'].astype(np.float64) ** 2, axis=1)
    hit = np.flatnonzero(f >= W)
    if not hit.size:
        return None, 'never reaches the nominal weight'
    t_lo = float(d['rpm/t'][hit[0]])

    pos = d[f'{source}/position'].astype(np.float64)
    i_lo = int(np.searchsorted(t, t_lo))
    if i_lo >= len(t) - 1:
        return None, 'lift-off at the end of the record'

    # t_70: 70% of the commanded climb, measured from the ground pose
    climbed = pos[:, 2] - pos[i_lo, 2]
    up = np.flatnonzero((climbed >= CLIMB_FRAC * Z_REF) &
                        (np.arange(len(t)) > i_lo))
    if not up.size:
        return None, f'never climbs to {CLIMB_FRAC * Z_REF:.2f} m'
    i_70 = int(up[0])

    s = slice(i_lo, i_70 + 1)
    q = d[f'{source}/quaternion'].astype(np.float64)[s]
    w = d['odom/angular_vel'].astype(np.float64)[s]
    v = d['odom/linear_vel'].astype(np.float64)[s]
    dxy = pos[s, :2] - pos[i_lo, :2]

    tilt = tilt_from_vertical(q)
    wxy = np.degrees(np.hypot(w[:, 0], w[:, 1]))
    drift = np.hypot(dxy[:, 0], dxy[:, 1])
    speed = np.hypot(v[:, 0], v[:, 1])
    j = int(np.argmax(drift))                   # the farthest excursion

    return dict(
        t_lo=t_lo, t_70=float(t[i_70]), dur=float(t[i_70]) - t_lo,
        n=i_70 - i_lo + 1,
        # magnitudes -- the summary
        tilt=float(tilt.max()),
        rate=float(wxy.max()),
        drift=float(drift.max()),
        speed=float(speed.max()),
        # direction of the farthest excursion, for the mechanism figure
        drift_dir=float(np.degrees(np.arctan2(dxy[j, 1], dxy[j, 0]))),
        # per-axis -- the appendix
        phi=float(np.degrees(np.abs(np.arctan2(
            2 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3]),
            1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)))).max()),
        theta=float(np.degrees(np.abs(np.arcsin(np.clip(
            2 * (q[:, 0] * q[:, 2] - q[:, 3] * q[:, 1]), -1, 1)))).max()),
        wx=float(np.degrees(np.abs(w[:, 0])).max()),
        wy=float(np.degrees(np.abs(w[:, 1])).max()),
        dx=float(np.abs(dxy[:, 0]).max()),
        dy=float(np.abs(dxy[:, 1]).max()),
        vx=float(np.abs(v[:, 0]).max()),
        vy=float(np.abs(v[:, 1]).max()),
    ), None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('DataSet/free_flight'))
    p.add_argument('--out', type=Path, default=Path('.'))
    p.add_argument('--source', choices=('odom', 'pose'), default='odom',
                   help='attitude/position source (odom is the estimate '
                        'the controller acted on)')
    a = p.parse_args()

    rows, skipped = [], []
    for f in sorted(a.data.glob('*/*/*.npz')):
        case, ctrl, fn = f.relative_to(a.data).parts
        var = re.sub(r'(_\d+)?\.npz$', '', fn)
        rep = (re.search(r'_(\d+)\.npz$', fn) or [None, '1'])[1]
        m, why = metrics(f, case, a.source)
        if m is None:
            skipped.append((str(f.relative_to(a.data)), why))
            continue
        rows.append(dict(case=case, controller=ctrl, variant=var,
                         repeat=int(rep), **m))

    if not rows:
        raise SystemExit(f'no usable trials under {a.data}')

    out = a.out / 'freeflight_metrics_runs.csv'
    with open(out, 'w', newline='') as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)

    # ---- take-off duration, reported as in section VIII-A ------------
    dur = np.array([r['dur'] for r in rows])
    print(f'{len(rows)} trials'
          + (f'   ({len(skipped)} skipped)' if skipped else ''))
    for s, why in skipped:
        print(f'   skipped {s}: {why}')
    print(f'take-off duration  median {np.median(dur):.2f} s, '
          f'95th pct {np.percentile(dur, 95):.2f} s\n')

    # ---- the summary: every trial, absolute change beside the % ------
    by = defaultdict(list)
    for r in rows:
        by[(r['case'], r['controller'], r['variant'])].append(r)

    KEYS = [('tilt', 'tilt [deg]', '{:.2f}'),
            ('rate', 'rate [deg/s]', '{:.1f}'),
            ('drift', 'drift [m]', '{:.3f}'),
            ('speed', 'speed [m/s]', '{:.3f}')]

    for key, name, fmt in KEYS:
        print(f'=== {name} — peak over [t_lo, t_70], mean of the repeats ===')
        print(f'{"case":<6}{"ctrl":<7}{"none":>10}'
              + ''.join(f'{LABEL[v]:>12}{"abs":>9}{"%":>7}'
                        for v in VARIANTS[1:]))
        agg = defaultdict(list)
        for case in sorted(MASS):
            for ctrl in ('hgdo', 'l1'):
                base = by.get((case, ctrl, 'wo_ff'))
                if not base:
                    continue
                b = float(np.mean([r[key] for r in base]))
                line = f'{case:<6}{ctrl:<7}{fmt.format(b):>10}'
                for v in VARIANTS[1:]:
                    g = by.get((case, ctrl, v))
                    if not g:
                        line += f'{"-":>12}{"-":>9}{"-":>7}'
                        continue
                    c = float(np.mean([r[key] for r in g]))
                    line += (f'{fmt.format(c):>12}'
                             f'{fmt.format(c - b):>9}'
                             f'{100 * (b - c) / b:>6.0f}%')
                    agg[v].append((b, c))
                print(line)
        for v in VARIANTS[1:]:
            if not agg[v]:
                continue
            arr = np.array(agg[v])
            red = 100 * (arr[:, 0] - arr[:, 1]) / arr[:, 0]
            print(f'  {LABEL[v]:<14} mean {arr[:, 0].mean():.3g} -> '
                  f'{arr[:, 1].mean():.3g}  '
                  f'(absolute {arr[:, 1].mean() - arr[:, 0].mean():+.3g}, '
                  f'{red.mean():+.0f}% mean, {np.sum(red > 0)}/{len(red)} '
                  f'improved)')
        print()

    # ---- does the drift point where the offset does? -----------------
    print('=== drift direction vs the load-cell offset direction ===')
    print(f'{"case":<6}{"offset":>9}{"drift (no comp.)":>20}{"delta":>9}')
    for case in sorted(MASS):
        g = [r for r in rows if r['case'] == case and r['variant'] == 'wo_ff']
        if not g:
            continue
        ox, oy = OFFSET[case]
        off = np.degrees(np.arctan2(oy, ox))
        ang = np.radians([r['drift_dir'] for r in g])
        obs = np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()))
        dlt = (obs - off + 180) % 360 - 180
        print(f'{case:<6}{off:>8.0f}°{obs:>19.0f}°{dlt:>8.0f}°')
    print('\n(the vehicle falls towards the heavy side, so a drift '
          'aligned with\n the offset direction is the signature of the '
          'mechanism)')
    print(f'\nwritten to {out}')


if __name__ == '__main__':
    main()
