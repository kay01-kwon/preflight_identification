#!/usr/bin/env python3
"""External in-flight identification baselines: RLS, EKF and UKF.

WHY. The comparison so far is internal -- onset detectors against each
other, two disturbance observers against each other. The online
identification literature this paper cites estimates the same offset
from flight data, so the fair question is what those estimators deliver
on this vehicle, and the answer has to be measured rather than argued.

THE MODEL, shared by all three. Rotor i sits at (l_x,i, l_y,i, 0) from
the geometric centre and produces T_i along body z; the centre of mass
sits at (x_off, y_off, z_c). Taking moments about the centre of mass,

    J_xx wdot_x + (J_zz - J_yy) w_y w_z = Mgeom_x - y_off f
    J_yy wdot_y + (J_xx - J_zz) w_x w_z = Mgeom_y + x_off f

with Mgeom_x = sum l_y,i T_i and Mgeom_y = -sum l_x,i T_i. Each axis is
therefore governed by its own inertia and its own offset component,
driven by signals the log already contains.

WHAT EXCITES WHAT. In steady hover wdot and w vanish and the balance
degenerates to Mgeom = p_off f -- the inertia is unidentifiable and the
offset is all that remains. During the uncompensated take-off transient
the vehicle does accelerate, and the inertia becomes identifiable. Both
regimes are present in every trial, which is why a filter run over the
whole record can separate them.

THE THREE ESTIMATORS, and where each comes from.
  RLS  faithful to the cited formulation: the quasi-static
       thrust-moment coupling Mgeom = -/+ p_off f, solved recursively
       with a forgetting factor. No rotational-dynamics term and no
       rate differentiation; the forgetting factor discounts the
       take-off transient, where the quasi-static assumption is
       violated, in favour of the hover that follows.
  EKF  after Wuest, Kumar and Loianno (2019), who carry the geometric
       and inertia parameters as filter states. Here the state is
       [w, a, p] with a = 1/J, propagated through
       wdot = a (Mgeom -/+ p f + gyro) and corrected by the measured w.
       The parameters enter as a product, which is what makes the
       problem nonlinear and the filter necessary.
  UKF  the same state and model as the EKF, propagated by the unscented
       transform rather than a Jacobian.

Neither filter differentiates the rate; that is the property the
filter-based works are chosen for, and it is preserved here.

Only the uncompensated trials are used: with the feedforward moment
applied the vehicle no longer flies with the offset uncorrected.

Usage
-----
  PYTHONPATH=<stubs> python analysis/inflight_baselines.py [outdir]
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

C_T = 1.3175e-7                  # N/rpm^2, table 9
ARM = 0.265                      # m,       table 9
G = 9.81
MASS = {'01': 3.066, '02': 3.220, '03': 3.220, '04': 3.220, '05': 3.220}
TRUTH = {'01': (-11.45, -2.90), '02': (-9.90, -14.29), '03': (3.14, -5.26),
         '04': (2.40, 6.67), '05': (-10.89, 10.91)}
# CAD inertias of table 7, used only for the small gyroscopic correction
J_CAD = dict(xx=0.051085, yy=0.050564, zz=0.073831)
FS = 100.0                       # Hz, the logged rate

DATA = Path('DataSet/free_flight')
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
OUT.mkdir(parents=True, exist_ok=True)

i = np.arange(6)                 # index base checked against the truth
LX = ARM * np.cos(np.pi / 6 + i * np.pi / 3)
LY = ARM * np.sin(np.pi / 6 + i * np.pi / 3)


# ── signals ──────────────────────────────────────────────────────────
def signals(path, case):
    """Resampled (w, Mgeom, f, gyro) on the rotor-speed clock."""
    d = np.load(path)
    t, tr = d['odom/t'], d['rpm/t']
    W = MASS[case] * G
    T = C_T * d['rpm/rpm'].astype(np.float64) ** 2
    f = T.sum(axis=1)
    hit = np.flatnonzero(f >= W)
    if not hit.size:
        return None
    t_lo = float(tr[hit[0]])
    # from just before lift-off to the end of powered flight, so the
    # transient and the hover are both in the record
    keep = (tr >= t_lo - 0.5) & (f > 0.5 * W)
    if keep.sum() < 300:
        return None
    tr, T, f = tr[keep], T[keep], f[keep]

    w = np.column_stack([np.interp(tr, t, d['odom/angular_vel'][:, k])
                         for k in range(3)]).astype(np.float64)
    Mg = np.column_stack([(T * LY).sum(axis=1), -(T * LX).sum(axis=1)])
    gyro = np.column_stack([                       # -(J_zz-J_yy) w_y w_z
        -(J_CAD['zz'] - J_CAD['yy']) * w[:, 1] * w[:, 2],
        -(J_CAD['xx'] - J_CAD['zz']) * w[:, 0] * w[:, 2]])
    return tr, w, Mg, f, gyro


# ── RLS ──────────────────────────────────────────────────────────────
def rls(w, Mg, f, gyro, lam=0.999):
    """Faithful to the cited formulation: the quasi-static
    thrust--moment coupling Mgeom = -/+ p_off f, solved recursively
    with a forgetting factor.  No rotational-dynamics term and no
    rate differentiation -- in hover the moment balance degenerates
    to the coupling alone, which is exactly what the recursion fits;
    the forgetting factor discounts the take-off transient where the
    quasi-static assumption is violated."""
    out = []
    for ax in (0, 1):                       # roll: +y_off, pitch: -x_off
        th, P = 0.0, 1e-2
        for k in range(len(f)):
            phi = f[k]
            g = P * phi / (lam + phi * P * phi)
            th = th + g * (Mg[k, ax] - phi * th)
            P = (P - g * phi * P) / lam
        out.append(th)
    return out[0], -out[1]                  # (y_off, x_off) in metres


# ── shared nonlinear model for the filters ───────────────────────────
def _step(x, u, dt, sgn):
    """x = [w, a, p], a = 1/J; u = (Mgeom, f, gyro)."""
    Mg, f, gy = u
    w, a, p = x[0], x[1], x[2]
    wdot = a * (Mg + gy + sgn * p * f)
    return np.array([w + wdot * dt, a, p])


def ekf(w, Mg, f, gyro, q=(1e-4, 1e-9, 1e-9), r=4e-4):
    dt = 1.0 / FS
    out = []
    for ax, sgn in ((0, -1.0), (1, +1.0)):
        J0 = J_CAD['xx' if ax == 0 else 'yy']
        x = np.array([w[0, ax], 1.0 / J0, 0.0])
        P = np.diag([1e-3, (0.3 / J0) ** 2, 1e-4])
        Q, H = np.diag(q), np.array([1.0, 0.0, 0.0])
        for k in range(len(f)):
            u = (Mg[k, ax], f[k], gyro[k, ax])
            a, p = x[1], x[2]
            F = np.array([[1.0, (u[0] + u[2] + sgn * p * u[1]) * dt,
                           a * sgn * u[1] * dt],
                          [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            x = _step(x, u, dt, sgn)
            P = F @ P @ F.T + Q
            S = P[0, 0] + r
            K = P @ H / S
            x = x + K * (w[k, ax] - x[0])
            P = (np.eye(3) - np.outer(K, H)) @ P
        out.append(x[2])          # p is y_off on roll, x_off on pitch
    return out[0], out[1]


def ukf(w, Mg, f, gyro, q=(1e-4, 1e-9, 1e-9), r=4e-4, alpha=1e-3, beta=2.0):
    dt, n = 1.0 / FS, 3
    lam = alpha ** 2 * n - n
    Wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
    Wc = Wm.copy()
    Wm[0] = lam / (n + lam)
    Wc[0] = Wm[0] + (1 - alpha ** 2 + beta)
    out = []
    for ax, sgn in ((0, -1.0), (1, +1.0)):
        J0 = J_CAD['xx' if ax == 0 else 'yy']
        x = np.array([w[0, ax], 1.0 / J0, 0.0])
        P = np.diag([1e-3, (0.3 / J0) ** 2, 1e-4])
        Q = np.diag(q)
        for k in range(len(f)):
            u = (Mg[k, ax], f[k], gyro[k, ax])
            try:
                S = np.linalg.cholesky((n + lam) * P)
            except np.linalg.LinAlgError:
                S = np.linalg.cholesky((n + lam) * P + 1e-12 * np.eye(n))
            sig = np.vstack([x, x + S.T, x - S.T])
            sig = np.array([_step(s, u, dt, sgn) for s in sig])
            x = Wm @ sig
            d = sig - x
            P = (d.T * Wc) @ d + Q
            z = sig[:, 0]
            zbar = Wm @ z
            Pzz = Wc @ (z - zbar) ** 2 + r
            Pxz = (d.T * Wc) @ (z - zbar)
            K = Pxz / Pzz
            x = x + K * (w[k, ax] - zbar)
            P = P - np.outer(K, K) * Pzz
        out.append(x[2])          # p is y_off on roll, x_off on pitch
    return out[0], out[1]


# ── sweep ────────────────────────────────────────────────────────────
def main():
  rows = []
  for p in sorted(DATA.glob('*/*/wo_ff*.npz')):
      case, ctrl, fn = p.relative_to(DATA).parts
      s = signals(p, case)
      if s is None:
          print(f'  skipped {p.relative_to(DATA)}')
          continue
      _, w, Mg, f, gy = s
      for name, fn_est in (('RLS', rls), ('EKF', ekf), ('UKF', ukf)):
          try:
              y_off, x_off = fn_est(w, Mg, f, gy)
          except Exception as exc:
              print(f'  {name} failed on {p.relative_to(DATA)}: {exc}')
              continue
          rows.append(dict(method=name, case=case, controller=ctrl,
                           run=fn[:-4], x_mm=1e3 * x_off, y_mm=1e3 * y_off,
                           x_truth=TRUTH[case][0], y_truth=TRUTH[case][1]))
      print(f'  done {p.relative_to(DATA)}', flush=True)

  with open(OUT / 'inflight_baselines.csv', 'w', newline='') as fh:
      wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
      wtr.writeheader()
      wtr.writerows(rows)

  by = defaultdict(lambda: defaultdict(list))
  for r in rows:
      by[r['method']][r['case']].append(r)

  print(f'\n{"method":<8}{"RMS [mm]":>10}{"median":>9}{"max":>8}'
        f'{"corr":>8}{"slope":>8}')
  for m in ('RLS', 'EKF', 'UKF'):
      e, tr_, es = [], [], []
      for case, g in by[m].items():
          x = np.mean([r['x_mm'] for r in g])
          y = np.mean([r['y_mm'] for r in g])
          tx, ty = TRUTH[case]
          e += [x - tx, y - ty]
          tr_ += [tx, ty]
          es += [x, y]
      e = np.abs(e)
      sl = np.polyfit(tr_, es, 1)[0]
      print(f'{m:<8}{np.sqrt(np.mean(e ** 2)):>10.2f}{np.median(e):>9.2f}'
            f'{e.max():>8.2f}{np.corrcoef(tr_, es)[0, 1]:>8.2f}{sl:>8.2f}')
  print(f"\nwritten to {OUT / 'inflight_baselines.csv'}")


if __name__ == '__main__':
    main()
