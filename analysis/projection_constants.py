#!/usr/bin/env python3
"""The constants in the pointwise residual bound, measured per ramp rate.

The bound proved in docs/access_pointwise_bound.tex is

    |r_i| <= E_i + D_i + k sigma (1 + sqrt(P_ii)),
    D_i   = sum_j |P_ij| E_j,

with P the orthogonal projector onto span{cosh C2 tau - 1, 1} on the
run's own sample grid.  Everything in it is computable before any data
is taken: P depends only on C2 and the grid, and E only on the geometry
box.  This prints them.

Three things are worth reading off the table.

  trace P = 2 exactly, at every rate and every window length, because a
  projector's trace is the dimension of its range.  That is the reason
  the constants below do not grow with the window: a two-dimensional
  projector cannot move much, however many samples it is given.

  the Lebesgue constant max_i sum_j |P_ij| sits at 1.92 to 2.10 and
  barely moves across a twelvefold change in ramp rate.  Using it gives
  the crude uniform bound (1 + Lambda) sup E.  Carrying the actual shape
  of E instead of its supremum gives 1 + max_i D_i / sup E, which is
  2.07 to 2.15 -- about a third smaller.

  the pointwise form |r| <= |e| + k sigma, without the D term, is FALSE
  near the onset, and the table shows why: E(tau) vanishes there like
  sinh(C2 tau) while D(tau) does not vanish at all, so their ratio
  diverges.  The last two columns give that ratio at the window end and
  at the first sample.

Usage: python analysis/projection_constants.py
"""
import collections
import os
import pickle
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_quality_bound import rho_bar

PHI_BOX = np.deg2rad(10.0)
HERE = os.path.dirname(os.path.abspath(__file__))


def per_run(d):
    """P, E, D and the constants, for one run."""
    tau, c2, k = d['tau'], d['c2'], d['k']
    jp = 1.0 / (k * c2 ** 2)
    rb, _, _ = rho_bar(d['axis'], PHI_BOX, d['dm_win'])
    E = rb * np.sinh(np.clip(c2 * tau, 0, 30)) / (jp * c2)
    A = np.column_stack([np.cosh(np.clip(c2 * tau, 0, 30)) - 1.0,
                         np.ones_like(tau)])
    P = A @ np.linalg.solve(A.T @ A, A.T)
    D = np.abs(P) @ E
    return dict(
        n=len(tau),
        trace=float(np.trace(P)),
        leb=float(np.abs(P).sum(1).max()),          # Lebesgue constant
        lam=float(D.max() / E.max()),               # sharp, carries E's shape
        kap=float(np.sqrt(np.diag(P).max())),
        gam_end=float((E[-1] + D[-1]) / E[-1]),
        gam_first=float((E[1] + D[1]) / E[1]),
        real=float(np.max(np.abs(d['r'])) / E.max()))


def main():
    with open(os.path.join(HERE, '.failing_cache.pkl'), 'rb') as fh:
        rows = pickle.load(fh)
    g = collections.defaultdict(list)
    for d in rows:
        g[d['rate']].append(per_run(d))

    print(f"\n  P = orthogonal projector onto span{{cosh C2 tau - 1, 1}},"
          f" per run\n")
    print(f"  {'Mdot':>6}{'N':>5}{'tr P':>7}{'Lebesgue':>10}"
          f"{'1 + Lambda':>12}{'1 + D/supE':>12}{'kappa':>8}"
          f"{'1 + kappa':>11}{'realised':>10}")
    print(f"  {'N m/s':>6}{'':5}{'':7}{'Lambda':>10}{'crude':>12}"
          f"{'sharp':>12}{'':8}{'':11}{'max|r|/supE':>10}")
    med = lambda v, kk: float(np.median([x[kk] for x in v]))
    for rt in sorted(g):
        v = g[rt]
        print(f"  {rt:6.2f}{int(med(v, 'n')):5d}{med(v, 'trace'):7.3f}"
              f"{med(v, 'leb'):10.3f}{1 + med(v, 'leb'):12.3f}"
              f"{1 + med(v, 'lam'):12.3f}{med(v, 'kap'):8.3f}"
              f"{1 + med(v, 'kap'):11.3f}{med(v, 'real'):10.3f}")

    allv = [x for v in g.values() for x in v]
    lam = 1 + max(x['lam'] for x in allv)
    kap = 1 + max(x['kap'] for x in allv)
    print(f"\n  over all {len(allv)} runs the uniform bound is")
    print(f"    max|r| <= {lam:.2f} sup E + {kap:.2f} k sigma_n")
    print(f"  and the worst realised ratio is"
          f" {max(x['real'] for x in allv):.3f} sup E.")

    print(f"\n  --- why the D term cannot be dropped ---\n")
    print(f"  {'Mdot':>6}{'(E+D)/E at':>14}{'(E+D)/E at':>14}")
    print(f"  {'N m/s':>6}{'window end':>14}{'first sample':>14}")
    for rt in sorted(g):
        v = g[rt]
        print(f"  {rt:6.2f}{med(v, 'gam_end'):14.2f}"
              f"{med(v, 'gam_first'):14.1f}")
    print(f"\n  E vanishes at the onset like sinh(C2 tau) and D does not,")
    print(f"  so any bound of the form |r| <= Gamma |e| + k sigma needs a")
    print(f"  Gamma that diverges there.  The additive form does not.")

    print(f"\n  --- the confidence multiplier ---\n")
    n = int(np.median([x['n'] for x in allv]))
    for a in (0.05, 0.01):
        print(f"  uniform over {n} samples at {100*(1-a):.0f}% confidence:"
              f"  k = {norm.ppf(1 - a / (4.0 * n)):.2f}"
              f"   (pointwise k = {norm.ppf(1 - a / 2):.2f})")
    print(f"\n  Two Gaussian events per sample, n_i and (Pn)_i, hence 4N in")
    print(f"  the union bound.  The figure's 3 sigma is a per-sample")
    print(f"  statement; a uniform one wants about 3.5.")

    misspecified(rows)
    verify(rows, k=3.5)
    return 0


def misspecified(rows, eps=(0.01, 0.02, 0.05, 0.10)):
    """What Lemma 1 costs when the pinned C2 is not the true one.

    The lemma removes the nominal because it lies in V.  It does so only
    if C2 is exact, and C2 is not given -- stage one estimates it by a
    FREE per-run nonlinear fit and takes a per-configuration median, so
    it carries an error.  With C2 = C2*(1 + eps) the nominal leaves V
    and the residual keeps ||(I-P) omega_nom||, which has to be added.

    It is small and it scales almost linearly in eps: about 2% of sup E
    per 1% of exponent error at the fastest ramp, 0.5% at the slowest.
    """
    g = collections.defaultdict(list)
    for d in rows:
        tau, c2, k = d['tau'], d['c2'], d['k']
        jp = 1.0 / (k * c2 ** 2)
        rb, _, _ = rho_bar(d['axis'], PHI_BOX, d['dm_win'])
        E = rb * np.sinh(np.clip(c2 * tau, 0, 30)) / (jp * c2)
        A = np.column_stack([np.cosh(np.clip(c2 * tau, 0, 30)) - 1.0,
                             np.ones_like(tau)])
        P = A @ np.linalg.solve(A.T @ A, A.T)
        out = []
        for e in eps:
            c2s = c2 * (1.0 + e)
            c1s = d['md_full'] / (jp * c2s ** 2)     # C1* = Mdot/(J_P C2*^2)
            f = c1s * (np.cosh(np.clip(c2s * tau, 0, 30)) - 1.0)
            out.append(float(np.max(np.abs(f - P @ f)) / E.max()))
        g[d['rate']].append(out)

    print(f"\n  --- Lemma 1 needs C2 to be exact, and stage one only"
          f" estimates it ---\n")
    print(f"  ||(I-P) omega_nom||_inf as a fraction of sup E,"
          f" for C2 -> C2(1+eps)\n")
    print(f"  {'Mdot':>6}" + ''.join(f"{'eps = ' + str(int(100 * e)) + '%':>12}"
                                     for e in eps))
    for rt in sorted(g):
        a = np.array(g[rt])
        print(f"  {rt:6.2f}" + ''.join(f"{np.median(a[:, i]):12.3f}"
                                       for i in range(len(eps))))
    print(f"\n  So a 2% exponent error costs 1 to 4% of the envelope and a")
    print(f"  5% error costs 3 to 11%.  It enters the bound additively,")
    print(f"  like the pinned amplitude, and is not covered by rho_bar.")


def verify(rows, k=3.5):
    """The bound itself, on every sample of every run.

    The theorem is about the PROJECTION -- the minimiser over
    span{u, 1} with C2 held -- so that is what is tested.  The deployed
    fit pins the amplitude as well and carries the extra |dC1| u(tau)
    term; it is shown alongside to make the size of that term visible.

    The noise scale is measured two ways.  The pre-onset record is the
    obvious choice and it fails, because before the onset the vehicle
    rests on every landing gear and afterwards it pivots on one edge.
    The in-window content above 5 Hz is the honest one: the fitted model
    is smooth on the scale of the window, so it cannot produce anything
    there, and whatever is there is disturbance by construction.
    """
    from failing_runs import split, amplitude_best
    d2 = np.rad2deg
    g = collections.defaultdict(list)
    for d in rows:
        tau, c2, kk = d['tau'], d['c2'], d['k']
        jp = 1.0 / (kk * c2 ** 2)
        rb, _, _ = rho_bar(d['axis'], PHI_BOX, d['dm_win'])
        E = rb * np.sinh(np.clip(c2 * tau, 0, 30)) / (jp * c2)
        A = np.column_stack([np.cosh(np.clip(c2 * tau, 0, 30)) - 1.0,
                             np.ones_like(tau)])
        P = A @ np.linalg.solve(A.T @ A, A.T)
        D = np.abs(P) @ E
        pii = np.sqrt(np.diag(P))
        _, rf = amplitude_best(tau, d['om'], c2)
        _, hi = split(rf, d['dt'])
        s_in = float(np.sqrt(np.mean(hi ** 2)))
        b_pre = E + D + k * d['sig'] * (1.0 + pii)
        b_in = E + D + k * s_in * (1.0 + pii)
        g[d['rate']].append((
            float(np.all(np.abs(rf) <= b_pre)),
            float(np.all(np.abs(rf) <= b_in)),
            float(np.max(np.abs(rf) / b_in)),
            float(np.max(np.abs(d['r']) / b_in)),
            d2(d['sig']), d2(s_in)))

    print(f"\n  --- the bound, every sample of every run, k = {k} ---\n")
    print(f"  {'Mdot':>6}{'sigma_pre':>12}{'sigma_in':>11}{'worst':>9}"
          f"{'deployed':>11}{'sigma_pre':>12}{'sigma_in':>10}")
    print(f"  {'N m/s':>6}{'runs inside':>12}{'inside':>11}{'ratio':>9}"
          f"{'worst':>11}{'deg/s':>12}{'deg/s':>10}")
    t = [0.0, 0.0]
    for rt in sorted(g):
        a = np.array(g[rt])
        t[0] += a[:, 0].sum()
        t[1] += a[:, 1].sum()
        print(f"  {rt:6.2f}{int(a[:, 0].sum()):9d}/{len(a)}"
              f"{int(a[:, 1].sum()):8d}/{len(a)}{a[:, 2].max():9.3f}"
              f"{a[:, 3].max():11.3f}{np.median(a[:, 4]):12.3f}"
              f"{np.median(a[:, 5]):10.3f}")
    print(f"\n  with the pre-onset floor {int(t[0])}/{len(rows)};"
          f" with the in-window disturbance {int(t[1])}/{len(rows)},")
    print(f"  every sample, worst ratio"
          f" {max(x[2] for v in g.values() for x in v):.3f}.")
    print(f"  The deployed fit reaches"
          f" {max(x[3] for v in g.values() for x in v):.3f}, which is the")
    print(f"  |dC1| u(tau) term the theorem says must be added for it.")


if __name__ == '__main__':
    sys.exit(main())
