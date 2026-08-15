#!/usr/bin/env python3
"""Is band membership a theorem, or was it arranged?

(D.13g) derives that the fitted cosh lies within omega_nom +/- E(tau).
It is a corollary of (D.13c), not an independent result: it takes
|delta| <= rho_bar/Mdot and multiplies by C1 C2 sinh, and E was itself
built from rho_bar with Wz = J_P C2^2 arranged to cancel.  It could not
have come out any other way.

The general question is different and does have an answer.  Let

    y = f + e,     f in M,     g_hat = argmin_{g in M} ||y - g||,

with M ANY set of candidate curves and e ANY perturbation.  Then

    ||g_hat - f|| <= ||g_hat - y|| + ||y - f||           triangle
                  <= ||f      - y|| + ||y - f||           optimality, f in M
                  =  2 ||e||.                                        (G1)

Two lines.  No linearisation, no convexity, no identifiability, no
structure on M whatsoever.  The only inputs are that the nominal is IN
the class and that the minimum is global.  If the class misses f by
d_M = dist(f, M), the same argument gives

    ||g_hat - f|| <= 2 ||e|| + d_M.                                  (G2)

The factor 2 is sharp for a general M, and it is the price of assuming
nothing.  Two refinements are checked here against it:

  affine     if M is a subspace through f, g_hat - f = P_V e exactly, so
             the constant is 1, not 2 -- orthogonal projection onto a
             subspace is non-expansive.                              (G3)
  reach      for a curved M of reach r, Federer's projection theorem
             gives constant 1/(1 - ||e||/r), which interpolates: 1 as
             ||e||/r -> 0, and 2 at ||e|| = r/2.                     (G4)

And a third route that is not a norm bound at all:

  direction  the projection (D.13b) uses <e, chi>, not ||e||, so it sees
             WHERE e points and not only how big it is.

The last gives the same constant as `affine` here, because E and chi
are parallel -- which is why (D.13c) and (D.13g) agree, and is the
honest statement of what the arrangement bought.

Usage: python analysis/general_band.py
"""
import sys

import numpy as np

C2, J_P = 5.046, 0.4260
WZ = J_P * C2 ** 2
RHO_BAR = 0.01204                      # N.m, the box value for roll
X = 3.0                                # C2 tau_end, the operating window
N_TAU, N_D = 3001, 1201


class L2:
    """L2 on a uniform grid, as a weighted Euclidean space.

    Everything below is an inner product, so carrying the trapezoid
    weights once as sqrt(w) turns every integral into a dot product and
    every sweep over the class into one matrix multiply.
    """

    def __init__(self, tau):
        w = np.gradient(tau)
        w[0] *= 0.5
        w[-1] *= 0.5
        self.tau, self.s = tau, np.sqrt(w)
        self.T = float(w.sum())
        self.w = w

    def n(self, v):
        return np.sqrt(np.sum((v * self.s) ** 2, axis=-1))

    def ip(self, a, b):
        return float(np.sum(a * b * self.w))

    def q(self, v):
        """Remove the constant component -- the baseline c is fitted."""
        return v - (v * self.w).sum(axis=-1, keepdims=True) / self.T


def build(mdot, c2=C2, j_p=J_P):
    tau = np.linspace(0.0, X / c2, N_TAU)
    L = L2(tau)
    c1 = mdot / (j_p * c2 ** 2)
    f = c1 * (np.cosh(c2 * tau) - 1.0)
    E = RHO_BAR * np.sinh(c2 * tau) / (j_p * c2)
    return L, c1, f, E


def family(L, c1, dmax, c2=C2, n_d=N_D):
    """The whole class as a matrix: one row per onset, and its quotient."""
    ds = np.linspace(-dmax, dmax, n_d)
    G = c1 * (np.cosh(c2 * (L.tau[None, :] - ds[:, None])) - 1.0)
    return ds, G, L.q(G)


def fit_batch(L, Y, ds, G, qG):
    """Global minimum over the class for every row of Y at once.

    c enters linearly and is eliminated by q() inside the distance; d is
    swept exhaustively rather than descended, so no local minimum can be
    returned.  The curve handed back is the FULL fitted curve, baseline
    included, so that it can be compared with the nominal directly.
    """
    qY = L.q(Y)
    d2 = (np.sum((qY * L.s) ** 2, axis=1)[:, None]
          - 2.0 * (qY * L.w) @ qG.T
          + np.sum((qG * L.s) ** 2, axis=1)[None, :])
    j = np.argmin(d2, axis=1)
    mu = lambda V: (V * L.w).sum(axis=-1, keepdims=True) / L.T
    return ds[j], G[j] + (mu(Y) - mu(G[j]))


def adversarial(L, c1, f, E, ds, G, qG, rng, n_try=600):
    """Search |e| <= E for the e that displaces the fit the most.

    e = +/-E are extremal for the affine case, E being parallel to chi.
    The random sign-switching profiles are the ones that could exploit
    the curvature of the class if anything could.
    """
    cands = [E, -E]
    for _ in range(n_try):
        k = int(rng.integers(1, 7))
        s = np.ones_like(L.tau)
        for ed in np.sort(rng.uniform(L.tau[0], L.tau[-1], k)):
            s[L.tau > ed] *= -1.0
        cands.append(E * s * rng.uniform(0.5, 1.0))
    Ecand = np.array(cands)
    _, gfit = fit_batch(L, f[None, :] + Ecand, ds, G, qG)
    dev = L.n(gfit - f[None, :])
    en = L.n(Ecand)
    r = dev / np.maximum(en, 1e-300)
    return float(r.max()), float(r[0])


def reach(L, qG, r_loc):
    """Reach = min(1/max curvature, half the shortest bottleneck).

    Non-local pairs only: nearby points on any curve are close, and that
    is curvature, already counted by r_loc.
    """
    g = (qG * L.w) @ qG.T
    dd = np.sqrt(np.maximum(np.diag(g)[:, None] - 2 * g
                            + np.diag(g)[None, :], 0.0))
    arc = np.concatenate([[0.0], np.cumsum(np.diag(dd, 1))])
    far = np.abs(arc[:, None] - arc[None, :]) > 2.0 * r_loc
    return min(r_loc, 0.5 * dd[far].min()) if far.any() else r_loc


def curvature(L, c1, dmax, c2=C2, n=201):
    """1 / max curvature of the class, as a curve in L2 modulo constants."""
    k = []
    for d in np.linspace(-dmax, dmax, n):
        g1 = L.q(-c1 * c2 * np.sinh(c2 * (L.tau - d)))
        g2 = L.q(c1 * c2 ** 2 * np.cosh(c2 * (L.tau - d)))
        a, b, ip = L.n(g1), L.n(g2), L.ip(g1, g2)
        k.append(np.sqrt(max(a * a * b * b - ip * ip, 0.0)) / a ** 3)
    return 1.0 / max(k)


def main():
    rng = np.random.default_rng(20260815)
    print("\n  class: C1 (cosh C2 (tau - d) - 1) + c, with C1 and C2 pinned;")
    print(f"  window x = C2 tau_end = {X:.0f}; rho_bar = {1e3 * RHO_BAR:.2f}"
          f" mN.m; |d| <= rho_bar/Mdot\n")
    ok, rows = True, []

    print("  (G1)/(G3)  displacement ratio  ||g_hat - f|| / ||e||\n")
    print(f"  {'Mdot':>6}{'||e||=||E||':>13}{'affine':>9}{'e = +E':>9}"
          f"{'worst of':>10}{'general':>9}{'holds':>7}")
    print(f"  {'N.m/s':>6}{'':13}{'P_V E':>9}{'':9}{'602':>10}{'(G1)':>9}")
    for mdot in (0.10, 0.20, 0.45, 0.80, 1.20):
        L, c1, f, E = build(mdot)
        dmax = RHO_BAR / mdot
        ds, G, qG = family(L, c1, dmax)
        chi = -c1 * C2 * np.sinh(C2 * L.tau)
        A = np.column_stack([chi, np.ones_like(chi)])
        coef, *_ = np.linalg.lstsq(A * L.s[:, None], E * L.s, rcond=None)
        aff = L.n(A @ coef) / L.n(E)
        wst, at_E = adversarial(L, c1, f, E, ds, G, qG, rng)
        ok &= wst <= 2.0 + 1e-9
        rows.append((mdot, L, c1, f, E, dmax, ds, G, qG, wst))
        print(f"  {mdot:6.2f}{L.n(E):13.5f}{aff:9.4f}{at_E:9.4f}"
              f"{wst:10.4f}{2.0:9.1f}{'yes' if wst <= 2 else 'NO':>7}")

    print(f"\n  (G4)  reach of the class, and the constant it gives\n")
    print(f"  {'Mdot':>6}{'1/kappa':>11}{'reach r':>11}{'||e||/r':>10}"
          f"{'1/(1-|e|/r)':>13}{'worst found':>13}")
    for mdot, L, c1, f, E, dmax, ds, G, qG, wst in rows:
        r_loc = curvature(L, c1, dmax)
        r = reach(L, qG, r_loc)
        ratio = L.n(E) / r
        ref = 1.0 / (1.0 - ratio) if ratio < 1 else np.inf
        ok &= wst <= ref + 1e-6
        print(f"  {mdot:6.2f}{r_loc:11.5f}{r:11.5f}{ratio:10.5f}"
              f"{ref:13.4f}{wst:13.4f}")

    print(f"\n  (D.13b) against (G3): three routes to the same |delta|\n")
    print(f"  {'Mdot':>6}{'<E,|chi|>/||chi||^2':>21}{'||E||/||chi||':>15}"
          f"{'rho_bar/Mdot':>14}{'agree':>7}")
    for mdot, L, c1, f, E, dmax, ds, G, qG, wst in rows:
        chi = np.abs(-c1 * C2 * np.sinh(C2 * L.tau))
        proj = L.ip(E, chi) / L.ip(chi, chi)
        cs = L.n(E) / L.n(chi)
        good = (abs(proj - RHO_BAR / mdot) < 1e-9
                and abs(cs - proj) < 1e-9)
        ok &= good
        print(f"  {mdot:6.2f}{proj:21.7f}{cs:15.7f}{RHO_BAR / mdot:14.7f}"
              f"{'yes' if good else 'NO':>7}")

    # The one assumption (G1) cannot do without: f in M.  The campaign's
    # fitted C2 spans 3.50 to 8.00, so this is not hypothetical, and the
    # bias it induces is a pure model error that no noise bound covers.
    print(f"\n  (G2)  when the class does NOT contain f: d_M = dist(f, M)")
    print(f"        data at C2_true, model pinned at {C2:.3f},"
          f" no noise at all, Mdot = 1.20\n")
    print(f"  {'C2 true':>9}{'err %':>8}{'d_M':>10}{'2||e||':>9}"
          f"{'ratio':>8}{'onset':>10}{'threshold':>12}{'vs rho_bar':>12}")
    print(f"  {'':9}{'':8}{'':10}{'':9}{'':8}{'ms':>10}{'mN.m':>12}{'':12}")
    for c2t in (4.10, 4.60, 5.046, 5.60, 6.40, 8.00):
        L = L2(np.linspace(0.0, X / C2, N_TAU))
        c1t = 1.20 / (J_P * c2t ** 2)
        f = c1t * (np.cosh(c2t * L.tau) - 1.0)
        E = RHO_BAR * np.sinh(c2t * L.tau) / (J_P * c2t)
        ds, G, qG = family(L, 1.20 / WZ, 0.150)    # +/-150 ms, the search cap
        dhat, gfit = fit_batch(L, f[None, :], ds, G, qG)
        dm = float(L.n(gfit[0] - f))
        en = 2 * L.n(E)
        dth = 1.20 * float(dhat[0])                # Mdot * onset bias
        print(f"  {c2t:9.3f}{100 * (c2t / C2 - 1):+8.1f}{dm:10.5f}"
              f"{en:9.5f}{dm / en:8.2f}{1e3 * float(dhat[0]):10.1f}"
              f"{1e3 * dth:12.2f}{abs(dth) / RHO_BAR:11.1f}x")

    print(f"\n  {'PASS' if ok else 'FAIL'}  every claim above")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
