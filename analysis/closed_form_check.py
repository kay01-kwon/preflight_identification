#!/usr/bin/env python3
"""Check hand-derived closed forms against numerics, three ways.

Every closed form in Sec. VI-E and its appendix was obtained by hand.
Three kinds of slip get past a reading of the algebra, and all three
have actually happened here:

  1. a term dropped in a quotient rule, when the numerator is itself an
     integral -- the derivative then looks plausible and has the right
     sign, but the wrong magnitude by orders;
  2. a sign inside sinh^2 u = (cosh 2u - 1)/2, or a missing 1/2 from the
     inner derivative of cosh 2u;
  3. a check run at small x that reads floating-point noise, because the
     closed forms cancel four or more orders there and double precision
     is exhausted.

So this harness tests each claim three ways.

  VALUE  the closed form against a numerical reference (quadrature for
         an integral, a central difference for a derivative), over a
         window chosen so that (3) cannot happen.

  ORDER  the exponent of the leading term at x = 0, evaluated in
         80-digit decimal arithmetic.  This is the sharpest test of the
         three and the cheapest to reason about: R -> 1/7 forces
         R' -> 0, so a claimed R' that behaves as x^-8 is wrong no
         matter how right its sign looks.

  SAFE   the x below which double precision has lost the value to
         cancellation.  Reported, not assumed: it is why the value test
         runs where it does.

Two known-bad formulas are included deliberately, marked xfail, so a
passing run also demonstrates that the harness has teeth.

Adding a claim: write the closed form as a function of (x, m), where m
supplies sinh/cosh/tanh/exp, and give a numerical reference.  Use only
integer literals and division by integers -- write m.sinh(2*x)/4, never
0.25*m.sinh(2*x) -- so the same source runs in float and in decimal.

Usage: python analysis/closed_form_check.py [-v]
"""
import sys
from decimal import Decimal, getcontext

import numpy as np
from scipy.integrate import quad

getcontext().prec = 80


class HP:
    """Decimal-backed sinh/cosh/tanh, same call signature as numpy."""

    @staticmethod
    def exp(x):
        return Decimal(x).exp()

    @staticmethod
    def sinh(x):
        e = Decimal(x).exp()
        return (e - 1 / e) / 2

    @staticmethod
    def cosh(x):
        e = Decimal(x).exp()
        return (e + 1 / e) / 2

    @staticmethod
    def tanh(x):
        e = Decimal(2 * x).exp()
        return (e - 1) / (e + 1)


# --- the shapes everything is built from -----------------------------
def lam(x, m):
    return m.sinh(x) - x


def lam_p(x, m):
    return m.cosh(x) - 1


def int_lam2(x, m):
    """Int_0^x Lambda(u)^2 du."""
    return m.sinh(2 * x) / 4 - x / 2 - 2 * (x * m.cosh(x) - m.sinh(x)) + x ** 3 / 3


def int_ulam(x, m):
    """Int_0^x u Lambda(u) du."""
    return x * m.cosh(x) - m.sinh(x) - x ** 3 / 3


# backend-generic quantities, then their float-only aliases
R_phi_m = lambda x, m: int_lam2(x, m) / (x * lam(x, m) ** 2)
R_GE_m = lambda x, m: int_ulam(x, m) / (x ** 2 * lam(x, m))
Psi_m = lambda x, m: x / m.tanh(x / 2)
D_gr_m = lambda x, m: m.tanh(x / 2) * lam(x, m) ** 2 / 2 - int_lam2(x, m)
E_ge_m = lambda x, m: x * m.tanh(x / 2) * lam(x, m) - int_ulam(x, m)

L = lambda x: lam(x, np)
Lp = lambda x: lam_p(x, np)
R_phi = lambda x: R_phi_m(x, np)
R_GE = lambda x: R_GE_m(x, np)
D_gr = lambda x: D_gr_m(x, np)
E_ge = lambda x: E_ge_m(x, np)


def fd(f, x, rel=1 / 300):
    """Five-point central difference with a step proportional to x.

    Two competing errors have to be kept below the tolerance at once,
    and both bit on the first run of this file.

    Too small a step amplifies noise: the reference functions are built
    from cancelling closed forms, so their last digits are meaningless,
    and dividing that by 2h magnifies it.  A three-point stencil at
    h = 1e-5 failed dR_phi/dx this way.

    Too large a step truncates: for a quantity vanishing as x^n the
    five-point rule errs by about n(n-1)(n-2)(n-3)/30 * (h/x)^4 relative,
    which for D' (n = 7) at h = 1e-2, x = 0.5 is 2e-6 -- exactly the
    discrepancy that run reported, and nothing to do with the formula.

    A step proportional to x holds (h/x)^4 fixed at 1e-10 instead, which
    clears both.
    """
    h = rel * x
    return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h)


class Claim:
    """One hand-derived closed form and how to check it."""

    def __init__(self, name, closed, ref, order=None, xfail=False, note=''):
        self.name, self.closed, self.ref = name, closed, ref
        self.order, self.xfail, self.note = order, xfail, note

    def safe_x(self, grid=np.geomspace(1e-3, 3.0, 60), want=8):
        """Smallest x on the grid at which float64 still holds `want` digits."""
        for x in grid:
            hi = self.closed(Decimal(repr(float(x))), HP)
            if hi == 0:
                continue
            err = abs(float(self.closed(float(x), np)) / float(hi) - 1)
            if err < 10.0 ** -want:
                return float(x)
        return float(grid[-1])

    def value_err(self, xs):
        out = 0.0
        for x in xs:
            c, r = float(self.closed(x, np)), float(self.ref(x))
            out = max(out, abs(c - r) / max(abs(r), 1e-30))
        return out

    def leading_order(self, a=Decimal('1e-3'), b=Decimal('1e-4')):
        fa, fb = self.closed(a, HP), self.closed(b, HP)
        if fa == 0 or fb == 0:
            return float('nan')
        return float((fa / fb).ln() / (a / b).ln())


CLAIMS = [
    Claim('Int_0^x Lam^2 du', int_lam2,
          lambda x: quad(lambda u: L(u) ** 2, 0, x, limit=200)[0], order=7,
          note='(1/4)sinh2x - x/2 - 2(x cosh x - sinh x) + x^3/3'),
    Claim('Int_0^x u Lam du', int_ulam,
          lambda x: quad(lambda u: u * L(u), 0, x, limit=200)[0], order=5,
          note='x cosh x - sinh x - x^3/3'),
    Claim('sinh^2 u = (cosh 2u - 1)/2',
          lambda x, m: (m.cosh(2 * x) - 1) / 2,
          lambda x: np.sinh(x) ** 2, order=2,
          note='the +2/4 vs -2/4 slip'),

    Claim('dR_phi/dx',
          lambda x, m: (x * lam(x, m) ** 3
                        - int_lam2(x, m) * (lam(x, m) + 2 * x * lam_p(x, m)))
                       / (x ** 2 * lam(x, m) ** 3),
          lambda x: fd(R_phi, x), order=1,
          note='numerator carries Int_0^x Lam^2'),
    Claim('dR_GE/dx',
          lambda x, m: (x ** 2 * lam(x, m) ** 2
                        - int_ulam(x, m) * (2 * lam(x, m) + x * lam_p(x, m)))
                       / (x ** 3 * lam(x, m) ** 2),
          lambda x: fd(R_GE, x), order=1,
          note='numerator carries Int_0^x u Lam'),

    Claim('D\' = Lam * phi / (4 cosh^2(x/2))',
          lambda x, m: lam(x, m) * (x + 2 * x * m.cosh(x) - 3 * m.sinh(x))
                       / (4 * m.cosh(x / 2) ** 2),
          lambda x: fd(D_gr, x), order=6, note='(100), gravity bound'),
    Claim('E\' = (Lam^2 + x^2 Lam\') / (cosh x + 1)',
          lambda x, m: (lam(x, m) ** 2 + x ** 2 * lam_p(x, m))
                       / (m.cosh(x) + 1),
          lambda x: fd(E_ge, x), order=4, note='(100a), ground-effect bound'),

    Claim('manuscript (B4):  -2(cosh x - 1)/(x Lam^3)',
          lambda x, m: -2 * lam_p(x, m) / (x * lam(x, m) ** 3),
          lambda x: fd(R_phi, x), order=1, xfail=True,
          note='claimed dR_phi/dx'),
    Claim('manuscript (B9):  -(Lam + x(cosh x - 1))/(x^3 Lam^2)',
          lambda x, m: -(lam(x, m) + x * lam_p(x, m))
                       / (x ** 3 * lam(x, m) ** 2),
          lambda x: fd(R_GE, x), order=1, xfail=True,
          note='claimed dR_GE/dx'),
]

# --- limits and signs -------------------------------------------------
# The x -> 0 limits MUST be taken in decimal.  R_phi at x = 3e-3 is a
# ratio whose numerator is 1e-18 of the terms that build it, so float64
# returns 3.02 for a quantity whose true value is 1/7.  That is not a
# hypothetical: it is what the first run of this file printed.
#
# The x -> infinity limits need the opposite care.  Psi R_GE approaches
# 1 as 1 - 1/x, so no reachable x gets close; at x = 60 it is 0.9833 and
# a naive tolerance calls it a failure.  One Richardson step against a
# 1/x tail, 2 f(2x) - f(x), removes that term and lands on the limit.
LIMITS = [
    ('R_phi(0+)', R_phi_m, 0, 1 / 7),
    ('R_GE (0+)', R_GE_m, 0, 1 / 5),
    ("Psi_phi(0+) = x Lam'/Lam", lambda x, m: x * lam_p(x, m) / lam(x, m), 0, 3.0),
    ('Psi(0+)', Psi_m, 0, 2.0),
    ('Psi R_phi (inf)', lambda x, m: Psi_m(x, m) * R_phi_m(x, m), np.inf, 0.5),
    ('Psi R_GE  (inf)', lambda x, m: Psi_m(x, m) * R_GE_m(x, m), np.inf, 1.0),
]
SIGNS = [
    ('D >= 0   (Psi R_phi <= 1/2)', D_gr_m),
    ('E >= 0   (Psi R_GE  <= 1)', E_ge_m),
    ('1/7 - R_phi >= 0', lambda x, m: 1 - 7 * R_phi_m(x, m)),
    ('1/5 - R_GE  >= 0', lambda x, m: 1 - 5 * R_GE_m(x, m)),
]


def main():
    verbose = '-v' in sys.argv
    print('closed forms: value, leading order at x = 0, and the double-'
          'precision floor\n')
    print(f"  {'claim':44}{'safe x':>8}{'val err':>10}{'order':>8}"
          f"{'want':>6}{'':>3}verdict")
    bad = 0
    for c in CLAIMS:
        xs = np.geomspace(max(c.safe_x(), 0.5), 6.0, 12)
        ve, od = c.value_err(xs), c.leading_order()
        ok = ve < 1e-6 and (c.order is None or abs(od - c.order) < 0.05)
        mark = ('xfail OK' if not ok else 'XPASS?!') if c.xfail else \
               ('ok' if ok else 'FAIL')
        if c.xfail and ok:
            bad += 1
        if not c.xfail and not ok:
            bad += 1
        print(f"  {c.name:44}{c.safe_x():8.3f}{ve:10.1e}{od:8.2f}"
              f"{c.order if c.order is not None else '-':>6}   {mark}")
        if verbose and c.note:
            print(f"  {'':44}{c.note}")

    print('\n  "safe x" is where float64 still holds 8 digits against 80-digit'
          '\n  decimal; below it a check reads cancellation noise, not error.')

    print('\nlimits   (x -> 0 in 80-digit decimal; x -> inf Richardson-extrapolated)\n')
    print(f"  {'quantity':30}{'claimed':>10}{'raw':>12}{'extrap':>12}{'':>3}verdict")
    for name, f, at, want in LIMITS:
        if at == 0:
            raw = float(f(Decimal('1e-5'), HP))
            est = raw
        else:
            a, b = float(f(30.0, np)), float(f(60.0, np))
            raw, est = b, 2 * b - a          # kills a 1/x tail
        ok = abs(est - want) < 1e-4 * max(abs(want), 1)
        bad += not ok
        print(f"  {name:30}{want:10.6f}{raw:12.6f}{est:12.6f}"
              f"   {'ok' if ok else 'FAIL'}")

    print('\nsigns over x in [0.05, 12], in decimal\n')
    g = [Decimal(repr(float(t))) for t in np.linspace(0.05, 12.0, 200)]
    for name, f in SIGNS:
        v = [float(f(t, HP)) for t in g]
        ok = min(v) >= 0.0
        bad += not ok
        print(f"  {name:32}min = {min(v):+.3e}   {'ok' if ok else 'FAIL'}")

    print(f"\n{'all checks behaved as declared' if not bad else str(bad) + ' problem(s)'}")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
