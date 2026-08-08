"""Would a rolling contact show up in the mocap circle fit?

The residual moment falls linearly with tilt, which is a restoring
stiffness k = 2.52 N.m/rad.  A contact that ROLLS on a foot of radius r
supplies exactly that, k = W r, needing r = 80 mm.

Earlier this was dismissed on the grounds that the marker's arc fits a
circle to 0.1-0.2 mm and an 8 mm drift "would leave several mm".  That
was an estimate, and it is wrong.  Fitting the trochoid with the same
routine -- centre held at ground level, horizontal offset and radius
free -- gives:

    foot r [mm]    circle RMS [mm]    fitted l_p [mm]    fitted R [mm]
         0               0.000              140.2            346.8
        20               0.024              156.7            353.8
        80               0.070              260.3            410.5

Even an 80 mm roll leaves less residual than the measurement shows.  The
circle fit absorbs the trochoid almost entirely by inflating cx and R.

The exclusion is real but it comes from the FITTED GEOMETRY, not the
residual: at r = 80 mm the fit would return l_p = 260 mm, and the runs
return 140.4 +- 3.6 mm.  At 1.5 mm of l_p per mm of foot radius, that
scatter allows r <~ 2.4 mm, hence a stiffness W r <= 1.3 mN.m/deg
against the 44 needed -- three per cent.
"""
import numpy as np

R_MARK, A_M, H_M = 0.3465, 0.1402, 0.3172     # from the mocap fit
SPAN_DEG = 6.0
N = 60


def trochoid(r, span_deg=SPAN_DEG, n=N):
    """Marker path when the contact rolls on a foot of radius r."""
    ph = np.radians(np.linspace(0, span_deg, n))
    # foot centre sits at height r, translating by r*phi as it rolls
    cx, cz = r * ph, np.full_like(ph, r)
    ox, oz = A_M, H_M - r                     # marker offset from foot centre
    x = cx + ox * np.cos(ph) - oz * np.sin(ph)
    z = cz + ox * np.sin(ph) + oz * np.cos(ph)
    return x - x[0], z          # estimator uses displacement from rest


def fit_circle_cz0(x, z):
    """Least-squares circle with the centre pinned to z = 0 (as in the code)."""
    # minimise sum ((x-cx)^2 + z^2 - R^2)^2 over cx, R -- linear in (cx, R^2)
    A = np.column_stack([-2 * x, np.ones_like(x)])
    b = -(x ** 2 + z ** 2)
    (cx, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    R = np.sqrt(max(cx ** 2 - c, 1e-12))
    res = np.sqrt((x - cx) ** 2 + z ** 2) - R
    return cx, R, float(np.sqrt(np.mean(res ** 2)))


print(f"marker at {A_M*1e3:.1f} mm out, {H_M*1e3:.1f} mm up, R = {R_MARK*1e3:.1f} mm")
print(f"arc span {SPAN_DEG} deg = {R_MARK*np.radians(SPAN_DEG)*1e3:.0f} mm\n")
print(f"{'foot radius':>12}{'contact drift':>15}{'fitted l_p':>12}{'fitted R':>10}"
      f"{'circle RMS':>12}")
print(f"{'[mm]':>12}{'over 6 deg [mm]':>15}{'[mm]':>12}{'[mm]':>10}{'[mm]':>12}")
for r in (0.0, 0.005, 0.010, 0.020, 0.040, 0.080):
    x, z = trochoid(r)
    cx, R, rms = fit_circle_cz0(x, z)
    print(f"{r*1e3:12.0f}{r*np.radians(SPAN_DEG)*1e3:15.1f}"
          f"{abs(cx)*1e3:12.1f}{R*1e3:10.1f}{rms*1e3:12.3f}")
print()
print("measured circle RMS over the 138 runs: 0.1-0.2 mm")
