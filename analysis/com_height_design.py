#!/usr/bin/env python3
"""Design a two-scale tilt measurement of the CoM height z_CoM.

NOTE ON SYMBOLS: the height measured here is z_CoM, the CoM height
above the landing-gear contact plane -- the same datum the pivot uses.
It is NOT the rotor-hub height h = 0.315 m of the ground-effect model,
which this repository already calls h; z is used throughout below to
keep the two apart.

The vehicle rests on two scales that sit on a rigid board; the board is
then tilted so the scales stay perpendicular to it and read the normal
reaction.  Taking moments about support A, with the CoM a horizontal
distance x from A (in the board plane) and a height z above the support
plane,

    N_B = (W / d) * (x cos(phi) + z sin(phi))                        (1)

so a single tilted reading gives

    z = (N_B d / W - x cos(phi)) / sin(phi).                         (2)

x is already known: it is the load-cell CoM offset plus the support
geometry, so the level reading is only a consistency check.  Any
restraint that stops the vehicle sliding must be placed AT support A,
where it contributes no moment.

Tilting BOTH ways removes x altogether.  Reading the downhill scale in
each direction gives N_B(+phi) = (W/d)(x cos + z sin) and
N_A(-phi) = (W/d)((d-x) cos + z sin), whose sum no longer contains x:

    z = ( (N_A + N_B) d / W - d cos(phi) ) / (2 sin(phi))            (3)

which is the same pivot-free averaging that makes M_ff insensitive to a
common-mode arm error.  Use (3) as the measurement and (2) per
direction as a cross-check on the known offset.

Angles come from the Pixhawk roll/pitch, differenced against a reading
on the level board, which removes the IMU mounting offset.

This script prints the reading to expect and the resulting uncertainty,
so the tilt angle and scale resolution can be chosen before going to
the bench.
"""
import numpy as np

W = 31.59            # N, vehicle weight
D = 0.28             # m, support separation along the tilt direction
X = 0.150            # m, CoM horizontal distance from support A
G = 9.81
Z_HYP = (0.205, 0.261)          # the two hypotheses to separate
SCALE_G = (1.0, 5.0, 10.0)      # scale resolution [g]
PHI_DEG = (10, 15, 20, 25, 30, 35, 40)
DPHI_DEG = 0.3       # Pixhawk attitude accuracy, static


def n_b(phi, z, x=X, d=D, w=W):
    return (w / d) * (x * np.cos(phi) + z * np.sin(phi))


def z_from(nb, phi, x=X, d=D, w=W):
    """Eq. (2): one tilt direction, needs the known offset x."""
    return (nb * d / w - x * np.cos(phi)) / np.sin(phi)


def z_two_sided(na, nb, phi, d=D, w=W):
    """Eq. (3): both tilt directions, independent of x."""
    return ((na + nb) * d / w - d * np.cos(phi)) / (2.0 * np.sin(phi))


print(f"W = {W:.2f} N ({W/G:.3f} kg), support separation d = {D*1e3:.0f} mm, "
      f"x = {X*1e3:.0f} mm\n")
print(f"{'tilt':>5} | {'N_B [kg] at z=':>16} | {'separation':>11} | "
      f"{'sigma_z [mm] for scale resolution':>34}")
print(f"{'[deg]':>5} | {Z_HYP[0]:>7.3f}{Z_HYP[1]:>9.3f} | "
      f"{'[g]':>11} | " + ''.join(f"{f'{s:.0f} g':>11}" for s in SCALE_G))
print('-' * 84)
for pd in PHI_DEG:
    phi = np.radians(pd)
    nb = [n_b(phi, z) for z in Z_HYP]
    sep = 1e3 * (nb[1] - nb[0]) / G                      # grams
    cells = []
    for s in SCALE_G:
        dn = s * 1e-3 * G                                # N
        dz_load = dn * D / (W * np.sin(phi))
        # angle term: differentiate (2) at fixed N_B
        z0 = Z_HYP[0]
        dz_ang = abs(z_from(n_b(phi, z0), phi + np.radians(DPHI_DEG)) - z0)
        cells.append(1e3 * np.hypot(dz_load, dz_ang))
    print(f"{pd:5d} | {nb[0]/G:7.3f}{nb[1]/G:9.3f} | {sep:11.0f} | "
          + ''.join(f"{c:11.1f}" for c in cells))
print('-' * 84)
print(f"sigma_z combines the scale resolution with a {DPHI_DEG} deg attitude "
      f"error, in quadrature.")
print(f"The two hypotheses differ by {1e3*(Z_HYP[1]-Z_HYP[0]):.0f} mm, so any "
      f"row whose sigma_z is below ~15 mm separates them decisively.")

print("\ntwo-sided reading, Eq. (3) -- x drops out:")
print(f"{'tilt':>5} | {'N_A + N_B [kg]':>22} | {'sigma_z [mm], 10 g scale':>26}")
print(f"{'[deg]':>5} | {'z=0.205':>10}{'z=0.261':>12} |")
print('-' * 60)
for pd in (15, 20, 25, 30, 35):
    phi = np.radians(pd)
    tot = []
    for z in Z_HYP:
        na = (W / D) * ((D - X) * np.cos(phi) + z * np.sin(phi))
        nb = n_b(phi, z)
        tot.append((na + nb) / G)
        assert abs(z_two_sided(na, nb, phi) - z) < 1e-9
    dn = 10e-3 * G * np.sqrt(2)               # two independent readings
    dz_load = dn * D / (2 * W * np.sin(phi))
    z0 = Z_HYP[0]
    na0 = (W / D) * ((D - X) * np.cos(phi) + z0 * np.sin(phi))
    dz_ang = abs(z_two_sided(na0, n_b(phi, z0), phi + np.radians(DPHI_DEG)) - z0)
    print(f"{pd:5d} | {tot[0]:10.3f}{tot[1]:12.3f} | "
          f"{1e3*np.hypot(dz_load, dz_ang):26.1f}")
print("\nx is not needed for (3); the per-direction readings still test it.")
