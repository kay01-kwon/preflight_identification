#!/usr/bin/env python3
"""The pivot-arm estimate: what the motion capture actually sees.

The critical moment fixes the balance of the vehicle about its
landing-gear contact line, and turning it into a CoM offset needs the
distance from the body origin to that line.  It is not assumed: while
the vehicle tips, the marker rides a circle whose centre lies on the
ground plane, so the pivot distance is the centre of the circle the
mocap trajectory traces out.

  (a) the interval the fit runs over -- from the detected onset to the
      end of the clean pivoting phase, with the horizontal excursion
      and the tilt that bound it
  (b) the geometry the fit recovers, to scale: the contact line on the
      ground, the radius arm, and the arc the marker sweeps
  (c) the radial residual of every fitted sample about the circle,
      against the scatter of the fit

Usage: python analysis/pivot_arc_figure.py [case] [axis] [run] [out.png]
"""
import contextlib
import io
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import critical_value_getter_piecewise as cvp
from constrained_calibration import ROOT
from utils.extractor import load_excitation_dataset

FIT = '#c0392b'          # the fitted circle and the onset
SIG = '#1f3b66'          # measured signal
WIN = '#f39c12'          # the fit interval
GND = '#5d6d7e'          # ground plane and pivot


def body_excursion(bag, onset_time, axis):
    """Horizontal marker excursion in the tipping direction [m].

    Same construction as estimate_pivot_from_mocap: the mocap position
    is referred to the pre-excitation rest point and de-rotated by the
    initial yaw, so the excursion is read in the body frame.
    """
    t_mc = cvp.align_mocap_time(bag)
    px, py = bag.pose.position[:, 0], bag.pose.position[:, 1]
    q0 = bag.pose.quaternion[0]
    yaw0 = cvp.quat_to_yaw(q0[0], q0[1], q0[2], q0[3])
    idle = t_mc < onset_time * 0.5
    if np.sum(idle) < 10:
        idle = np.arange(len(t_mc)) < 50
    dx, dy = px - np.mean(px[idle]), py - np.mean(py[idle])
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    return t_mc, (dx * c - dy * s) if axis == 'y' else (dx * s + dy * c)


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else 'case_05'
    axis_dir = sys.argv[2] if len(sys.argv) > 2 else 'My'
    run = sys.argv[3] if len(sys.argv) > 3 else 'pos_My_065'
    out = sys.argv[4] if len(sys.argv) > 4 else 'docs/exp_pivot_arc.png'
    axis = 'x' if axis_dir == 'Mx' else 'y'

    # the reported detector, with the rig constants it is calibrated
    # with -- the same call analysis/pivot_geom.py makes
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(ROOT / case / axis_dir)
        crits, _ = cvp.extract_piecewise_batch(bags, axis)
    crit = next(c for c in crits if c.bag_name == run)
    bag = next(b for b in bags if b.name == run)
    with contextlib.redirect_stdout(io.StringIO()):
        p = cvp.estimate_pivot_from_mocap(bag, crit.onset_time, axis)

    xi, z, cx, R = p['xy_fit'], p['z_fit'], p['cx'], p['R']
    rad = np.hypot(xi - cx, z)
    res = rad - R

    t_mc, d = body_excursion(bag, crit.onset_time, axis)
    tilt = cvp._tilt_deg(bag.pose.quaternion)
    i0 = int(np.searchsorted(t_mc, crit.onset_time))
    i1 = i0 + p['N'] - 1
    chk = float(np.max(np.abs(d[i0:i1 + 1] * 1e3 - xi)))

    print(f'{case}/{axis_dir} {bag.name}')
    print(f'  onset t   = {crit.onset_time:.3f} s   '
          f'M_crit = {crit.onset_moment:+.3f} N.m')
    print(f'  interval  = {t_mc[i0]:.3f} -> {t_mc[i1]:.3f} s  '
          f'({p["N"]} mocap samples, {t_mc[i1] - t_mc[i0]:.2f} s)')
    print(f'  ends at   : excursion {abs(xi[-1]):.1f} mm, '
          f'tilt {tilt[i1]:.1f} deg (cap 10.0)')
    print(f'  l_p*      = {abs(cx):.1f} mm    R* = {R:.1f} mm')
    print(f'  residual  = {p["residual"]:.2f} mm (1 sigma), '
          f'max |r-R*| = {np.max(np.abs(res)):.2f} mm')
    print(f'  excursion reconstruction agrees to {chk:.2e} mm')

    fig, ax = plt.subplots(3, 1, figsize=(4.6, 7.2),
                           gridspec_kw=dict(height_ratios=[1.0, 1.45, 0.85],
                                            hspace=0.60))

    # ── (a) the interval the fit runs over ───────────────────────
    A = ax[0]
    lo, hi = crit.onset_time - 0.6, t_mc[i1] + 0.6
    keep = (t_mc >= lo) & (t_mc <= hi)
    A.axvspan(t_mc[i0], t_mc[i1], color=WIN, alpha=0.15, lw=0, zorder=0)
    A.plot(t_mc[keep], d[keep] * 1e3, '-', lw=1.2, color=SIG, zorder=3)
    A.axvline(crit.onset_time, ls='--', lw=1.2, color=FIT, zorder=2)
    A.set_ylabel(r'$\xi$ [mm]', fontsize=9, color=SIG)
    A.tick_params(axis='y', labelcolor=SIG)
    A2 = A.twinx()
    A2.plot(t_mc[keep], tilt[keep], '--', lw=1.1, color='#7d3c98', zorder=3)
    A2.axhline(10.0, ls=':', lw=1.0, color='#7d3c98')
    A2.plot([t_mc[i1]], [tilt[i1]], 'o', ms=6, mfc='none', mec='#7d3c98',
            mew=1.4, zorder=4)
    A2.text(t_mc[keep][0], 10.0, r' tilt cap $10^{\circ}$', color='#7d3c98',
            fontsize=7.6, ha='left', va='bottom')
    A2.set_ylabel('tilt [deg]', fontsize=9, color='#7d3c98')
    A2.tick_params(axis='y', labelcolor='#7d3c98', labelsize=8)
    A.set_xlabel('$t$ [s]', fontsize=9)
    A.set_title('(a) fit interval', fontsize=9.5)
    A.text(crit.onset_time, A.get_ylim()[1], ' onset ', color=FIT,
           fontsize=7.8, ha='left', va='top')

    # ── (b) the arc the marker sweeps, and the circle fitted to it ──
    B = ax[1]
    th = np.arctan2(z, xi - cx)
    arc = np.linspace(th.min() - 0.05, th.max() + 0.05, 300)
    B.plot(cx + R * np.cos(arc), R * np.sin(arc), '-', lw=1.4, color=FIT,
           zorder=3, label=rf'circle, centre on $z=0$')
    B.plot(xi, z, 'o', ms=3.4, mfc='none', mec=SIG, mew=1.0, zorder=4,
           label='mocap marker')
    B.set_xlabel(r'$\xi$ [mm]', fontsize=9)
    B.set_ylabel('$z$ [mm]', fontsize=9)
    B.set_title('(b) arc fit  (inset: same fit to scale)', fontsize=9.5)
    B.legend(fontsize=7.4, loc='lower left', framealpha=0.9)
    B.text(0.03, 0.97,
           '\n'.join((rf'$l_p^{{*}} = {abs(cx):.1f}$ mm',
                      rf'$R^{{*}} = {R:.1f}$ mm')),
           transform=B.transAxes, fontsize=8.2, va='top',
           bbox=dict(fc='white', ec='0.8', lw=0.6, pad=3.0))

    # the same fit drawn to scale, so the centre on the ground plane
    # and the length the arc actually subtends are not lost to the
    # stretched axes of the main panel
    ins = B.inset_axes([0.60, 0.06, 0.38, 0.54])
    ins.set_aspect('equal')
    wide = np.linspace(th.min() - 0.30, th.max() + 0.30, 300)
    ins.plot(cx + R * np.cos(wide), R * np.sin(wide), '-', lw=1.0,
             color=FIT, zorder=3)
    ins.plot([cx, xi[0]], [0, z[0]], '-', lw=0.9, color=GND, zorder=2)
    ins.plot([cx], [0], 'v', ms=6, color=GND, zorder=5)
    ins.plot([0], [0], 's', ms=4, color='0.25', zorder=5)
    ins.axhline(0, lw=1.0, color=GND, zorder=1)
    ins.annotate('', (cx, -0.13 * R), (0, -0.13 * R),
                 arrowprops=dict(arrowstyle='<->', lw=0.8, color=GND))
    ins.text(0.5 * cx, -0.18 * R, r'$l_p^{*}$', fontsize=8, color=GND,
             ha='center', va='top')
    ins.text(cx + 0.55 * (xi[0] - cx), 0.55 * z[0], r' $R^{*}$',
             fontsize=8, color=GND, ha='left', va='center')
    ins.set_xlim(min(0, cx) - 0.22 * R, max(xi.max(), cx) + 0.22 * R)
    ins.set_ylim(-0.46 * R, 1.16 * R)
    ins.set_xticks([])
    ins.set_yticks([])


    # ── (c) radial residual about the fitted circle ──────────────
    C = ax[2]
    C.axhspan(-p['residual'], p['residual'], color='#2874a6', alpha=0.15,
              lw=0, zorder=0)
    C.plot(xi, res, 'o-', ms=2.6, lw=0.8, color=SIG, zorder=3)
    C.axhline(0, lw=0.6, color='0.5')
    C.set_xlabel(r'$\xi$ [mm]', fontsize=9)
    C.set_ylabel(r'$r_i-R^{*}$ [mm]', fontsize=9)
    C.set_title('(c) radial residual', fontsize=9.5)
    C.text(0.97, 0.06, rf'$\sigma = {p["residual"]:.2f}$ mm  '
           rf'($N = {p["N"]}$)', transform=C.transAxes, fontsize=8,
           ha='right', va='bottom')

    for A_ in ax:
        A_.tick_params(labelsize=8)
        A_.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=600, bbox_inches='tight')
    print(f'written {out}')


if __name__ == '__main__':
    main()
