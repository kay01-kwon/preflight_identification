#!/usr/bin/env python3
"""The five experimental CoM-offset cases in the offset plane.

Load-cell truth (Table 10) against the identifiable rectangle of the
pre-flight excitation at f/W = 0.70 -- (1 - f/W) l_p per axis, 42 mm on
pitch (x) and 33 mm on roll (y) -- and the declared 20 x 20 mm design
box.  E1 is the unloaded vehicle (3.066 kg), E2-E5 carry the 3.220 kg
loaded configuration.

Usage: python analysis/exp_case_map_figure.py [out.png]
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from freeflight_metrics import MASS, OFFSET          # noqa: E402

L_ROLL, L_PITCH, THRUST_FRAC = 0.110, 0.140, 0.70
XLIM = (1 - THRUST_FRAC) * L_PITCH * 1e3           # identifiable on x (pitch)
YLIM = (1 - THRUST_FRAC) * L_ROLL * 1e3            # identifiable on y (roll)
NUDGE = {'01': (-2.2, -2.6), '02': (-2.2, -2.8), '03': (2.0, -2.6), '04': (2.0, 1.8), '05': (-2.2, 1.8)}


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'docs/exp_case_offsets.png'
    fig, ax = plt.subplots(figsize=(5.4, 5.0))

    def box(bx, by, **kw):
        ax.plot([-bx, bx, bx, -bx, -bx], [-by, -by, by, by, -by], **kw)

    box(XLIM, YLIM, color='#0072B2', lw=1.8, zorder=2,
        label=rf'identifiable, ${XLIM:.0f}\times{YLIM:.0f}$ mm ($f/W = {THRUST_FRAC:.2f}$)')
    box(20, 20, color='#D55E00', lw=1.4, ls='--', zorder=2, label=r'declared design box, $20\times20$ mm')
    ax.axhline(0, lw=0.5, color='0.7', zorder=1)
    ax.axvline(0, lw=0.5, color='0.7', zorder=1)
    for case, (x, y) in OFFSET.items():
        loaded = MASS[case] > 3.1
        ax.plot(x, y, 's' if loaded else 'o', ms=8.5 if loaded else 8, mfc='#1B7837' if loaded else '#4DAF4A',
                mec='0.2', mew=0.9, zorder=4)
        dx, dy = NUDGE[case]
        ax.annotate(rf'E{int(case)}  ({x:+.1f}, {y:+.1f})', (x, y), xytext=(dx * 3, dy * 3),
                    textcoords='offset points', fontsize=8.5,
                    ha='right' if dx < 0 else 'left', va='top' if dy < 0 else 'bottom')
    ax.plot([], [], 'o', ms=8, mfc='#4DAF4A', mec='0.2', mew=0.9, ls='', label='E1: unloaded, 3.066 kg')
    ax.plot([], [], 's', ms=8.5, mfc='#1B7837', mec='0.2', mew=0.9, ls='', label='E2–E5: loaded, 3.220 kg')
    r = np.hypot(*np.array(list(OFFSET.values())).T)
    print('radial offsets [mm]:', ', '.join(f'E{int(c)} {v:.1f}' for c, v in zip(OFFSET, r)))
    ax.set_xlim(-48, 48)
    ax.set_ylim(-38, 38)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x_{\mathrm{off}}$ [mm]  (identified by the pitch excitation $M_y$)', fontsize=9.5)
    ax.set_ylabel(r'$y_{\mathrm{off}}$ [mm]  (roll excitation $M_x$)', fontsize=9.5)
    ax.set_title('experimental cases: load-cell CoM offset (Table 10)', fontsize=10)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    ax.grid(alpha=0.25, lw=0.4)
    ax.tick_params(labelsize=8.5)
    fig.tight_layout()
    fig.savefig(out, dpi=600)
    print('written', out)


if __name__ == '__main__':
    main()
