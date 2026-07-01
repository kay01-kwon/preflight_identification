"""
CoM Estimator
=============
Estimate mass and centre-of-mass offset from critical thrust / moment
pairs obtained at tip-over onset.

Pipeline (accounts for compliant roll landing gear)
----------------------------------------------------
1. Mass from My pairs only (pitch landing gear is rigid → x_p reliable)
       W = 0.5 · [ f_My_pos + f_My_neg - (M_y_neg - M_y_pos) / x_p ]

2. Recover effective y_p from Mx mass equation + known W
       y_p = (M_x_neg - M_x_pos) / (f_Mx_pos + f_Mx_neg - 2W)
   (roll landing gear is compliant → geometric y_p unreliable)

3. x_off from My + mass
       x_off_neg = x_p · (f_neg/W - M_neg/(W·x_p) - 1)
       x_off_pos = x_p · (1 - f_pos/W - M_pos/(W·x_p))

4. y_off from Mx + mass + recovered y_p
       y_off_neg = y_p · (1 - f_neg/W + M_neg/(W·y_p))
       y_off_pos = y_p · (f_pos/W + M_pos/(W·y_p) - 1)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.critical_value_loader import CriticalValueLoader


# ═════════════════════════════════════════════════════════════
#  Constants
# ═════════════════════════════════════════════════════════════

G = 9.81               # [m/s²]
X_P_CG = 0.250 / 2.0   # pitch pivot half-span [m]  (rigid — trusted)
Y_P_CG = 0.288 / 2.0   # roll  pivot half-span [m]  (geometric — NOT used directly)


# ═════════════════════════════════════════════════════════════
#  Result container
# ═════════════════════════════════════════════════════════════

@dataclass
class EstimationResult:
    """Estimation output."""
    # Mass (from My only)
    m_from_My: np.ndarray       # (N_My,) per-trial mass
    m_mean: float               # [kg]
    m_std: float                # [kg]

    # Recovered y_p
    y_p_samples: np.ndarray     # (N_Mx,) per-trial recovered y_p [m]
    y_p_mean: float             # [m]
    y_p_std: float              # [m]

    # CoM offsets
    x_off_samples: np.ndarray   # (2*N_My,) from pos & neg My
    y_off_samples: np.ndarray   # (2*N_Mx,) from pos & neg Mx
    x_off_mean: float           # [m]
    x_off_std: float            # [m]
    y_off_mean: float           # [m]
    y_off_std: float            # [m]


# ═════════════════════════════════════════════════════════════
#  CoMEstimator
# ═════════════════════════════════════════════════════════════

class CoMEstimator:
    """
    Estimate mass and CoM offset from critical thrust / moment pairs.

    Accounts for compliant roll landing gear by recovering
    effective y_p from data rather than using geometric value.

    Parameters
    ----------
    x_p : float   pitch pivot half-span [m]  (rigid, trusted)
    y_p : float   roll  pivot half-span [m]  (geometric, for reference only)
    g   : float   gravity [m/s²]
    """

    def __init__(
        self,
        x_p: float = X_P_CG,
        y_p: float = Y_P_CG,
        g: float = G,
    ):
        self.x_p = x_p
        self.y_p = y_p
        self.g = g

    # ── Core estimation ───────────────────────────────────

    def estimate(self, loader: CriticalValueLoader) -> EstimationResult:
        """
        Run mass + CoM offset estimation.

        Pipeline:
          1. Mass from My pairs (pitch gear rigid → x_p trusted)
          2. Recover y_p from Mx pairs + known W
          3. x_off from My + mass
          4. y_off from Mx + mass + recovered y_p
        """

        # ── Extract arrays ──
        f_Mx_pos = loader.get_array('Mx', 'pos', 'collective_thrust_N')
        f_Mx_neg = loader.get_array('Mx', 'neg', 'collective_thrust_N')
        M_x_pos  = loader.get_array('Mx', 'pos', 'moment_Nm')
        M_x_neg  = loader.get_array('Mx', 'neg', 'moment_Nm')

        f_My_pos = loader.get_array('My', 'pos', 'collective_thrust_N')
        f_My_neg = loader.get_array('My', 'neg', 'collective_thrust_N')
        M_y_pos  = loader.get_array('My', 'pos', 'moment_Nm')
        M_y_neg  = loader.get_array('My', 'neg', 'moment_Nm')

        # ── Validate trial counts ──
        N_Mx = min(len(f_Mx_pos), len(f_Mx_neg))
        N_My = min(len(f_My_pos), len(f_My_neg))

        if N_My == 0:
            raise ValueError(
                "No matched My pos/neg trial pairs found — cannot estimate mass.\n"
                f"  My: pos={len(f_My_pos)}, neg={len(f_My_neg)}\n"
                "Check that CSV files are present and filenames match the pattern:\n"
                "  critical_values_{pos|neg}_{My}_0_{N}.csv"
            )

        if len(f_Mx_pos) != len(f_Mx_neg):
            print(f"  [WARN] Mx trial count mismatch: "
                  f"pos={len(f_Mx_pos)}, neg={len(f_Mx_neg)}  → using first {N_Mx}")
            f_Mx_pos, f_Mx_neg = f_Mx_pos[:N_Mx], f_Mx_neg[:N_Mx]
            M_x_pos, M_x_neg = M_x_pos[:N_Mx], M_x_neg[:N_Mx]

        if len(f_My_pos) != len(f_My_neg):
            print(f"  [WARN] My trial count mismatch: "
                  f"pos={len(f_My_pos)}, neg={len(f_My_neg)}  → using first {N_My}")
            f_My_pos, f_My_neg = f_My_pos[:N_My], f_My_neg[:N_My]
            M_y_pos, M_y_neg = M_y_pos[:N_My], M_y_neg[:N_My]

        # ── Step 1: Mass from My pairs only ──
        W_from_My = 0.5 * (f_My_pos + f_My_neg
                           - (M_y_neg - M_y_pos) / self.x_p)
        m_from_My = W_from_My / self.g

        m_mean = float(np.mean(m_from_My))
        m_std = float(np.std(m_from_My, ddof=1)) if len(m_from_My) > 1 else 0.0
        W_avg = m_mean * self.g

        # ── Step 2: Recover y_p from Mx mass equation ──
        # W = 0.5 * (f_pos + f_neg - (M_neg - M_pos) / y_p)
        # → y_p = (M_neg - M_pos) / (f_pos + f_neg - 2W)
        y_p_samples = []
        for i in range(N_Mx):
            denom = f_Mx_pos[i] + f_Mx_neg[i] - 2.0 * W_avg
            if abs(denom) > 1e-10:
                y_p_i = (M_x_neg[i] - M_x_pos[i]) / denom
                y_p_samples.append(y_p_i)

        y_p_arr = np.array(y_p_samples)
        y_p_mean = float(np.mean(y_p_arr)) if len(y_p_arr) > 0 else self.y_p
        y_p_std = float(np.std(y_p_arr, ddof=1)) if len(y_p_arr) > 1 else 0.0

        # ── Step 3: x_off from My + mass ──
        x_off_neg = self.x_p * (f_My_neg / W_avg - M_y_neg / (W_avg * self.x_p) - 1.0)
        x_off_pos = self.x_p * (1.0 - f_My_pos / W_avg - M_y_pos / (W_avg * self.x_p))
        x_off_samples = np.empty(2 * N_My)
        x_off_samples[0::2] = x_off_neg
        x_off_samples[1::2] = x_off_pos

        # ── Step 4: y_off from Mx + mass + recovered y_p ──
        if N_Mx > 0 and len(y_p_arr) > 0:
            y_off_neg = y_p_mean * (1.0 - f_Mx_neg[:N_Mx] / W_avg
                                    + M_x_neg[:N_Mx] / (W_avg * y_p_mean))
            y_off_pos = y_p_mean * (f_Mx_pos[:N_Mx] / W_avg
                                    + M_x_pos[:N_Mx] / (W_avg * y_p_mean) - 1.0)
            y_off_samples = np.empty(2 * N_Mx)
            y_off_samples[0::2] = y_off_neg
            y_off_samples[1::2] = y_off_pos
        else:
            y_off_samples = np.array([])

        x_off_mean = float(np.mean(x_off_samples)) if len(x_off_samples) > 0 else 0.0
        y_off_mean = float(np.mean(y_off_samples)) if len(y_off_samples) > 0 else 0.0
        x_off_std = float(np.std(x_off_samples, ddof=1)) if len(x_off_samples) > 1 else 0.0
        y_off_std = float(np.std(y_off_samples, ddof=1)) if len(y_off_samples) > 1 else 0.0

        return EstimationResult(
            m_from_My=m_from_My,
            m_mean=m_mean,
            m_std=m_std,
            y_p_samples=y_p_arr,
            y_p_mean=y_p_mean,
            y_p_std=y_p_std,
            x_off_samples=x_off_samples,
            y_off_samples=y_off_samples,
            x_off_mean=x_off_mean,
            x_off_std=x_off_std,
            y_off_mean=y_off_mean,
            y_off_std=y_off_std,
        )

    # ── Print ─────────────────────────────────────────────

    @staticmethod
    def print_result(res: EstimationResult) -> None:
        print("\n" + "=" * 65)
        print("  Mass & CoM Offset Estimation Result")
        print("  (My→mass, Mx+mass→y_p recovery, then offsets)")
        print("=" * 65)

        print(f"\n── Step 1: Mass from My pairs ──")
        print(f"   per-trial : {res.m_from_My}  [kg]")
        print(f"   mean = {res.m_mean:.4f} kg,  std = {res.m_std:.4f} kg")

        print(f"\n── Step 2: Recovered y_p from Mx + known W ──")
        print(f"   per-trial : {res.y_p_samples * 1e3}  [mm]")
        print(f"   y_p = {res.y_p_mean * 1e3:.3f} ± {res.y_p_std * 1e3:.3f} mm")

        print(f"\n── Step 3: x_off from My ──")
        print(f"   samples : {res.x_off_samples * 1e3}  [mm]")
        print(f"   x_off = {res.x_off_mean * 1e3:+.3f} ± {res.x_off_std * 1e3:.3f} mm")

        print(f"\n── Step 4: y_off from Mx (using recovered y_p) ──")
        print(f"   samples : {res.y_off_samples * 1e3}  [mm]")
        print(f"   y_off = {res.y_off_mean * 1e3:+.3f} ± {res.y_off_std * 1e3:.3f} mm")

        print(f"\n{'='*65}")
        print(f"  SUMMARY")
        print(f"{'='*65}")
        print(f"  Mass   = {res.m_mean:.4f} ± {res.m_std:.4f} kg")
        print(f"  y_p    = {res.y_p_mean * 1e3:.3f} ± {res.y_p_std * 1e3:.3f} mm  (recovered)")
        print(f"  x_off  = {res.x_off_mean * 1e3:+.3f} ± {res.x_off_std * 1e3:.3f} mm")
        print(f"  y_off  = {res.y_off_mean * 1e3:+.3f} ± {res.y_off_std * 1e3:.3f} mm")
        print("=" * 65)
        print()

    # ── Plot ───────────────────────────────────────────────

    @staticmethod
    def plot_result(
        res: EstimationResult,
        save_dir=None,
        show: bool = True,
    ) -> None:
        """
        Two figures:
          Fig 1: x-y plane CoM offset scatter + mean + ellipse
          Fig 2: Strip plots for mass, y_p, x_off, y_off
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        from pathlib import Path

        x_mm = res.x_off_samples * 1e3
        y_mm = res.y_off_samples * 1e3
        x_mean_mm = res.x_off_mean * 1e3
        y_mean_mm = res.y_off_mean * 1e3
        x_std_mm = res.x_off_std * 1e3
        y_std_mm = res.y_off_std * 1e3

        # ════════════════════════════════════════════════════
        #  Fig 1 : x-y plane CoM offset
        # ════════════════════════════════════════════════════
        fig1, ax1 = plt.subplots(figsize=(7, 7))

        # Geometric centre
        ax1.plot(0, 0, '+', ms=12, mew=1.5, color='gray', zorder=3)

        # All (x_off, y_off) combinations as scatter
        xx, yy = np.meshgrid(x_mm, y_mm)
        ax1.scatter(
            xx.ravel(), yy.ravel(),
            s=25, color='tab:blue', alpha=0.5, zorder=4,
            label=f'Samples ({len(xx.ravel())})',
        )

        # Mean ± std ellipse
        ellipse = Ellipse(
            (x_mean_mm, y_mean_mm),
            width=2 * x_std_mm, height=2 * y_std_mm,
            facecolor='tab:red', alpha=0.15, edgecolor='tab:red',
            linewidth=1.5, linestyle='-', zorder=5,
            label=f'±1$\\sigma$ ({x_std_mm:.2f}×{y_std_mm:.2f} mm)',
        )
        ax1.add_patch(ellipse)

        # Mean marker
        ax1.plot(
            x_mean_mm, y_mean_mm,
            'o', ms=10, color='tab:red', zorder=6,
            label=f'Mean ({x_mean_mm:+.2f}, {y_mean_mm:+.2f}) mm',
        )

        # Annotation
        ax1.annotate(
            f'  ({x_mean_mm:+.2f}, {y_mean_mm:+.2f})',
            (x_mean_mm, y_mean_mm),
            fontsize=12, color='tab:red', zorder=7,
        )

        pad_x = x_std_mm * 2
        pad_y = y_std_mm * 2
        pad = max(pad_x, pad_y)
        ax1.set_xlim(-pad + x_mean_mm, pad + x_mean_mm)
        ax1.set_ylim(-pad + y_mean_mm, pad + y_mean_mm)
        ax1.set_aspect('equal')
        ax1.axhline(0, color='gray', lw=0.5, alpha=0.3)
        ax1.axvline(0, color='gray', lw=0.5, alpha=0.3)
        ax1.set_xlabel('$\\hat{x}_{off}$ [mm]', fontsize=14)
        ax1.set_ylabel('$\\hat{y}_{off}$ [mm]', fontsize=14)
        ax1.set_title('CoM Offset Estimation — x-y Plane', fontsize=18)
        ax1.legend(loc='upper left', fontsize=14)
        ax1.grid(True, alpha=0.2)
        fig1.tight_layout()

        # ════════════════════════════════════════════════════
        #  Fig 2 : Strip plots (mass, y_p, x_off, y_off)
        # ════════════════════════════════════════════════════
        fig2, axes = plt.subplots(1, 4, figsize=(16, 5))

        def _strip_plot(ax, samples, mean, std, ylabel, unit, color):
            n = len(samples)
            location = np.zeros((n,))
            ax.scatter(
                location, samples,
                s=50, color=color, alpha=0.7, zorder=4,
                edgecolors='white', linewidth=0.5,
            )
            ax.axhline(mean, color=color, lw=1.5, ls='-', alpha=0.8,
                        label=f'mean = {mean:.4f}')
            ax.axhspan(
                mean - std, mean + std,
                color=color, alpha=0.12,
                label=f'$\\pm 1 \\sigma$ = {std:.4f}',
            )
            ax.set_xlim(-0.8, 0.8)
            pad_y = std * 3 if std > 0 else abs(mean) * 0.1
            ax.set_ylim(mean - pad_y, mean + pad_y)
            ax.set_xticks([])
            ax.set_ylabel(f'{ylabel} [{unit}]', fontsize=14)
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, axis='y', alpha=0.3)

        # Mass
        _strip_plot(
            axes[0], res.m_from_My, res.m_mean, res.m_std,
            '$\\hat{m}$', 'kg', 'tab:green',
        )
        axes[0].set_title('Mass', fontsize=16)

        # Recovered y_p
        _strip_plot(
            axes[1], res.y_p_samples * 1e3, res.y_p_mean * 1e3, res.y_p_std * 1e3,
            '$\\hat{y}_p$', 'mm', 'tab:purple',
        )
        axes[1].set_title('Recovered $y_p$', fontsize=16)

        # x_off
        _strip_plot(
            axes[2], x_mm, x_mean_mm, x_std_mm,
            '$\\hat{x}_{off}$', 'mm', 'tab:blue',
        )
        axes[2].set_title('$\\hat{x}_{off}$', fontsize=16)

        # y_off
        _strip_plot(
            axes[3], y_mm, y_mean_mm, y_std_mm,
            '$\\hat{y}_{off}$', 'mm', 'tab:orange',
        )
        axes[3].set_title('$\\hat{y}_{off}$', fontsize=16)

        fig2.suptitle('Mass & CoM Offset — Sample Distribution', fontsize=18)
        fig2.tight_layout()

        # ── Save ──
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            p1 = save_dir / 'com_offset_xy_plane.png'
            fig1.savefig(p1, dpi=600, bbox_inches='tight')
            print(f"  Plot → {p1}")

            p2 = save_dir / 'mass_com_strip_plot.png'
            fig2.savefig(p2, dpi=600, bbox_inches='tight')
            print(f"  Plot → {p2}")

        if show:
            plt.show()