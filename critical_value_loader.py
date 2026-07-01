"""
CoM Estimator
=============
Estimate mass and centre-of-mass offset from critical thrust / moment
pairs obtained at tip-over onset.

Physical model (tip-over balance)
---------------------------------
Mx excitation (roll axis, pivot = landing gear in y):
    pos_Mx :  f_col_pos · y_p  =  W · (y_p + y_off) - M_x_pos
    neg_Mx :  f_col_neg · y_p  =  W · (y_p - y_off) + M_x_neg

    →  W  = 0.5 · [ f_col_pos + f_col_neg  -  (M_x_neg - M_x_pos) / y_p ]
    →  y_off from each direction separately

My excitation (pitch axis, pivot = landing gear in x):
    pos_My :  f_col_pos · x_p  =  W · (x_p + x_off) - M_y_pos
    neg_My :  f_col_neg · x_p  =  W · (x_p - x_off) + M_y_neg

    →  W  = 0.5 · [ f_col_pos + f_col_neg  -  (M_y_neg - M_y_pos) / x_p ]
    →  x_off from each direction separately
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.critical_value_loader import CriticalValueLoader


# ═════════════════════════════════════════════════════════════
#  Constants
# ═════════════════════════════════════════════════════════════

G = 9.81              # [m/s²]
X_P_CG = 0.255 / 2.0  # pitch pivot half-span [m]
Y_P_CG = 0.288 / 2.0  # roll  pivot half-span [m]


# ═════════════════════════════════════════════════════════════
#  Result container
# ═════════════════════════════════════════════════════════════

@dataclass
class EstimationResult:
    """Estimation output."""
    # Mass
    m_from_Mx: np.ndarray       # (N_Mx,) per-trial mass from roll pairs
    m_from_My: np.ndarray       # (N_My,) per-trial mass from pitch pairs
    m_all: np.ndarray           # concatenated
    m_mean: float               # [kg]
    m_std: float                # [kg]

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

    Parameters
    ----------
    x_p : float   pitch pivot half-span [m]
    y_p : float   roll  pivot half-span [m]
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
        """Run mass + CoM offset estimation from all available trial pairs."""

        # ── Extract arrays ──
        f_Mx_pos = loader.get_array('Mx', 'pos', 'collective_thrust_N')
        f_Mx_neg = loader.get_array('Mx', 'neg', 'collective_thrust_N')
        M_x_pos  = loader.get_array('Mx', 'pos', 'moment_Nm')
        M_x_neg  = loader.get_array('Mx', 'neg', 'moment_Nm')

        f_My_pos = loader.get_array('My', 'pos', 'collective_thrust_N')
        f_My_neg = loader.get_array('My', 'neg', 'collective_thrust_N')
        M_y_pos  = loader.get_array('My', 'pos', 'moment_Nm')
        M_y_neg  = loader.get_array('My', 'neg', 'moment_Nm')

        # ── Mass estimation ──
        W_from_Mx = 0.5 * (f_Mx_pos + f_Mx_neg
                           - (M_x_neg - M_x_pos) / self.y_p)
        m_from_Mx = W_from_Mx / self.g

        W_from_My = 0.5 * (f_My_pos + f_My_neg
                           - (M_y_neg - M_y_pos) / self.x_p)
        m_from_My = W_from_My / self.g

        m_all = np.concatenate([m_from_Mx, m_from_My])
        m_mean = float(np.mean(m_all))
        m_std = float(np.std(m_all, ddof=1)) if len(m_all) > 1 else 0.0

        W_avg = m_mean * self.g

        # ── CoM offset estimation ──
        # x_off from My (2 estimates per trial: neg & pos)
        x_off_neg = self.x_p * (f_My_neg / W_avg - M_y_neg / (W_avg * self.x_p) - 1.0)
        x_off_pos = self.x_p * (1.0 - f_My_pos / W_avg - M_y_pos / (W_avg * self.x_p))
        x_off_samples = np.empty(2 * len(f_My_pos))
        x_off_samples[0::2] = x_off_neg
        x_off_samples[1::2] = x_off_pos

        # y_off from Mx (2 estimates per trial: neg & pos)
        y_off_neg = self.y_p * (1.0 - f_Mx_neg / W_avg + M_x_neg / (W_avg * self.y_p))
        y_off_pos = self.y_p * (f_Mx_pos / W_avg + M_x_pos / (W_avg * self.y_p) - 1.0)
        y_off_samples = np.empty(2 * len(f_Mx_pos))
        y_off_samples[0::2] = y_off_neg
        y_off_samples[1::2] = y_off_pos

        x_off_mean = float(np.mean(x_off_samples))
        y_off_mean = float(np.mean(y_off_samples))
        x_off_std = float(np.std(x_off_samples, ddof=1)) if len(x_off_samples) > 1 else 0.0
        y_off_std = float(np.std(y_off_samples, ddof=1)) if len(y_off_samples) > 1 else 0.0

        return EstimationResult(
            m_from_Mx=m_from_Mx,
            m_from_My=m_from_My,
            m_all=m_all,
            m_mean=m_mean,
            m_std=m_std,
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
        print("\n" + "=" * 60)
        print("  Mass & CoM Offset Estimation Result")
        print("=" * 60)

        print(f"\n── Mass from Mx trials : {res.m_from_Mx}  [kg]")
        print(f"── Mass from My trials : {res.m_from_My}  [kg]")
        print(f"── Mass (all)          : {res.m_all}  [kg]")
        print(f"   mean = {res.m_mean:.4f} kg,  std = {res.m_std:.4f} kg")

        print(f"\n── x_off samples : {res.x_off_samples * 1e3}  [mm]")
        print(f"   x_off = {res.x_off_mean * 1e3:+.3f} ± {res.x_off_std * 1e3:.3f} mm")

        print(f"\n── y_off samples : {res.y_off_samples * 1e3}  [mm]")
        print(f"   y_off = {res.y_off_mean * 1e3:+.3f} ± {res.y_off_std * 1e3:.3f} mm")
        print()