"""
Critical Value Extractor
========================
Maximum Log-Likelihood Ratio change-point detection
on angular velocity to identify the onset of body rotation
during moment excitation experiments.

Method
------
Variance-change log-likelihood ratio:

    S(j) = N·log(σ_total²) − j·log(σ_left²) − (N−j)·log(σ_right²)

The change-point j* = argmax S(j) is the moment angular velocity
transitions from zero (ground contact) to non-zero (one leg lifted).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from utils.extractor import BagData, OdometryData, HexaRpmData
from utils import math_tools

# ─── Physical constants ──────────────────────────────────────
C_T_DEFAULT = 1.3175e-7       # N / rpm²
ARM_LENGTH_DEFAULT = 0.265   # m


# ═════════════════════════════════════════════════════════════
#  Result container
# ═════════════════════════════════════════════════════════════

@dataclass
class CriticalValueResult:
    """Extraction result for a single bag."""
    bag_name: str
    axis: str                  # 'x' or 'y'

    # Full time series (odom timeline)
    t: np.ndarray              # (N,)  [s]
    omega: np.ndarray          # (N,)  ω_x or ω_y raw [rad/s]
    f_col: np.ndarray          # (N,)  collective thrust [N]
    moment: np.ndarray         # (N,)  M_x or M_y [N·m]

    # Score (defined only on excitation window)
    score_t: np.ndarray        # (M,)  time of score samples [s]
    score_values: np.ndarray   # (M,)  GLR score

    # Detected onset
    onset_idx: int             # index into t
    onset_time: float          # [s]
    onset_score: float
    onset_thrust: float        # [N]
    onset_moment: float        # [N·m]
    onset_omega: float         # [rad/s]


# ═════════════════════════════════════════════════════════════
#  CriticalValueExtractor
# ═════════════════════════════════════════════════════════════

class CriticalValueExtractor:
    """
    Detect the onset of angular velocity motion using
    variance-change log-likelihood ratio on ω.

    Parameters
    ----------
    C_T            : thrust coefficient [N/rpm²]
    arm_length     : motor arm length [m]
    window_margin  : manual window margin (0 = auto from moment baseline 5σ).
    """

    def __init__(
        self,
        C_T: float = C_T_DEFAULT,
        arm_length: float = ARM_LENGTH_DEFAULT,
        window_margin: int = 0,
    ):
        self._C_T = C_T
        self._arm_length = arm_length
        self._window_margin = window_margin

    # ─────────────────────────────────────────────────────
    #  Internal: prepare time-aligned signals
    # ─────────────────────────────────────────────────────

    @staticmethod
    def _prepare_signals(
        odom: OdometryData,
        rpm: HexaRpmData,
        C_T: float,
        arm_length: float,
        axis: str,
    ) -> dict[str, np.ndarray]:
        """
        Build time-aligned signals on the odom timeline.

        Parameters
        ----------
        odom       : OdometryData with .t, .angular_vel
        rpm        : HexaRpmData  with .t, .rpm
        C_T        : thrust coefficient [N/rpm²]
        arm_length : motor arm length [m]
        axis       : 'x' (roll) or 'y' (pitch)

        Returns
        -------
        dict with keys: 't', 'omega', 'f_col', 'moment'
        All arrays are (N,) on the odom time grid.
        """
        # Reference time = odom.t[0]
        t0 = odom.t[0]
        t = odom.t - t0
        t_rpm = rpm.t - t0

        # Angular velocity: body frame, select axis
        axis_idx = 0 if axis == 'x' else 1  # wx=0, wy=1
        omega = odom.angular_vel[:, axis_idx]

        # Collective thrust on rpm timeline
        f_col_raw = math_tools.collective_thrust_vectorized(C_T, rpm.rpm)

        # Roll/pitch moments on rpm timeline: (N,2) → select axis
        moments_raw = math_tools.rpm_to_moments_vectorized(
            C_T, rpm.rpm, arm_length=arm_length,
        )
        moment_raw = moments_raw[:, axis_idx]  # 0=tau_x, 1=tau_y

        # Interpolate onto odom timeline
        f_col = np.interp(t, t_rpm, f_col_raw)
        moment = np.interp(t, t_rpm, moment_raw)

        return dict(t=t, omega=omega, f_col=f_col, moment=moment)

    # ─────────────────────────────────────────────────────
    #  Internal: excitation window detection
    # ─────────────────────────────────────────────────────

    @staticmethod
    def _detect_excitation_window(
        moment: np.ndarray,
        omega: np.ndarray,
        margin: int = 0,
        threshold: float = 0.002,
        baseline_frac: float = 0.2,
        n_sigma: float = 5.0,
    ) -> tuple[int, int]:
        """
        Find [idx_start, idx_end] of the excitation ramp.

        idx_end = index of |moment| global max

        margin == 0, auto mode:
            idx_start = first index where |M - baseline_mean| > n_sigma * baseline_std

        margin > 0, manual mode:
            idx_start = max(0, idx_end - margin)
        """
        N = len(moment)

        # ── End point: max |moment| ──
        idx_end = int(np.argmax(np.abs(moment)))

        if margin > 0:
            idx_start = max(0, idx_end - margin)
            return idx_start, idx_end

        # Auto: baseline statistics on moment
        n_base = max(int(N * baseline_frac), 20)
        base = moment[:n_base]
        mu = np.mean(base)
        sigma = np.std(base)

        if sigma > 1e-10:
            above = np.where(np.abs(moment - mu) > n_sigma * sigma)[0]
            if len(above) > 0 and above[0] < idx_end:
                return int(above[0]), idx_end

        # Fallback: threshold-based
        above = np.where(np.abs(moment) > threshold)[0]
        if len(above) == 0:
            return 0, N - 1

        idx_start = int(above[0])
        if idx_start >= idx_end:
            idx_start = 0

        return idx_start, idx_end

    # ─────────────────────────────────────────────────────
    #  Internal: variance-change GLR
    # ─────────────────────────────────────────────────────

    @staticmethod
    def _compute_glr_score(
        omega: np.ndarray,
        min_seg: int = 5,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Maximise log-likelihood ratio for variance change:
            S(j) = N·log(σ²_total) − j·log(σ²_left) − (N−j)·log(σ²_right)

        Returns
        -------
        local_indices : (M,) candidate j values (window-local)
        scores        : (M,) corresponding S(j)
        best_j        : local index of onset (argmax S)
        """
        N = len(omega)
        total_var = np.var(omega)

        if total_var < 1e-15 or N < 2 * min_seg:
            return np.array([]), np.array([]), N // 2

        js = np.arange(min_seg, N - min_seg)
        scores = np.empty(len(js), dtype=np.float64)

        cumsum = np.cumsum(omega)
        cumsum2 = np.cumsum(omega ** 2)

        for i, j in enumerate(js):
            mean_l = cumsum[j - 1] / j
            var_l = cumsum2[j - 1] / j - mean_l ** 2
            var_l = max(var_l, 1e-15)

            n_r = N - j
            mean_r = (cumsum[-1] - cumsum[j - 1]) / n_r
            var_r = (cumsum2[-1] - cumsum2[j - 1]) / n_r - mean_r ** 2
            var_r = max(var_r, 1e-15)

            scores[i] = (N * np.log(total_var)
                         - j * np.log(var_l)
                         - n_r * np.log(var_r))

        best_local = js[int(np.argmax(scores))]
        return js, scores, best_local

    # ─────────────────────────────────────────────────────
    #  Public: extract from one bag
    # ─────────────────────────────────────────────────────

    def extract(self, bag: BagData, axis: str) -> CriticalValueResult:
        """
        Detect onset via variance-change GLR on |ω|.

        Pipeline:
          1. Prepare signals (odom ω, rpm → f_col, moment)
          2. Auto excitation window [ramp_start(5σ on M), max|M|]
          3. GLR S(j) on |ω| in window
          4. onset = argmax S(j)

        Parameters
        ----------
        bag  : BagData
        axis : 'x' (roll / ω_x) or 'y' (pitch / ω_y)
        """
        if axis not in ('x', 'y'):
            raise ValueError(f"axis must be 'x' or 'y', got '{axis}'")

        sig = self._prepare_signals(
            bag.odom, bag.rpm, self._C_T, self._arm_length, axis,
        )

        t      = sig['t']
        omega  = sig['omega']
        f_col  = sig['f_col']
        moment = sig['moment']

        detect_omega = omega

        # ── Excitation window ──
        idx_start, idx_end = self._detect_excitation_window(
            moment, detect_omega, margin=self._window_margin,
        )
        win = slice(idx_start, idx_end + 1)

        # ── GLR on |ω| in window ──
        local_js, scores, best_j = self._compute_glr_score(
            np.abs(detect_omega[win]),
        )

        # Map back to global index
        onset_idx = idx_start + best_j

        # Score time axis (global)
        score_t = t[idx_start + local_js] if len(local_js) > 0 else np.array([])
        onset_score = float(scores[int(np.argmax(scores))]) if len(scores) > 0 else 0.0

        return CriticalValueResult(
            bag_name=bag.name,
            axis=axis,
            t=t,
            omega=omega,
            f_col=f_col,
            moment=moment,
            score_t=score_t,
            score_values=scores,
            onset_idx=onset_idx,
            onset_time=float(t[onset_idx]),
            onset_score=onset_score,
            onset_thrust=float(f_col[onset_idx]),
            onset_moment=float(moment[onset_idx]),
            onset_omega=float(omega[onset_idx]),
        )

    # ─────────────────────────────────────────────────────
    #  Public: batch extraction
    # ─────────────────────────────────────────────────────

    def extract_batch(
        self,
        bags: list[BagData],
        axis: str,
    ) -> list[CriticalValueResult]:
        """Run extraction on every bag."""
        results = []
        for bag in bags:
            print(f"  Processing {bag.name} (axis={axis}) ...")
            res = self.extract(bag, axis)
            print(
                f"    onset t={res.onset_time:.4f}s  "
                f"f_col={res.onset_thrust:.3f}N  "
                f"M_{axis}={res.onset_moment:.5f}N·m  "
                f"ω_{axis}={res.onset_omega:.6f}rad/s  "
                f"score={res.onset_score:.2f}"
            )
            results.append(res)
        return results

    # ─────────────────────────────────────────────────────
    #  CSV export
    # ─────────────────────────────────────────────────────

    @staticmethod
    def save_csv(
        result: CriticalValueResult,
        output_dir: Path,
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        parts = result.bag_name.split('_')
        if len(parts) >= 3:
            axis_label = parts[0]
            direction = parts[1]
            trial = int(parts[2])
            csv_name = f"critical_values_{direction}_{axis_label}_0_{trial}.csv"
        else:
            csv_name = f"critical_values_{result.bag_name}.csv"

        csv_path = output_dir / csv_name
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'bag_name', 'axis',
                'onset_time_s', 'onset_idx',
                'collective_thrust_N',
                f'moment_M_{result.axis}_Nm',
                f'omega_{result.axis}_rad_s',
                'glr_score',
            ])
            writer.writerow([
                result.bag_name,
                result.axis,
                f'{result.onset_time:.6f}',
                result.onset_idx,
                f'{result.onset_thrust:.6f}',
                f'{result.onset_moment:.8f}',
                f'{result.onset_omega:.8f}',
                f'{result.onset_score:.6f}',
            ])
        print(f"    Saved → {csv_path}")
        return csv_path

    @staticmethod
    def save_batch_csv(
        results: list[CriticalValueResult],
        output_dir: Path,
    ) -> list[Path]:
        paths = []
        for res in results:
            paths.append(CriticalValueExtractor.save_csv(res, output_dir))
        return paths

    # ─────────────────────────────────────────────────────
    #  Plotting
    # ─────────────────────────────────────────────────────

    @staticmethod
    def plot_results(
        results: list[CriticalValueResult],
        suptitle: str = "",
        save_dir: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        for res in results:
            fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=False)
            t_onset = res.onset_time

            # Row 1: Moment
            ax = axes[0]
            label_M = f'$M_{res.axis}$'
            ax.plot(res.t, res.moment, color='tab:green', linewidth=0.8, label=label_M)
            ax.axvline(t_onset, color='red', ls='--', alpha=0.4)
            ax.plot(t_onset, res.onset_moment, 'o', ms=8, color='red', zorder=5,
                    label=f'onset ({res.onset_moment:.5f} N·m)')
            ax.set_ylabel(f'Moment {label_M} [N·m]')
            ax.legend(loc='upper left', fontsize=8)
            ax.set_title(f'{res.bag_name}  —  Moment Excitation (axis={res.axis})')
            ax.grid(True, alpha=0.3)

            # Row 2: Angular velocity
            ax = axes[1]
            label_w = f'$\\omega_{res.axis}$'
            ax.plot(res.t, res.omega, color='tab:orange', linewidth=0.5,
                    alpha=0.6, label=f'{label_w} raw')
            ax.axvline(t_onset, color='red', ls='--', alpha=0.4)
            ax.plot(t_onset, res.onset_omega, 'o', ms=8, color='red', zorder=5,
                    label=f'onset ({res.onset_omega:.6f} rad/s)')
            ax.set_ylabel(f'Angular Vel {label_w} [rad/s]')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

            # Row 3: Total thrust
            ax = axes[2]
            ax.plot(res.t, res.f_col, color='tab:blue', linewidth=0.8, label='$f_{col}$')
            ax.axvline(t_onset, color='red', ls='--', alpha=0.4)
            ax.plot(t_onset, res.onset_thrust, 'o', ms=8, color='red', zorder=5,
                    label=f'onset ({res.onset_thrust:.3f} N)')
            ax.set_ylabel('Total Thrust [N]')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

            # Row 4: GLR Score
            ax = axes[3]
            if len(res.score_t) > 0:
                ax.plot(res.score_t, res.score_values, color='tab:purple',
                        linewidth=0.8, label='S(j)')
                ax.plot(t_onset, res.onset_score, 'o', ms=8, color='red', zorder=5,
                        label=f'onset ({res.onset_score:.1f})')
            ax.axvline(t_onset, color='red', ls='--', alpha=0.4)
            ax.set_ylabel('GLR Score S(j)')
            ax.set_xlabel('Time [s]')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

            fig.tight_layout()
            if suptitle:
                fig.suptitle(suptitle, fontsize=11, y=1.01)

            if save_dir is not None:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                fig_path = save_dir / f"critical_values_{res.bag_name}.png"
                fig.savefig(fig_path, dpi=600, bbox_inches='tight')
                print(f"    Plot → {fig_path}")

        if show:
            plt.show()