"""
Critical Value CSV Loader
=========================
Load critical_values_*.csv files produced by CriticalValueExtractor,
classify by axis (Mx / My) and direction (pos / neg).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ═════════════════════════════════════════════════════════════
#  Single CSV row
# ═════════════════════════════════════════════════════════════

@dataclass
class OnsetRecord:
    """One parsed CSV row."""
    bag_name: str
    axis: str              # 'x' or 'y'
    direction: str         # 'pos' or 'neg'
    axis_label: str        # 'Mx' or 'My'
    trial: int

    onset_time_s: float
    onset_idx: int
    collective_thrust_N: float
    moment_Nm: float
    omega_rad_s: float
    glr_score: float


# ═════════════════════════════════════════════════════════════
#  Filename parser
# ═════════════════════════════════════════════════════════════

# critical_values_pos_Mx_0_1.csv  →  (pos, Mx, 1)
_FNAME_RE = re.compile(
    r"critical_values_"
    r"(?P<direction>pos|neg)_"
    r"(?P<axis_label>M[xy])_"
    r"0_(?P<trial>\d+)\.csv$"
)


def _parse_filename(name: str) -> Optional[dict]:
    m = _FNAME_RE.match(name)
    if m is None:
        return None
    return dict(
        direction=m.group('direction'),
        axis_label=m.group('axis_label'),
        trial=int(m.group('trial')),
    )


# ═════════════════════════════════════════════════════════════
#  CriticalValueLoader
# ═════════════════════════════════════════════════════════════

class CriticalValueLoader:
    """
    Scan directories for critical_values_*.csv,
    parse and classify by axis / direction / trial.

    Parameters
    ----------
    csv_dir   : path or list of paths containing CSV files
    recursive : search subdirectories recursively
    """

    def __init__(
        self,
        csv_dir: str | Path | list[str | Path],
        recursive: bool = True,
    ):
        if isinstance(csv_dir, (str, Path)):
            csv_dir = [csv_dir]

        self._records: list[OnsetRecord] = []

        for d in csv_dir:
            d = Path(d)
            pattern = "**/" if recursive else ""
            for f in sorted(d.glob(f"{pattern}critical_values_*.csv")):
                rec = self._load_csv(f)
                if rec is not None:
                    self._records.append(rec)

        self._by_axis: dict[str, dict[str, list[OnsetRecord]]] = {
            'Mx': {'pos': [], 'neg': []},
            'My': {'pos': [], 'neg': []},
        }
        for r in self._records:
            self._by_axis[r.axis_label][r.direction].append(r)

        for axis_dict in self._by_axis.values():
            for lst in axis_dict.values():
                lst.sort(key=lambda r: r.trial)

    # ── CSV parsing ───────────────────────────────────────

    @staticmethod
    def _load_csv(csv_path: Path) -> Optional[OnsetRecord]:
        meta = _parse_filename(csv_path.name)
        if meta is None:
            return None

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return None

        row = rows[0]
        axis_char = meta['axis_label'][1]
        moment_col = f'moment_M_{axis_char}_Nm'
        omega_col = f'omega_{axis_char}_rad_s'

        return OnsetRecord(
            bag_name=row['bag_name'],
            axis=row['axis'],
            direction=meta['direction'],
            axis_label=meta['axis_label'],
            trial=meta['trial'],
            onset_time_s=float(row['onset_time_s']),
            onset_idx=int(row['onset_idx']),
            collective_thrust_N=float(row['collective_thrust_N']),
            moment_Nm=float(row[moment_col]),
            omega_rad_s=float(row[omega_col]),
            glr_score=float(row['glr_score']),
        )

    # ── Accessors ─────────────────────────────────────────

    @property
    def all_records(self) -> list[OnsetRecord]:
        return list(self._records)

    @property
    def Mx_pos(self) -> list[OnsetRecord]:
        return self._by_axis['Mx']['pos']

    @property
    def Mx_neg(self) -> list[OnsetRecord]:
        return self._by_axis['Mx']['neg']

    @property
    def My_pos(self) -> list[OnsetRecord]:
        return self._by_axis['My']['pos']

    @property
    def My_neg(self) -> list[OnsetRecord]:
        return self._by_axis['My']['neg']

    def get_array(
        self,
        axis_label: str,
        direction: str,
        field: str,
    ) -> np.ndarray:
        """
        Extract 1-D numpy array for (axis, direction, field).

        field : OnsetRecord attribute name, e.g.
                'collective_thrust_N', 'moment_Nm', 'omega_rad_s',
                'onset_time_s', 'glr_score', 'trial'
        """
        records = self._by_axis[axis_label][direction]
        return np.array([getattr(r, field) for r in records])

    # ── Summary ───────────────────────────────────────────

    def print_summary(self) -> None:
        for axis_label in ('Mx', 'My'):
            for direction in ('pos', 'neg'):
                records = self._by_axis[axis_label][direction]
                if not records:
                    continue
                print(f"\n── {axis_label} / {direction}  ({len(records)} trials) ──")
                print(f"  {'Trial':>5}  {'f_col [N]':>12}  {'Moment [N·m]':>14}  "
                      f"{'ω [rad/s]':>12}  {'t_onset [s]':>12}")
                print("  " + "-" * 65)
                for r in records:
                    print(f"  {r.trial:>5d}  {r.collective_thrust_N:>12.4f}  "
                          f"{r.moment_Nm:>14.8f}  {r.omega_rad_s:>12.8f}  "
                          f"{r.onset_time_s:>12.4f}")