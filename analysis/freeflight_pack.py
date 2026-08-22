#!/usr/bin/env python3
"""Pack free-flight take-off bags into compact arrays that fit in git.

WHY THIS EXISTS.  The excitation campaign already occupies 477 MB of
working tree (140 bags) and 229 MB packed.  The free-flight take-off
campaign cannot be added on top of that: GitHub rejects any single file
over 100 MB outright, caps one push at 2 GB, and starts warning above
1 GB of repository.  Raw ROS 2 bags are the wrong thing to version
anyway -- they carry every topic at full rate, whereas the take-off
metrics of section VIII need six scalars per trial, computed from four
signals over a window of a few seconds.

WHAT IS KEPT.  Everything the published metrics are computed from, and
nothing else:

    odom : t, position, quaternion, linear_vel, angular_vel
    pose : t, position, quaternion        (mocap, for drift truth)
    rpm  : t, rpm                         (thrust reconstruction)
    imu  : t, angular_vel, linear_acc     (optional, --imu)

so that peak |phi|/|theta|, peak |w_x|/|w_y|, the drift components
d_x/d_y and the peak linear velocity all remain recomputable from the
committed arrays.  Nothing here is a summary: the time series survive,
only the container changes.

WHAT IS DROPPED.  Message headers, frame ids, covariances, per-motor
acceleration, and every topic not listed above.  float32 is used by
default -- the odometry is EKF2 output at ~1e-3 m and ~1e-3 rad, some
four orders above float32 resolution, so the cast is lossless at the
level anything is reported.  Pass --float64 to keep the exact bits.

TYPICAL RESULT.  A 10 s take-off at 100 Hz packs to roughly 100 kB
compressed against several MB of bag, i.e. 30-50x.  A campaign of a
few hundred trials lands in tens of MB and is comfortable to commit.

RAW BAGS.  Keep them, but out of git -- an archive with a DOI (Zenodo
accepts 50 GB per record) is what a Data Availability statement should
point at.  This script is the bridge: the repository stays clonable and
the analysis stays reproducible from what it contains.

Usage
-----
  python analysis/freeflight_pack.py <bag_dir> <out_dir> [options]

    --window T0 T1   keep only t_rel in [T0, T1] seconds (odom t[0] = 0)
    --imu            include the raw IMU topic (roughly doubles size)
    --float64        do not downcast
    --dry-run        report sizes without writing

Example
-------
  python analysis/freeflight_pack.py \
      ~/bags/freeflight DataSet/freeflight --window -1 6
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.extractor import RosBagExtractor          # noqa: E402
from utils.extractor import (OdometryData, PoseData,  # noqa: E402
                             HexaRpmData, ImuData)

# topic role -> the fields worth keeping, in the order they are stored
FIELDS = {
    'odom': ('position', 'quaternion', 'linear_vel', 'angular_vel'),
    'pose': ('position', 'quaternion'),
    'rpm':  ('rpm',),
    'imu':  ('angular_vel', 'linear_acc'),
}
KIND = {OdometryData: 'odom', PoseData: 'pose',
        HexaRpmData: 'rpm', ImuData: 'imu'}


def _slice(t, window):
    """Index mask for a t_rel window, or all-true when no window given."""
    if window is None:
        return np.ones(t.shape, dtype=bool)
    return (t >= window[0]) & (t <= window[1])


def pack_bag(bag_dir, want_imu, dtype, window):
    """One bag -> {array name: ndarray}, ready for np.savez_compressed."""
    ext = RosBagExtractor(bag_dir)
    loaded = ext.load_all()

    out, t0 = {}, None
    # the odometry start is the time reference, exactly as BagData.t0
    for data in loaded.values():
        if isinstance(data, OdometryData):
            t0 = float(data.t[0])
            break
    if t0 is None:
        raise RuntimeError(f'{bag_dir.name}: no odometry topic, cannot '
                           f'establish the time reference')

    for topic, data in loaded.items():
        kind = KIND.get(type(data))
        if kind is None or (kind == 'imu' and not want_imu):
            continue
        t = np.asarray(data.t, dtype=np.float64) - t0
        keep = _slice(t, window)
        if not keep.any():
            continue
        # time stays float64: it is a relative second count, and the
        # metrics window is defined on it
        out[f'{kind}/t'] = t[keep]
        for f in FIELDS[kind]:
            out[f'{kind}/{f}'] = np.asarray(
                getattr(data, f), dtype=dtype)[keep]
    if not out:
        raise RuntimeError(f'{bag_dir.name}: nothing matched the window')
    out['t0'] = np.array([t0], dtype=np.float64)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('bag_dir', type=Path, help='directory of bag folders')
    p.add_argument('out_dir', type=Path, help='where the .npz files go')
    p.add_argument('--window', type=float, nargs=2, metavar=('T0', 'T1'),
                   default=None, help='keep t_rel in [T0, T1] seconds')
    p.add_argument('--imu', action='store_true', help='include raw IMU')
    p.add_argument('--float64', action='store_true', help='do not downcast')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()

    dtype = np.float64 if a.float64 else np.float32
    bags = sorted(d for d in a.bag_dir.iterdir()
                  if d.is_dir() and (d / 'metadata.yaml').exists())
    if not bags:
        p.error(f'no bag folders (with metadata.yaml) under {a.bag_dir}')

    if not a.dry_run:
        a.out_dir.mkdir(parents=True, exist_ok=True)

    src_tot = dst_tot = 0
    print(f'{"bag":<34}{"bag [MB]":>10}{"npz [MB]":>10}{"ratio":>8}')
    for b in bags:
        src = sum(f.stat().st_size for f in b.rglob('*') if f.is_file())
        try:
            arrays = pack_bag(b, a.imu, dtype, a.window)
        except Exception as exc:                      # keep the sweep going
            print(f'{b.name:<34}{"":>10}{"":>10}   SKIPPED  {exc}')
            continue
        dst_path = a.out_dir / f'{b.name}.npz'
        if a.dry_run:
            # measure by packing to memory rather than guessing
            import io
            buf = io.BytesIO()
            np.savez_compressed(buf, **arrays)
            dst = buf.getbuffer().nbytes
        else:
            np.savez_compressed(dst_path, **arrays)
            dst = dst_path.stat().st_size
        src_tot, dst_tot = src_tot + src, dst_tot + dst
        print(f'{b.name:<34}{src/1e6:>10.1f}{dst/1e6:>10.3f}'
              f'{src/max(dst, 1):>8.1f}x')

    print(f'\n{len(bags)} bags   {src_tot/1e6:.0f} MB -> '
          f'{dst_tot/1e6:.1f} MB   ({src_tot/max(dst_tot, 1):.0f}x)')
    if a.dry_run:
        print('dry run: nothing written')
    else:
        print(f'written to {a.out_dir}')
    if dst_tot > 200e6:
        print('WARNING: still large for git; narrow --window or drop --imu')


if __name__ == '__main__':
    main()
