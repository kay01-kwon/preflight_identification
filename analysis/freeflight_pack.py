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
    rpm  : t, rpm, acc                    (thrust reconstruction)
    cmd  : t, cmd                         (what the allocator asked for)
    imu  : t, angular_vel, linear_acc     (optional, --imu)

so that peak |phi|/|theta|, peak |w_x|/|w_y|, the drift components
d_x/d_y and the peak linear velocity all remain recomputable from the
committed arrays.  Nothing here is a summary: the time series survive,
only the container changes.

The raw command is kept even though no published metric reads it: the
rotor-lag ablation compares the moment that was commanded against the
moment the rotors delivered, and without cmd only one half of that pair
survives packing.  It costs about 0.02 MB per excitation run, an order
below the rotor speeds it is paired with.

WHAT IS DROPPED.  Message headers, frame ids, covariances, and every
topic not listed above.  float32 is used by
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

CHOOSING THE CONTAINER.  npz and MATLAB v5 are within 4% of each other
(0.412 vs 0.397 MB on pos_Mx_03): both deflate the same float32 arrays,
so the container is not where anything is saved.  Pick by workflow --
npz for the Python analysis here, mat to open the trials in MATLAB;
scipy.io reads both either way.  The window is the lever that matters:
trimming a 51 s record to 8 s takes 0.412 MB to 0.065 MB.

Usage
-----
  python analysis/freeflight_pack.py <bag_dir> <out_dir> [options]

    --window T0 T1   keep only t_rel in [T0, T1] seconds (odom t[0] = 0)
    --format {npz,mat}   container; size is equivalent, see above
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
                             HexaRpmData, HexaCmdData, ImuData)

# topic role -> the fields worth keeping, in the order they are stored
FIELDS = {
    'odom': ('position', 'quaternion', 'linear_vel', 'angular_vel'),
    'pose': ('position', 'quaternion'),
    'rpm':  ('rpm', 'acc'),
    'cmd':  ('cmd',),
    'imu':  ('angular_vel', 'linear_acc'),
}
KIND = {OdometryData: 'odom', PoseData: 'pose', HexaRpmData: 'rpm',
        HexaCmdData: 'cmd', ImuData: 'imu'}


C_T = 1.3175e-7            # N/rpm^2, table 9 (load-cell identified)
G = 9.81


def probe_bag(bag_dir, mass):
    """Where the lift-off sits inside a record, so a window can be set.

    A window is measured from the odometry start, but the take-off is
    not: the vehicle is armed, idles, and climbs when commanded. Packing
    a window that lands before lift-off would keep the quiet part and
    discard the transient the metrics are computed from, at a
    compression ratio that looks flattering precisely because the
    interesting samples were dropped. Reconstruct the collective thrust
    from the rotor speeds as in (1) and report the first crossing of the
    nominal weight -- the same t_lo the paper defines.
    """
    ext = RosBagExtractor(bag_dir)
    loaded = ext.load_all()
    odom = next((d for d in loaded.values()
                 if isinstance(d, OdometryData)), None)
    rpm = next((d for d in loaded.values()
                if isinstance(d, HexaRpmData)), None)
    if odom is None:
        return None
    t0 = float(odom.t[0])
    span = float(odom.t[-1]) - t0
    t_lo = None
    if rpm is not None and len(rpm.t):
        f = C_T * np.sum(np.asarray(rpm.rpm, dtype=np.float64) ** 2, axis=1)
        hit = np.flatnonzero(f >= mass * G)
        if hit.size:
            t_lo = float(rpm.t[hit[0]]) - t0
    return span, t_lo


def find_bags(root):
    """Every bag directory at or under *root*, at any depth.

    A campaign is usually filed by condition -- case / controller /
    compensation variant -- so the bags sit several levels down and the
    depth differs between branches.  Recognise a bag by its
    metadata.yaml wherever it is, and accept *root* itself being one.
    """
    root = Path(root)
    if (root / 'metadata.yaml').exists():
        return [root]
    return [d for d in root.rglob('*')
            if d.is_dir() and (d / 'metadata.yaml').exists()]


def _write(arrays, dest, fmt):
    """Serialise to *dest* (a path or a file object) in *fmt*."""
    if fmt == 'npz':
        np.savez_compressed(dest, **arrays)
    else:
        # MATLAB v5 variable names cannot carry '/', so the topic
        # separator becomes '_': odom/t -> odom_t
        from scipy.io import savemat
        savemat(dest, {k.replace('/', '_'): v for k, v in arrays.items()},
                do_compression=True)


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
    p.add_argument('--format', choices=('npz', 'mat'), default='npz',
                   help='container (sizes are equivalent; pick by workflow)')
    p.add_argument('--imu', action='store_true', help='include raw IMU')
    p.add_argument('--float64', action='store_true', help='do not downcast')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--probe', action='store_true',
                   help='report each record span and its lift-off time, so '
                        '--window can be checked against the take-off')
    p.add_argument('--mass', type=float, default=3.066,
                   help='nominal mass [kg] for the lift-off probe')
    a = p.parse_args()

    dtype = np.float64 if a.float64 else np.float32
    bags = sorted(find_bags(a.bag_dir))
    if not bags:
        sub = sorted(d.name for d in a.bag_dir.iterdir() if d.is_dir())[:8] \
            if a.bag_dir.is_dir() else []
        p.error(f'no bag folders (a directory holding metadata.yaml) at or '
                f'under {a.bag_dir}'
                + (f'; it contains {", ".join(sub)}' if sub else ''))

    if a.probe:
        root = a.bag_dir.parent if bags == [a.bag_dir] else a.bag_dir
        w = max(34, min(60, max(len(str(b.relative_to(root)))
                                for b in bags)))
        print(f'{"bag":<{w}}{"span [s]":>10}{"lift-off [s]":>14}')
        lows, missing = [], 0
        for b in sorted(bags):
            r = probe_bag(b, a.mass)
            if r is None:
                continue
            span, t_lo = r
            if t_lo is None:
                missing += 1
                print(f'{str(b.relative_to(root)):<{w}}{span:>10.1f}'
                      f'{"never":>14}')
            else:
                lows.append(t_lo)
                print(f'{str(b.relative_to(root)):<{w}}{span:>10.1f}'
                      f'{t_lo:>14.2f}')
        if lows:
            lo, hi = min(lows), max(lows)
            print(f'\nlift-off spans {lo:.2f} to {hi:.2f} s after the '
                  f'odometry start')
            print(f'a window covering every trial: '
                  f'--window {np.floor(lo) - 1:.0f} {np.ceil(hi) + 8:.0f}')
        if missing:
            print(f'{missing} bags never reach {a.mass * G:.1f} N; check '
                  f'--mass, or that the rotor-speed topic was recorded')
        return

    if not a.dry_run:
        a.out_dir.mkdir(parents=True, exist_ok=True)

    # the campaign's directory structure (case / controller / variant)
    # is the labelling, so mirror it in the output rather than
    # flattening to bag names that would collide across branches
    root = a.bag_dir.parent if len(bags) == 1 and bags[0] == a.bag_dir \
        else a.bag_dir
    width = max(34, min(60, max(len(str(b.relative_to(root))) for b in bags)))

    src_tot = dst_tot = 0
    print(f'{"bag":<{width}}{"bag [MB]":>10}'
          f'{a.format + " [MB]":>10}{"ratio":>8}')
    for b in bags:
        rel = b.relative_to(root)
        src = sum(f.stat().st_size for f in b.rglob('*') if f.is_file())
        try:
            arrays = pack_bag(b, a.imu, dtype, a.window)
        except Exception as exc:                      # keep the sweep going
            print(f'{str(rel):<{width}}{"":>10}{"":>10}   SKIPPED  {exc}')
            continue
        dst_path = a.out_dir / rel.with_suffix(f'.{a.format}')
        if not a.dry_run:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
        if a.dry_run:
            # measure by packing to memory rather than guessing
            import io
            buf = io.BytesIO()
            _write(arrays, buf, a.format)
            dst = buf.getbuffer().nbytes
        else:
            _write(arrays, dst_path, a.format)
            dst = dst_path.stat().st_size
        src_tot, dst_tot = src_tot + src, dst_tot + dst
        print(f'{str(rel):<{width}}{src/1e6:>10.1f}{dst/1e6:>10.3f}'
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
