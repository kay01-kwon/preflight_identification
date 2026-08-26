#!/usr/bin/env python3
"""Mirror a tree of packed .npz runs as MATLAB .mat files.

The npz packs are the repository's canonical form; this emits a
parallel tree for MATLAB users, byte-for-byte the same arrays in a v5
container (readable by any MATLAB via load()). Variable names cannot
carry '/', so the topic separator becomes '_': odom/t -> odom_t.
Sizes come out within ~4% of the npz -- both deflate the same arrays.

Usage
-----
  python analysis/mat_mirror.py <npz_root> <mat_root>
"""
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    files = sorted(src.rglob('*.npz'))
    if not files:
        raise SystemExit(f'no .npz under {src}')
    tot_in = tot_out = 0
    for p in files:
        rel = p.relative_to(src).with_suffix('.mat')
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        d = dict(np.load(p))
        savemat(out, {k.replace('/', '_'): v for k, v in d.items()},
                do_compression=True)
        tot_in += p.stat().st_size
        tot_out += out.stat().st_size
    print(f'{len(files)} runs   {tot_in/1e6:.1f} MB npz -> '
          f'{tot_out/1e6:.1f} MB mat   ({dst})')


if __name__ == '__main__':
    main()
