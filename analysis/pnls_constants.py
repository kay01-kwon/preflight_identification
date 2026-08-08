"""Stage-2 constants of the two-stage calibration.

Regenerate with `python analysis/pnls_centred.py`: stage 1 takes the
per-dataset medians of (C2, K) from the free per-run nonlinear least
squares, stage 2 minimises the ramp-invariance score on a grid around
that centre.  Frozen here so the benchmark and the figures do not have
to repeat the search.
"""

PNLS_CONSTANTS = {
    ('case_01', 'Mx'): (5.2988, 0.1964),
    ('case_01', 'My'): (4.7930, 0.3020),
    ('case_02', 'Mx'): (5.4276, 0.1503),
    ('case_02', 'My'): (5.3997, 0.3375),
    ('case_03', 'Mx'): (3.5177, 0.2884),
    ('case_03', 'My'): (4.0963, 0.3138),
    ('case_04', 'Mx'): (4.3499, 0.2034),
    ('case_04', 'My'): (5.4452, 0.2297),
    ('case_05', 'Mx'): (5.6666, 0.1613),
    ('case_05', 'My'): (5.4868, 0.2294),
}
