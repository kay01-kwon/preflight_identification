# preflight_identification

Pre-flight CoM / moment-offset identification for a multirotor, from a **ramp
moment excitation performed on the ground**, on the vehicle's own landing gear
— no rig, no mocap in the loop, no stable in-flight data.

Ramping the commanded moment tilts the vehicle until it starts to rotate about
its landing-gear contact line. The moment at that **rotation onset** is the
critical moment `M_crit`; comparing the two tip directions yields the CoM
offset and the feed-forward moment offset used to compensate take-off.

---

## Method

Past the balance point the tip-over is **unstable** (gravity feeds back
positively), so the linearised dynamics

```
J_P·φ̈ = Ṁ·τ + W·z_CoM·φ        ⇒        φ̈ − d·φ = Ṁ·τ/J_P ,   d = W·z_CoM/J_P > 0
```

have real eigenvalues `±√d` and a **hyperbolic** closed-form solution. With the
onset conditions ω(t_crit) = 0 and α(t_crit) = 0 (the net moment vanishes when
`M = M_crit`) it collapses to

```
ω(τ) = C₁·(cosh(C₂·τ) − 1) + C ,    τ = t − t_crit ,   C₂ = √d
```

**No shape parameter is free.** Since `J_P·d = W·z_CoM`, the amplitude is fixed
by the *measured* ramp rate:

```
C₁ = Ṁ / (W·z_CoM) = K·Ṁ
```

so with `C₂` and `K` pinned as rig constants, `Ṁ` measured per run and ω
continuous at the onset (`C` = pre-onset baseline), the only quantity searched
per run is the **onset index** — swept exhaustively over the excitation window.
There is no optimiser, no initial guess and no seed model in this path.

The rig constants `(C₂, K)` are estimated once per dataset by *physical
self-consistency*: `M_crit` is a static threshold, so it must not depend on how
fast the moment was ramped. See `estimate_rig_constants()`.

> The time-quadratic model (`--model piecewise`) is the `d = 0` limit of the
> same solution — it drops the gravity term that creates the instability — and
> its truncation error over the window actually used is 14–75%. It is kept only
> as a comparison baseline, not for identification.

---

## Install

```bash
pip install numpy scipy matplotlib
pip install ruptures        # only for the change-point benchmark
```

Reading the bags needs a sourced **ROS 2** environment (`rosbag2_py`, `rclpy`)
and the `ros2_libcanard_msgs` package for `/uav/actual_rpm`.

---

## Data layout

```
DataSet/exp/<case>/<Mx|My>/<neg|pos>_<Mx|My>_<rate>/   ← one rosbag2 folder per run
```

`<rate>` is `01`/`02`/`03` for the slow runs, or the commanded ramp rate in
centi-units: `045` = 0.45, `065` = 0.65, `090` = 0.90, `120` = 1.20 N·m/s.

Topics used: `/mavros/local_position/odom`, `/S550/pose`, `/uav/actual_rpm`,
and optionally `/mavros/imu/data_raw`.

---

## Usage

### Identification

```bash
python critical_value_getter_piecewise.py DataSet/exp/case_05/My
```

The defaults are the reported method (`--model cosh`, `--omega-source odom`),
so no options are needed for a normal run.

```bash
# axis explicit, save figures, no interactive window (batch use)
python critical_value_getter_piecewise.py DataSet/exp/case_05/My \
    --axis y --save-fig --no-plot

# 95% confidence intervals
python critical_value_getter_piecewise.py DataSet/exp/case_05/My --ci

# known mass
python critical_value_getter_piecewise.py DataSet/exp/case_05/My --mass 3.22

# results elsewhere
python critical_value_getter_piecewise.py DataSet/exp/case_05/My \
    --output-dir results/case_05_My --save-fig --no-plot

# raw IMU as the rate source (200 Hz, propeller vibration → low-pass it)
python critical_value_getter_piecewise.py DataSet/exp/case_05/My \
    --omega-source imu --lpf-cutoff 15
```

| Option | Default | Meaning |
|---|---|---|
| `data_dir` | required | directory holding the run folders |
| `--axis {x,y}` | auto | `x` = roll (Mx), `y` = pitch (My) |
| `--model {cosh,piecewise}` | `cosh` | identification model; `piecewise` is a baseline only |
| `--omega-source {odom,imu}` | `odom` | `imu` = `/mavros/imu/data_raw` |
| `--lpf-cutoff` | off | Butterworth cutoff [Hz]; use ~15 with `--omega-source imu` |
| `--lpf-order` | 4 | Butterworth order |
| `--mass` | — | known mass [kg] |
| `--ci` | off | 95% CI (analytic for the moment offset, bootstrap for CoM / mass) |
| `--output-dir` | `data_dir` | where CSVs and figures are written |
| `--save-fig` / `--no-plot` | off | write figures / suppress the interactive window |

`--robust`, `--robust-sides`, `--huber-k` apply to `--model piecewise` only;
the closed form has no free shape parameter and needs no robustification.

**Outputs** (in `--output-dir`): `com_estimation_summary_<axis>.csv`,
`com_estimation_pairs_<axis>.csv`, `com_estimation_combs_<axis>.csv`,
`com_estimation_result_<axis>.csv`, plus figures with `--save-fig`.

**Run-level quality gates.** Before the rig constants are estimated, the batch
pipeline excludes runs whose executed ramp fails any of three onset-free
criteria, all evaluated on the measured moment trace alone (see
`extract_piecewise_batch`):

| Gate | Threshold | Rationale |
|---|---|---|
| slope error \|ε\| | ≤ 3 % | evaluated on the \|M\| ≥ floor segment (0.7 N·m roll / 0.4 N·m pitch = \|M_crit\| lower bound over the ±20 mm CoM-offset box); a 3 % Ṁ error perturbs C₁ = K·Ṁ by 3 % |
| window samples N_full | ≥ 38 | necessary condition for any onset position: 30 pre-onset (baseline + sweep minimum) + 8 post-onset samples |
| linearity RMSE | ≤ 30 mN·m | fault detector for aborted/stepped ramps: ~10× the execution noise floor, above the healthy-run maximum |

Pass `ramp_gate_pct=None`, `n_full_min=None` or `lin_rmse_max=None` to disable
individually. With the floored slope gate all 140 reference runs pass; scored
on the full window instead, the slope gate would reject 8 fast runs on their
early spin-up transients.

### Ramp-execution quality

Grades every run's executed ramp against its commanded rate — the same window
and least-squares slope the identification consumes for C₁ = K·Ṁ, so the
metric scores exactly what the model depends on. Signal-level only (no onset
model), hence fast and free of circularity:

```bash
python analysis/ramp_quality.py DataSet/exp --save-fig      # all cases, both axes
python analysis/ramp_quality.py DataSet/exp --cases case_05 # one case
python analysis/ramp_quality.py DataSet/exp/case_05/My      # single directory
```

| Option | Default | Meaning |
|---|---|---|
| `--gate` | 3 | run-level \|slope error\| gate [%] |
| `--nmin` | 38 | minimum realized window sample count N_full |
| `--linmax` | 0.030 | maximum linearity RMSE [N·m] (fault detector) |
| `--cases` | all | restrict to specific case names |
| `--save-fig` | off | two-panel figure: ε and linearity RMSE vs rate |

Writes `ramp_quality_runs.csv` (per run), `ramp_quality_by_rate.csv`
(aggregate over passing runs), `ramp_quality_excluded.csv` and, with
`--save-fig`, `ramp_quality_vs_rate.png`.

### Sensitivity to the rig constants

Quantifies how much the deliverables (M_crit, M_ff, pivot-based CoM offset,
mass) move when the rig constants (C₂, K) move — one-at-a-time scaling around
the reference plus a full 2-D grid over the physically admissible box:

```bash
python analysis/sensitivity.py DataSet/exp/case_05/My --axis y --save-fig
python analysis/sensitivity.py DataSet/exp/case_05/My --axis y \
    --truth-com -10.89 --save-fig        # adds a |CoM − truth| error heatmap
python analysis/sensitivity.py DataSet/exp/case_05/My --axis y --c2 6.4 --k 0.19
```

| Option | Default | Meaning |
|---|---|---|
| `--axis {x,y}` | required | excitation axis |
| `--c2`, `--k` | estimated | pin the reference constants instead of estimating |
| `--scales` | 0.8…1.2 | one-at-a-time scale factors |
| `--c2-grid`, `--k-grid` | 3–8 / 0.1–0.5 | MIN MAX STEP of the 2-D grid |
| `--truth-com` | — | ground-truth CoM [mm]; adds the error heatmap panel |
| `--stride` | 2 | onset sweep stride (speed/accuracy trade-off) |

Writes `sensitivity_oat_<axis>.csv`, `sensitivity_grid_<axis>.csv` and, with
`--save-fig`, `sensitivity_grid_<axis>.png` (M_ff and CoM maps over (C₂, K),
plus the error map when `--truth-com` is given). On the reference dataset a
±20 % mis-specification of either constant moves the CoM by < 0.9 mm, and the
constant choice moves the ground-truth error by < 0.9 mm over the whole box.

### CoM height (`z_CoM`) from the resting tilt

Asks whether the ground data can measure `z_CoM` — and whether the ground-effect
moment can be told apart from it — given the load-cell truth for `W` and the CoM
offset. Carrying the absolute resting tilt `φ₀` through the pivot balance,

```
M_crit = sgn·(W cos φ₀ − f)·l + s_ax·W·λ·cos φ₀ − W·z_CoM·sin φ₀ + ΔM_GE
```

the `z_CoM` term is **symmetric** between the two tip directions: a tilted
vehicle simply has its CoM displaced by `z_CoM sin φ₀`, so it is algebraically
indistinguishable from a CoM offset and it survives `M_ff`. Per group that is
one equation in which `z_CoM`, the landing-gear asymmetry `(l₊ − l₋)` and the
symmetric part of the ground effect all enter as constants — knowing `W` and `λ`
does not break that tie. Only a *varying* regressor does, and `sin φ₀` is the one
`z_CoM` owns:

```bash
python analysis/zcom_tilt.py --collect DataSet/exp --output-dir .   # 140 runs
python analysis/zcom_tilt.py --table zcom_tilt_runs.csv            # re-analyse
python analysis/zcom_tilt.py --table zcom_tilt_runs.csv --inject 0.30
python analysis/zcom_tilt.py --table zcom_tilt_runs.csv --dynamic DataSet/exp
python analysis/zcom_tilt.py --table zcom_tilt_runs.csv --budget DataSet/exp
```

| Estimator | What it uses | Result |
|---|---|---|
| A within-group | run-to-run variation of `φ₀`, with case×axis×direction fixed effects — **truth-free**, and GE-proof (the GE moment is constant within a group, so the fixed effect absorbs it) | `z_CoM = +1 ± 27` mm (mocap attitude, errors-in-variables corrected); `−32 ± 38` mm (odom) |
| B between-group | offset error of `M_ff` against the load-cell truth, on `−s_ax sin φ₀` | `z_CoM = −23 ± 48` mm |
| C rig constant | `W z_CoM = 1/K` from the ramp-invariance calibration | 61–554 mm over the ten case–axis groups — the `(C₂, K)` ridge, a 9× spread |
| E truth-pinned dynamics | the truth pins `½(M₊ + M₋)` exactly, so the onset needs no sweep; then a 2-parameter `ω = C₁(cosh C₂τ − 1)` fit per run | `C₂` runs to a bound on 49% of the 134 fitted runs, `z_CoM` median 1 mm with IQR `[0, 166]` and range up to 12.6 m — only `C₁C₂² = Ṁ/J_P` is determined (`J_P` median `0.123` kg·m², IQR `[0.074, 0.267]`) |

A and B agree on a null: the identified threshold does not respond to the
resting tilt the way an `O(0.1–0.3 m)` CoM height demands. The estimators are
not blind — `--inject 0.30` puts a synthetic `z_CoM = 300` mm into every
`M_crit` and is recovered as `299 ± 42` mm (A) and `277 ± 48` mm (B) — and the
orthogonal-axis placebo returns `+9 ± 14` mm. The limitation is the **lever**:
the resting tilt is `+0.5°` (roll) / `−1.5°` (pitch) on every run and varies by
only `0.21°` within a group, and the odom/mocap cross-check attributes just
`44%` of the odom spread to a real attitude change (the rest is
attitude-estimate noise, which attenuates the slope). Read the result as a
bound rather than a measurement: `|z_CoM| ≲ 55` mm at 95%, in tension with the
`0.3 m` used a priori — which is why the resting attitude belongs in the error
budget, and why a deliberate `±5°` wedge is the fix (same per-run residual and
run count → `SE(z_CoM) ≈ 1` mm).

**The ground-effect residual is not separable here.** Within a group `ΔM_GE` is
a constant (same collective, same tilt) sharing a fixed effect with the arm, the
offset and the gear asymmetry. Its only structured channels are `sgn·c_a f l`,
antisymmetric and hence degenerate with the pivot arm / contact lever, and
`b·M`, a 1–4% scale on `M_crit` — 3–14 mN·m against a 29 mN·m within-group
residual and a 52 mN·m group-to-group scatter. The static data bound the
combined ground-contact budget; they cannot attribute it
(`analysis/static_attribution.py`).

Reading it off the post-onset balance instead does not work either, and
`--budget` says why: `ΔM_res = J_P α − M − f l + W(l + λ) cos δφ − W z_CoM sin δφ`
leaves `J_P` and `W z_CoM` as the only unmeasured coefficients, and each is
multiplied by a large measured signal — `|δφ| = 5.95°` at the window edge,
`|α| = 2.24` rad/s² at the moment peak (medians). A 2 mN·m readout therefore
needs `z_CoM` to `±0.6` mm and `J_P` to `0.7%`; what the dataset supplies
(`±27` mm and an IQR half-width of `0.097` kg·m²) propagates to `88` and `216`
mN·m — which is why the GE moment is bounded by forward modelling
(`analysis/ge_trajectory.py`) rather than read off the balance.

### GE pinned by the model: are `J_P` and `z_CoM` then consistent?

The reverse of the section above — instead of leaving the ground effect as an
unknown residual, pin it with the parameter-free rotor-interference model and
ask what the data then say about the two mechanical constants:

```bash
python analysis/ge_pinned_consistency.py --table zcom_tilt_runs.csv --verbose
```

**Static.** With GE pinned, the onset balance leaves only the CoM height and
the contact-lever offset `dl`, and they sit in *orthogonal* channels: `dl` and
the GE thrust term `sgn·c_a f l` are antisymmetric between the tip directions,
`z_CoM sin φ₀` is symmetric. So the GE model can only move `dl`:

| GE model | `dl` roll [mm] | `dl` pitch [mm] | `z_CoM` [mm] | resid |
|---|---|---|---|---|
| none | `17.5 ± 2.6` | `20.7 ± 2.6` | `−39 ± 29` | 78 mN·m |
| single (Cheeseman) | `13.3 ± 2.5` | `17.4 ± 2.5` | `−35 ± 28` | 76 mN·m |
| interference | `−0.1 ± 2.5` | `6.8 ± 2.5` | `−23 ± 28` | 71 mN·m |

The interference model absorbs the entire deficit that the rigid fit otherwise
has to buy with a 17–21 mm inboard contact shift — consistency, not
attribution, since the two stay degenerate in that channel. `z_CoM` moves by
only 16 mm across the whole GE bracket, inside its own ±28 mm standard error:
**pinning GE cannot inform the CoM height.**

**Dynamic.** The parallel-axis theorem is a free, GE-independent and
truth-independent constraint on every calibrated pair: `J_P = J_cm + m(l² +
z_CoM²) ≥ m(l² + z_CoM²)`. **4 of the 10 pairs violate it** — the ridge wanders
where the claimed height alone would need more inertia than the same pair's
`J_P` supplies, so the `(C₂, K)` box should be carrying this inequality as a
constraint and currently is not.

| axis | `J_P` [kg·m²] | `z_CoM` from `1/K` [mm] | `z_CoM` from `J_P` [mm] |
|---|---|---|---|
| roll | `0.249 ± 0.016` (CV 6.5%) | `271 ± 181` (CV 67%) | `214 ± 13` |
| pitch | `0.140 ± 0.034` (CV 24%) | `106 ± 51` (CV 48%) | `131 ± 43` |

So the ridge's invariant is `J_P`, not `W z_CoM`: the same pairs that repeat
`J_P` to 6.5% scatter `z_CoM` by 67%. Across axes neither closes. `z_CoM`
cancels in the axis *difference*, giving a parameter-free test — `J_P(roll) −
J_P(pitch)` must equal `(J_xx − J_yy) + m(l_r² − l_p²)`, i.e. `+0.022` kg·m²
on a six-fold symmetric airframe; measured `+0.109 ± 0.017`, which would demand
`J_xx − J_yy = 0.087` kg·m² (5.2σ, a 165 mm difference in mass spread between
two nominally identical body axes). The roll pivot is the compliant one
(`analysis/com_estimator.py`), so its `J_P` is an effective constant, not a
rigid-body pivot inertia.

### Model-fidelity analysis

Certifies that the post-onset response stays in the cosh family — every
modeled effect (gravity nonlinearity, ground effect) only renames the
effective constants (C₂, K) that the calibration estimates, and the residual
forcing outside the estimator's absorbable span {1, τ, δφ} is measured to be
below the execution-noise floor. One script per link of the chain, all taking
the dataset root (e.g. `DataSet/exp`):

| Script | What it certifies |
|---|---|
| `analysis/ge_linearity.py` | exact IGE model is linear in tilt over the measured range (closed-form k_GE; no dataset needed) |
| `analysis/ge_trajectory.py` | exact GE moment along measured trajectories: slope ≈ 1% of Ṁ (rate-invariant → absorbed by K), projection residual ≤ ~2 mN·m |
| `analysis/small_angle_trajectory.py` | out-of-span small-angle residual along measured trajectories (vs the a-priori Lagrange bound) |
| `analysis/tilt_range.py` | measured tilt range (onset / window edge) that all bounds are evaluated over |
| `analysis/postonset_linearity.py` | ramp linearity over the fit segment (vs the full-window gate value) |
| `analysis/cosh_fidelity.py` | zero-free-parameter fit residual: NRMSE flat across ramp rates = the empirical shape certificate |
| `analysis/tiltcap_ablation.py` | identification repeated with a 3° relative tilt cap — result insensitive to the window extent |
| `analysis/alpha_at_peak.py` | signal level (angular acceleration at the moment peak) against which error bounds are compared |
| `analysis/excitation_angle_design.py` | theoretical design tables: systematic shape deviation vs ramp rate × excursion cap (Duhamel propagation of the a-priori remainder along the nominal trajectory; no dataset needed) — shows the 5° cutoff is the largest common cap keeping the deviation ≤1% at the slowest rate |

The full argument — proposition, Lagrange remainder, Duhamel bound, design
consequence, ideal-vs-experimental operating point, and the four measured
certificates — is written up as a drop-in manuscript section in
`docs/fidelity_section.tex`, whose header maps every quoted number to the
script that reproduces it.

### Estimator benchmark (COSH vs NLS / CPD / CUSUM)

Five onset estimators on identical inputs (windows, caps, gates), compared on
the deliverable (CoM offset vs load-cell truth, Table 7) and on run-level
dispersion; figures carry Welch-t 95 % confidence intervals.

```bash
python analysis/nls_comparison.py <outdir>          # runs all 5 estimators → CSVs
python analysis/estimator_ci_figure.py <outdir>     # Fig: forest plot (offset + CI vs truth)
python analysis/estimator_error_figure.py <outdir>  # Fig: per-case error, 1×5 subplots
```

| Script | What it shows |
|---|---|
| `analysis/nls_comparison.py` | per-run M_crit for cosh / TRF-NLS / CPD (Gaussian, RBF cost) / CUSUM; directional means, CVs, M_ff, CoM offset vs truth (`nls_comparison_runs.csv`, `nls_comparison_summary.csv`) |
| `analysis/estimator_ci_figure.py` | forest plot `docs/fig_estimator_ci.pdf`: offset ± CI₉₅ per method vs truth, per case–axis; also prints the LaTeX table cells |
| `analysis/estimator_error_figure.py` | `docs/fig_estimator_err.pdf`: offset error ± CI₉₅ folded about the truth, 1×5 case subplots, x/y components |
| `analysis/baseline_stat_ablation.py` | onset-sweep baseline statistic ablation (median vs mean vs pinned zero), full recalibration included |
| `analysis/bias_drift.py` | in-window pre-onset baseline drift (bounds the constant-baseline assumption) |

Both figure scripts read `nls_comparison_runs.csv` from `<outdir>` (produced by
`nls_comparison.py`) and write the figures into `docs/`. The comparison is
written up in `docs/cosh_methodology.pdf`, Sec. "Comparison with per-run
estimators".

### Change-point benchmark

Validates that the identified onset does not depend on the detection method,
against single change-point detection (CPD; Gaussian and RBF-kernel cost) and CUSUM:

```bash
python analysis/pelt_crosscheck.py DataSet/exp/case_05/My --axis y --save-fig --no-plot
```

Writes `onset_benchmark_<axis>.csv` and
`onset_benchmark_downstream_<axis>.csv`.

### All cases, both axes

```bash
for c in case_01 case_02 case_03 case_04 case_05; do
  for s in Mx My; do
    ax=$([ "$s" = "Mx" ] && echo x || echo y)
    python critical_value_getter_piecewise.py DataSet/exp/$c/$s \
        --axis $ax --save-fig --no-plot --output-dir results/${c}_${s}
  done
done
```

---

## Python API

```python
from utils.extractor import load_excitation_dataset
from critical_value_getter_piecewise import extract_piecewise_batch

bags = load_excitation_dataset("DataSet/exp/case_05/My")
crits, fits = extract_piecewise_batch(bags, 'y')   # estimates and pins C₂, K

for c in crits:
    print(c.bag_name, c.onset_time, c.onset_moment)
```

Pin the rig constants yourself:

```python
from critical_value_getter_piecewise import estimate_rig_constants, extract_piecewise

C2, K = estimate_rig_constants(bags, 'y')          # method='de' for a global search
crit, fit = extract_piecewise(bags[0], 'y', model='cosh', cosh_c2=C2, ramp_gain=K)
```

Excitation quality — how well the applied ramp tracked the command:

```python
from critical_value_getter_piecewise import commanded_ramp_rate, assess_ramp_quality

q = assess_ramp_quality(crit, commanded_ramp_rate(crit.bag_name))
# actual_rate, slope_error_pct, linearity_rmse, tracking_rmse
```

Signals only, with no onset model fitted:

```python
from critical_value_getter_piecewise import prepare_signals
sig = prepare_signals(bags[0], 'y')     # t, omega, f_col, moment
```

---

## Notes and limitations

- **Quasi-static regime.** The static equilibrium model assumes the tip-over
  happens at the static threshold. At slow ramps the closed form, its
  leading-order limit and generic change-point detectors agree to ~0.05 N·m;
  as `Ṁ` grows, inertial dynamics enter and the methods spread apart. Ramp rate
  is therefore a reported experimental variable, not a nuisance.
- **Excitation range.** On a sloped surface the negative-pitch direction only
  reaches ~3° of tilt (vs ~8° positive), which lowers the SNR of those runs.
  Constraining the amplitude by physics is what keeps them usable.
- **Rig constants are loosely determined.** `C₂` and `K` sit on a flat ridge of
  the objective, so `W·z_CoM = 1/K` and `J_P = 1/(K·C₂²)` are order-of-magnitude
  sanity checks, not precision measurements. The identified onset — and hence
  `M_crit` — is robust along that ridge, which is what the consistency criterion
  pins down.
- **`pivot-based` vs `pivot-free`.** The pivot-based CoM estimate needs mocap and
  is used offline as a reference; the pivot-free moment offset
  `M_ff = ½(M_pos + M_neg)` needs no mocap and additionally retains the effect of
  landing-gear asymmetry, which the CoM→moment conversion drops.
