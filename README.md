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

### Change-point benchmark

Validates that the identified onset does not depend on the detection method,
against PELT (Gaussian and RBF-kernel cost) and CUSUM:

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
