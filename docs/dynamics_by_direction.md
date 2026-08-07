# Contact-phase dynamics, per axis and per ramp direction

Frame: FLU (`x` forward, `y` left, `z` up), as `utils/math_tools.py` uses —
`τ_x = L Σ f_i sin θ_i`, `τ_y = −L Σ f_i cos θ_i`, motors at
`30°, 90°, …, 330°`. So `+τ_x` lifts the left side and `+τ_y` puts the nose
down.

`δφ ≥ 0` is the rotation away from the **resting** attitude in the tipping
sense of that direction (the floor is tilted and the vehicle rests flat on it,
so rest — not the gravity horizontal — is the zero). `λ_y`, `λ_x` are the CoM
offsets from the point the arms are measured from, `l_p,φ` / `l_p,θ` the half
landing-gear spans, `z_CoM` the CoM height above the contact plane, and `f` the
collective.

## Which contact line each direction pivots about

Verified run by run against the signed circle-fit centre `cx` of
`estimate_pivot_from_mocap`, called once the vehicle is actually turning
(`δφ ≥ 1°`) — called at the excitation-window start instead it fits noise and
returns a meaningless centre on part of the runs:

| axis | direction | commanded | pivot line | measured `cx` | model uses |
|---|---|---|---|---|---|
| roll `M_x` | pos (`Ṁ > 0`) | `+τ_x`, left lifts | right contact, `y = −l_p,φ` | `−138.1 ± 1.2` mm | `−130` mm |
| roll `M_x` | neg (`Ṁ < 0`) | `−τ_x`, right lifts | left contact, `y = +l_p,φ` | `+141.3 ± 4.2` mm | `+130` mm |
| pitch `M_y` | pos (`Ṁ > 0`) | `+τ_y`, nose down | front contact, `x = +l_p,θ` | `+120.9 ± 23.3` mm | `+100` mm |
| pitch `M_y` | neg (`Ṁ < 0`) | `−τ_y`, nose up | rear contact, `x = −l_p,θ` | `−105.6 ± 3.8` mm | `−100` mm |

## The four balances

Each is written about its own pivot line, positive in that direction's tipping
sense:

**roll, pos**
```
J_P δφ̈ = +M_x + f·l_p,φ − W[(l_p,φ + λ_y)·cos δφ − z_CoM·sin δφ] + ΔM_GE
```
**roll, neg**
```
J_P δφ̈ = −M_x + f·l_p,φ − W[(l_p,φ − λ_y)·cos δφ − z_CoM·sin δφ] + ΔM_GE
```
**pitch, pos**
```
J_P δφ̈ = +M_y + f·l_p,θ − W[(l_p,θ − λ_x)·cos δφ − z_CoM·sin δφ] + ΔM_GE
```
**pitch, neg**
```
J_P δφ̈ = −M_y + f·l_p,θ − W[(l_p,θ + λ_x)·cos δφ − z_CoM·sin δφ] + ΔM_GE
```

Three things are common to all four and are what the sign bookkeeping has to
get right:

- the **collective always helps**, `+f·l_p`, whichever way it tips — the thrust
  is on the body side of the pivot in every case, and it acts through `l_p`
  (the same arm `ΔM_GE` uses), not through the CoM arm;
- the **`z_CoM` term always has the `+` sign** on `sin δφ` — gravity is
  anti-restoring once it starts turning, in both directions;
- only the **commanded moment and the CoM-offset term change sign** between
  `pos` and `neg`. That is exactly why the pivot-free average
  `M_ff = ½(M₊ + M₋)` keeps `λ` and cancels the arm.

Setting `δφ = 0`, `δφ̈ = 0` recovers the static thresholds of the manuscript:

```
M_x,± = ±(W − f) l_p,φ + W λ_y − ΔM_GE
M_y,± = ±(W − f) l_p,θ − W λ_x − ΔM_GE
```

## The ground-effect term, per direction

Eq. (43) with the pivot-relative arm of Eq. (38)/(39), signed so that
`ΔM_GE > 0` always means *aids the tip*:

| axis | direction | `b_i` in `ΔM_GE = Σ_i b_i η_i^GE T_i` | `Σ_i b_i T_i` |
|---|---|---|---|
| roll | pos | `+l_y,i + l_p,φ` | `+M_x + f l_p,φ` |
| roll | neg | `−l_y,i + l_p,φ` | `−M_x + f l_p,φ` |
| pitch | pos | `−l_x,i + l_p,θ` | `+M_y + f l_p,θ` |
| pitch | neg | `+l_x,i + l_p,θ` | `−M_y + f l_p,θ` |

`Σ_i b_i T_i` reproduces that direction's own commanded term **plus** `f·l_p`,
which is why evaluating (43) on the measured per-rotor thrusts `T_i` already
carries both channels of the affine form (44), `η_M M_cmd + η_f f l_p`. No
separate collective term is needed, and the balance above must use the same
`l_p` for its `f·l_p`.

The rotor heights entering `η_i^GE = γ_i^GE − 1` are heights above the
**ground**, so they are taken from the attitude *relative to rest*:
`h_i = (R_rest^T R(t) · p_i^P)_z` with `p_i^P` the pivot-relative hub position
of Eqs. (38)/(39), `z`-component `h = 0.315` m.

## After the throttle cut

The excitation stops the rotors at 5° of absolute tilt (the logged cut lands at
`8.0°` median — trigger latency plus spin-down). Past it `f = 0`, `M = 0` and
`ΔM_GE = 0`, and all four balances collapse to one pendulum:

```
J_P δφ̈ = W ρ sin(δφ − ψ) ,   ρ = √(h₀² + z_CoM²) ,   tan ψ = h₀ / z_CoM
```

with `h₀` that direction's own gravity arm (`l_p ± λ`). Its energy integral,
`ω² = C − A cos δφ − B sin δφ` with `A = 2W z_CoM/J_P = 2C₂²` and
`B = 2W h₀/J_P`, is what `analysis/zcom_freefall.py` fits — the only part of
the run with no thrust, no commanded moment and no ground effect in it.
