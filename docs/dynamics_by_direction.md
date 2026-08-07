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

## Why `ΔM_GE` comes out almost flat over the window

Split Eq. (43) with `η_i = η̄ + δη_i`:

```
ΔM_GE = η̄ · Σ_i b_i T_i  +  Σ_i b_i δη_i T_i
      = η̄ · (M_cmd + f·l_p)   +   [differential]
        └── common channel ──┘
```

Measured over the 68 pitch runs, `δφ` = 1–4°, interference model:

| channel | level | slope |
|---|---|---|
| common `η̄ (M_cmd + f l_p)` | `127.2` mN·m | `+3.72` mN·m/deg |
| differential `Σ b_i δη_i T_i` | `−0.1` | `−2.77` |
| **total** | **`127.1`** | **`+0.95`** |

Four things conspire:

1. **GE is weak and slowly varying here.** `h/R = 2.48`, so
   `η = R²/(16h² − R²) = 1.03%`, and `d ln η / d ln h = −2.02` — a 1% height
   change buys only a 2% change in `η`.
2. **The level is `η̄` times a constant.** `f·l_p = 21 × 0.10 = 2.1` N·m against
   `M_cmd ≈ 0.9–1.4` N·m, and the excitation *holds the collective fixed* by
   design. The dominant factor in the product does not move.
3. **Turning about a pivot at the vehicle's own edge barely moves the mean
   rotor height.** Only the pivot offset survives the average:
   `dh̄/dδφ = l_p ≈ 1.7` mm/deg on 315 mm, i.e. `0.6%`/deg, so `η̄` moves
   `1.2%`/deg — a few mN·m/deg on a 127 mN·m level.
4. **The differential channel cancels most of what is left.** The pivot arms
   are `−129, +100, +329, +329, +100, −129` mm (pitch, pos): the two front
   rotors sit at *negative* arm and descend as it tips (their `η` rises), the
   two rear rotors carry the largest arm and rise (their `η` falls). The two
   contributions are of opposite sign and comparable size — `+3.72` against
   `−2.77` mN·m/deg.

So the flatness is structural, not an artefact: at `h/R ≈ 2.5`, with the
collective pinned and the rotation taken about a contact line 0.1 m from the
body centre, `ΔM_GE` *is* essentially a constant over a 4° sweep. Making it
inform anything would need a lower hover height, a much larger tilt, or a
deliberately varied collective.

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
