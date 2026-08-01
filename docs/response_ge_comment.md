# Response to the ground-effect comment (Reviewer 2)

> *"The paper ignores critical ground-effect disturbance literature. It
> only calculates a rough 1.1% thrust bias from rotor ground effect but
> omits discussions on ground-induced rolling/pitching aerodynamic
> moments, which introduce extra offset errors during pre-flight
> excitation. This oversight might create an incomplete system modeling
> foundation."*

Every number below is reproducible from the repository
(`analysis/ge_linearity.py`, `analysis/ge_trajectory.py`; constants:
h = 0.315 m, R = 0.127 m, l_p = 0.140/0.110 m conservative,
W = 31.59 N).

---

**Response.** We thank the reviewer and agree that the presentation was
incomplete: the manuscript reported the rotor-common thrust bias
explicitly but left the ground-induced roll/pitch moments implicit.
They were, however, contained in the model from the outset — the
per-rotor heights $h_i(\phi) = h\cos\phi + s_i\sin\phi$ differ under
tilt, and the resulting differential thrust produces a net aerodynamic
moment, the standard mechanism of asymmetric ground effect. In the
revision we make this explicit, angle-resolved, and bracketed by two
models:

1. **Lower bound — single-rotor superposition.** Each rotor follows the
   classical Cheeseman–Bennett result at its own tilted height;
   rotor–rotor interference is neglected. Moment-proportional
   coefficient $b = \bar\gamma - 1 = 1.0\%$.

2. **Upper bound — attitude-dependent adaptation of Garofano-Soldado
   *et al.*, IEEE RA-L 9(2):1907–1914, 2024.** Their co-planar
   hexarotor model — characterized on a **static test bench**, i.e. in
   the same quasi-static, zero-airspeed regime as our contact-phase
   excitation, free of the translational-inflow effects present in
   in-flight characterizations — is evaluated with the rotor
   interference sum at the tilted rotor heights (exact pairwise
   distances) and the fountain-driven body lift of their Eq. (8),
   calibrated at the level configuration to their Eq. (9) for our
   geometry. This gives $b = 4.2\%$ and a thrust-channel gain of
   $11.7\%$ of $f\,l_p$. Its empirical constants are transferred from
   the authors' vehicle — an informed estimate rather than a
   measurement, which is why we present the two models as a bracket.

**Angle-resolved decomposition (new Fig./Table [—]).** In both models
the pivot moment decomposes identically,
$\Delta M_{\mathrm{GE},P} = a(\phi) + b(\phi)\,M_x$, and each channel
has a definite fate in the identification:

- **Thrust channel $a$** (rotor-common gain *and* body lift): acts
  through the moment arm $l_p$ *exactly*, hence is antisymmetric
  between the two tip directions and **cancels exactly in the
  pivot-free average $M_{\mathrm{ff}}$, regardless of its magnitude or
  model** (up to $0.34$ N·m per direction under the upper model — and
  still cancelled).
- **Moment coefficient $b$**: its ramp component is proportional to the
  commanded rate with a rate-independent coefficient — confirmed *in
  situ* by evaluating the exact models along all 140 measured
  trajectories: slope ratios $+0.99\%$ (lower) and $+3.85\%$ (upper,
  IQR $[3.75, 3.98]\%$), flat from 0.10 to 1.20 N·m/s — and is
  therefore absorbed by the ramp-invariance calibration of $K$. The
  onset-instant part leaves a relative bias on $M_{\mathrm{ff}}$
  bracketed between $-1.0\%$ and $-4.2\%$ ($3$–$14$ mN·m, equivalent
  to $0.10$–$0.43$ mm of CoM offset).
- **State-linear slope** (the ground-induced restoring spring): $1.0\%$
  to $3.8\%$ of the gravitational stiffness $W z_{\mathrm{CoM}}$,
  absorbed by the data-calibrated $C_2$ and well inside the $\pm 20\%$
  range the sensitivity analysis shows harmless.
- **Residual (shape-relevant) part**: measured by projecting the exact
  trajectory-evaluated moment out of the estimator span
  $\{1, \tau, \delta\phi\}$ — at most $2.3$ (lower) to $22$ mN·m
  (upper), median RMS $0.16$–$1.7$ mN·m, edge-concentrated, with a
  propagated shape deviation at the few-percent level, below the
  noise-dominated fit residual.

**Why the bracketed bias does not affect the conclusions.** The
load-cell validation over ten CoM configurations closes the loop
model-independently: the signed $M_{\mathrm{ff}}$-versus-truth residual
has an RMS of $52$ mN·m, dominated by genuine contact asymmetries
(landing-gear geometry) that the pivot-free average is *designed* to
capture; both models' predicted biases lie below this scatter, and a
regression of the signed error against the true moment gives a slope of
$+0.2\% \pm 6.9\%$ — consistent with either model and too small to
matter. We further note that the attitude-common part of any
near-ground aerodynamic moment is present identically at lift-off —
the very instant $M_{\mathrm{ff}}$ compensates — so its capture by the
pivot-free average is by design, not an error source.

**Literature.** The revised related-work discussion now cites the
ground-effect disturbance literature: Cheeseman & Bennett (1955);
Garofano-Soldado *et al.* (RA-L 2024, static-bench hexarotor — the
regime matching our excitation); Sánchez-Cuevas *et al.* (2017,
partial-ground-effect moments, cited qualitatively since their
measurements include translational inflow); Conyers *et al.* (2018);
and the review of Matus-Vargas *et al.* (2021).

**Changes:** angle-resolved GE moment figure/table with the two-model
bracket [Fig./Table —]; channel-fate discussion in [Sec. —]; expanded
related work [Sec. —]; limitation note on transferred empirical
constants [Sec. —].
