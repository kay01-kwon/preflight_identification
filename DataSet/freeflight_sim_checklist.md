# Simulated free-flight validation -- run checklist

Only the identifiable cases fly (S10, S12 tip under the excitation collective and are excluded by construction).

Protocol per trial: set the case offset and mass in the model, configure the feedforward from the IDENTIFIED offset below (w_ff runs only), arm, take off to the standard hover point, hold at least 5 s, stop the log. Same topics as the excitation runs. Bag name: `<variant>_<trial>` under `S<k>/`.

### S1 --- truth (-6, +0) mm, mass 3.066 kg
    feedforward from identified (-6.01, -0.11) mm ->  W*y = -0.003 N.m, W*x = -0.181 N.m

- [ ] wo_ff_1  [ ] wo_ff_2  [ ] wo_ff_3
- [ ] w_ff_1  [ ] w_ff_2  [ ] w_ff_3

### S2 --- truth (+0, +10) mm, mass 3.066 kg
    feedforward from identified (-0.01, +10.00) mm ->  W*y = +0.301 N.m, W*x = -0.000 N.m

- [ ] wo_ff_1  [ ] wo_ff_2  [ ] wo_ff_3
- [ ] w_ff_1  [ ] w_ff_2  [ ] w_ff_3

### S3 --- truth (+10, -5) mm, mass 3.066 kg
    feedforward from identified (+10.02, -5.05) mm ->  W*y = -0.152 N.m, W*x = +0.301 N.m

- [ ] wo_ff_1  [ ] wo_ff_2  [ ] wo_ff_3
- [ ] w_ff_1  [ ] w_ff_2  [ ] w_ff_3

### S4 --- truth (+20, +20) mm, mass 3.066 kg
    feedforward from identified (+20.19, +20.19) mm ->  W*y = +0.607 N.m, W*x = +0.607 N.m

- [ ] wo_ff_1  [ ] wo_ff_2  [ ] wo_ff_3
- [ ] w_ff_1  [ ] w_ff_2  [ ] w_ff_3

### S5 --- truth (+20, -20) mm, mass 3.066 kg
    feedforward from identified (+20.22, -20.16) mm ->  W*y = -0.606 N.m, W*x = +0.608 N.m

- [ ] wo_ff_1  [ ] wo_ff_2  [ ] wo_ff_3
- [ ] w_ff_1  [ ] w_ff_2  [ ] w_ff_3

### S6 --- truth (-20, +20) mm, mass 3.066 kg
    feedforward from identified (-20.26, +20.09) mm ->  W*y = +0.604 N.m, W*x = -0.609 N.m

- [ ] wo_ff_1  [ ] wo_ff_2  [ ] wo_ff_3
- [ ] w_ff_1  [ ] w_ff_2  [ ] w_ff_3

### S7 --- truth (-20, -20) mm, mass 3.066 kg
    feedforward from identified (-20.21, -20.24) mm ->  W*y = -0.609 N.m, W*x = -0.608 N.m

- [ ] wo_ff_1  [ ] wo_ff_2  [ ] wo_ff_3
- [ ] w_ff_1  [ ] w_ff_2  [ ] w_ff_3

### S8 --- truth (+25, +25) mm, mass 3.066 kg
    feedforward from identified (+25.22, +25.20) mm ->  W*y = +0.758 N.m, W*x = +0.759 N.m

- [ ] wo_ff_1  [ ] wo_ff_2  [ ] wo_ff_3
- [ ] w_ff_1  [ ] w_ff_2  [ ] w_ff_3

### S9 --- truth (+32, +32) mm, mass 3.066 kg
    feedforward from identified (+32.37, +32.09) mm ->  W*y = +0.965 N.m, W*x = +0.974 N.m

- [ ] wo_ff_1  [ ] wo_ff_2  [ ] wo_ff_3
- [ ] w_ff_1  [ ] w_ff_2  [ ] w_ff_3

### S11 --- truth (+38, +14) mm, mass 3.066 kg
    feedforward from identified (+38.13, +14.22) mm ->  W*y = +0.428 N.m, W*x = +1.147 N.m

- [ ] wo_ff_1  [ ] wo_ff_2  [ ] wo_ff_3
- [ ] w_ff_1  [ ] w_ff_2  [ ] w_ff_3

### S13 --- truth (+25, +25) mm, mass 3.220 kg
    feedforward from identified (+25.35, +25.23) mm ->  W*y = +0.797 N.m, W*x = +0.801 N.m

- [ ] wo_ff_1  [ ] wo_ff_2  [ ] wo_ff_3
- [ ] w_ff_1  [ ] w_ff_2  [ ] w_ff_3

**66 flights** = 11 cases x 2 variants x 3 trials.

After the campaign, pack with:

    python analysis/freeflight_pack.py <bag_root> SimDataSet/free_flight
