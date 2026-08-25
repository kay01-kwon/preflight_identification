# Simulated free-flight validation -- run checklist

Only the identifiable cases fly (S10, S12 tip under the excitation collective and are excluded by construction).

Protocol per trial: set the case offset and mass in the model, configure the feedforward from the IDENTIFIED offset below (w_ff runs only), arm, take off to the standard hover point, hold at least 5 s, stop the log. Same topics as the excitation runs. Bag name: `<variant>_<trial>` under `S<k>/`.

### S1 --- truth (-6, +0) mm, mass 3.066 kg
    identified (-6.01, -0.11) mm  ->  pivot-free M_ff: Mx -0.003, My +0.181 N.m

- [ ] wo_ff_1  [ ] wo_ff_2  [ ] wo_ff_3
- [ ] w_ff_1  [ ] w_ff_2  [ ] w_ff_3

### S2 --- truth (+0, +10) mm, mass 3.066 kg
    identified (-0.01, +10.00) mm  ->  pivot-free M_ff: Mx +0.301, My +0.000 N.m

- [ ] wo_ff_1  [ ] wo_ff_2  [ ] wo_ff_3
- [ ] w_ff_1  [ ] w_ff_2  [ ] w_ff_3

### S3 --- truth (+10, -5) mm, mass 3.066 kg
    identified (+10.02, -5.05) mm  ->  pivot-free M_ff: Mx -0.152, My -0.301 N.m

- [ ] wo_ff_1  [ ] wo_ff_2  [ ] wo_ff_3
- [ ] w_ff_1  [ ] w_ff_2  [ ] w_ff_3

### S4 --- truth (+20, +20) mm, mass 3.066 kg
    identified (+20.19, +20.19) mm  ->  pivot-free M_ff: Mx +0.607, My -0.607 N.m

- [ ] wo_ff_1  [ ] wo_ff_2  [ ] wo_ff_3
- [ ] w_ff_1  [ ] w_ff_2  [ ] w_ff_3

### S5 --- truth (+20, -20) mm, mass 3.066 kg
    identified (+20.22, -20.16) mm  ->  pivot-free M_ff: Mx -0.606, My -0.608 N.m

- [ ] wo_ff_1  [ ] wo_ff_2  [ ] wo_ff_3
- [ ] w_ff_1  [ ] w_ff_2  [ ] w_ff_3

### S6 --- truth (-20, +20) mm, mass 3.066 kg
    identified (-20.26, +20.09) mm  ->  pivot-free M_ff: Mx +0.604, My +0.609 N.m

- [ ] wo_ff_1  [ ] wo_ff_2  [ ] wo_ff_3
- [ ] w_ff_1  [ ] w_ff_2  [ ] w_ff_3

### S7 --- truth (-20, -20) mm, mass 3.066 kg
    identified (-20.21, -20.24) mm  ->  pivot-free M_ff: Mx -0.609, My +0.608 N.m

- [ ] wo_ff_1  [ ] wo_ff_2  [ ] wo_ff_3
- [ ] w_ff_1  [ ] w_ff_2  [ ] w_ff_3

### S8 --- truth (+25, +25) mm, mass 3.066 kg
    identified (+25.22, +25.20) mm  ->  pivot-free M_ff: Mx +0.758, My -0.759 N.m

- [ ] wo_ff_1  [ ] wo_ff_2  [ ] wo_ff_3
- [ ] w_ff_1  [ ] w_ff_2  [ ] w_ff_3

### S9 --- truth (+32, +32) mm, mass 3.066 kg
    identified (+32.37, +32.09) mm  ->  pivot-free M_ff: Mx +0.965, My -0.974 N.m

- [ ] wo_ff_1  [ ] wo_ff_2  [ ] wo_ff_3
- [ ] w_ff_1  [ ] w_ff_2  [ ] w_ff_3

### S11 --- truth (+38, +14) mm, mass 3.066 kg
    identified (+38.13, +14.22) mm  ->  pivot-free M_ff: Mx +0.428, My -1.147 N.m

- [ ] wo_ff_1  [ ] wo_ff_2  [ ] wo_ff_3
- [ ] w_ff_1  [ ] w_ff_2  [ ] w_ff_3

### S13 --- truth (+25, +25) mm, mass 3.220 kg
    identified (+25.35, +25.23) mm  ->  pivot-free M_ff: Mx +0.797, My -0.801 N.m

- [ ] wo_ff_1  [ ] wo_ff_2  [ ] wo_ff_3
- [ ] w_ff_1  [ ] w_ff_2  [ ] w_ff_3

**66 flights** = 11 cases x 2 variants x 3 trials.

After the campaign, pack with:

    python analysis/freeflight_pack.py <bag_root> SimDataSet/free_flight
