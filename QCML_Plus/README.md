# 3. QCML+

QCML+ adds three steps in front of supervised QCML: time-delay inputs, noise weights and compression with PCA. The scripts are in the order they were written, and each builds on the one before.

**To reproduce:**
- `python3 QCML110226.py` makes Table 3.5, the final comparison of QCML+ with the classical readouts, and the uncertainty check.
- `python3 QCML111326.py` makes Table 3.6 and the data for Figures 3.3 and 3.4.
- `python3 QCML111326_3.py` draws Figures 3.1 to 3.4 from the saved results.

`QCML102426_3.py` and `QCML102426_5.py` are earlier steps, kept to show how QCML+ was developed.

| File | What it does | Runs in |
|---|---|---|
| `QCML102426_3.py` | Ridge regression as the baseline readout, and a screening of ways to improve QCML. The winners, time-delay inputs and noise weights, become the improved inputs. | about 30 min |
| `QCML102426_5.py` | QCML+: the improved inputs compressed with PCA, the number of components and the dimension n chosen on validation, and longer training. Also the quantum-variance uncertainty. | about 45 min |
| `QCML110226.py` | **The final comparison** of the thesis: QCML+, the diffusion map and kernel ridge regression on the same inputs, with noise weights measured from two halves of the data. | about 10 min on 8 cores |
| `QCML111326.py` | Sweeps of the weak record (10² to 10⁸ runs) and the projective record (9³ to 9⁹ shots) for every readout. Imports `QCML110226.py`. | about 45 min on 8 cores |
| `QCML111326_3.py` | The figures of the results chapter, drawn from the saved CSVs. | a few seconds |

**Saved results:** `qcml102426_3_*.csv`, `qcml102426_5_*.csv`, `qcml110226_*.csv` (the final comparison table and the uncertainty check) and `qcml111326_*.csv` (the sweeps and the experiment-time comparison).

**Copied from other folders** for `QCML111326_3.py`: `qcml102426_2_cost.csv` (from `5_Dudas_Reservoir_Simulation`) and `qcml111326_2_removal.csv` (from `6_WDBC_Breast_Cancer`).

`QCML111326.py` also caches its raw results in `qcml111326_raw.pkl`. Rerunning it with `--saved` rebuilds the tables from that cache. The cache is not in the repository.
