# 2. Supervised QCML

**To reproduce:** `python3 QCML102326_2.py` makes Table 3.2 of the thesis, the Mackey–Glass benchmark.

| File | What it does | Runs in |
|---|---|---|
| `QCML102326_2.py` | Adds supervised QCML (Di Caro et al., 2025): a target observable B whose expected value in the ground state is the prediction. Part A classifies the breast cancer data with it (mixed loss) beside unsupervised QCML, PCA and the diffusion map, and removes features one at a time. Part B forecasts the Mackey–Glass series six steps ahead, the benchmark of the results chapter. | about 20 min |

**Data:** `wdbc.data`, `wdbc.names`.

**Saved results:** `qcml102326_2_wdbc_table.csv`, `qcml102326_2_wdbc_metrics.csv`, `qcml102326_2_mackey_glass_table.csv` (the Mackey–Glass benchmark table of the thesis).
