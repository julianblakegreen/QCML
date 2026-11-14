# 4. Baseline methods

The classical methods that QCML is measured against.

**Earlier steps, cited in the Methods chapter.** Nothing here is needed to reproduce the results: the final comparison of the classical readouts with QCML+ is `3_QCML_Plus/QCML110226.py`.

| File | What it does | Runs in |
|---|---|---|
| `QCML102326.py` | QCML against PCA and a diffusion map (with an adaptive kernel width). Part A uses the breast cancer data: describing the patients, classifying them with one model per diagnosis, and removing features. Part B forecasts the Mackey–Glass series. | about 15 min |
| `QCML102426_4.py` | On the quantum reservoir, every readout on the same inputs: ridge regression, the diffusion map, kernel ridge regression, a neural network and supervised QCML, plus a control without the reservoir. | about 12 min |

**Data:** `wdbc.data`, `wdbc.names`.

**Saved results:** `qcml102326_wdbc_table.csv`, `qcml102326_mackey_glass_table.csv`, `qcml102426_4_readouts.csv`.
