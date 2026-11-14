# 6. Wisconsin Diagnostic Breast Cancer (WDBC)

QCML on 569 tumors with 30 measurements of their cell nuclei, each benign or malignant. The scripts are in the order they were written.

**To reproduce:** `python3 QCML110426.py` makes Table 3.1, and `python3 QCML111326_2.py` makes the data for Figure 3.1, which `3_QCML_Plus/QCML111326_3.py` draws. `QCML092926_2.py` and `QCML100626.py` are earlier steps.

| File | What it does | Runs in |
|---|---|---|
| `QCML092926_2.py` | The first QCML model of the breast cancer data. | not timed |
| `QCML100626.py` | One QCML manifold per diagnosis against a PCA classifier, with stratified splits, z-scores and features removed. | about 10 min |
| `QCML110426.py` | **The accuracy table** of the results chapter: all six classifiers (supervised and unsupervised QCML, the diffusion map, logistic regression, PCA, the nearest centroid) on the same ten splits, with the diagnostic measures and the corrected z-test. Its Part B redraws two earlier reservoir figures. | about 2 min |
| `QCML111326_2.py` | **Missing features:** every feature removed in turn, then the three most damaging together, for all six classifiers. | about 10 min on 8 cores |

**Data:** `wdbc.data`, `wdbc.names`.

**Saved results:** `qcml100626_feature_sweep.csv`, `qcml110426_wdbc_results.csv`, `qcml110426_wdbc_splits.csv`, `qcml111326_2_removal.csv`, `qcml111326_2_removal_splits.csv`.

**Copied from other folders** for Part B of `QCML110426.py`: `qcml110226_readouts.csv` (from `3_QCML_Plus`) and `qcml102426_2_cost.csv` (from `5_Dudas_Reservoir_Simulation`).
