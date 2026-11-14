# 7. The egg

2,500 noisy points around an egg-shaped curve: the example that teaches QCML. The scripts are in the order they were written.

**To reproduce:** `python3 QCML110426_2.py` draws Figure 1.1, and `python3 QCML102826.py` draws Figures 1.2, 2.3 and 2.4 (it imports `QCML102726.py`, which computes them). The other scripts are earlier steps.

| File | What it does | Runs in |
|---|---|---|
| `QCML_92126.py` | The first egg model, with its own Hermitian matrices written out from scratch. | not timed |
| `QCML_92226.py`, `QCML_92426.py` | The egg with the `QCML` class of `1_Plain_QCML`. | not timed |
| `QCML092926.py` | The egg model with its plots. | not timed |
| `QCML102726.py` | The pictures of the introduction and methods chapters: PCA, the diffusion map and QCML on the egg's cut-off and chunk tests, and a learned observable of the breast cancer data. | about 3 min |
| `QCML102826.py` | The thesis versions of those figures. Imports `QCML102726.py`. | about 30 s |
| `QCML110426_2.py` | The egg figures of the introduction. | a few seconds |

**Data:** `wdbc.data` and `wdbc.names`, which `QCML102726.py` uses for its breast cancer observable.

**Saved results:** `qcml102726_*.csv`.
