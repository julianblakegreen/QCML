# 5. Dudas reservoir simulation

The two coupled quantum oscillators of Dudas et al. (2023), rebuilt in Python, and the measurement protocols of the thesis.

**To reproduce:** `python3 QCML102426.py` makes Tables 3.3 and 3.4, filling in unmeasured values and the first readouts. `QCML102426_2.py` is the weak-measurement study described in the Methods chapter, and `Dudas_Python_Weak.py` only checks the simulation against the published results.

| File | What it does | Runs in |
|---|---|---|
| `QCML102426.py` | Rebuilds the reservoir (an exact coherent-state shortcut, checked against the full master equation and their saved results). It then measures fewer values (the 40 values, or the vacuum probabilities alone) and with fewer shots, fills in the unmeasured values with QCML trained to fill in, and predicts with supervised QCML. | about 20 min |
| `QCML102426_2.py` | Weak continuous measurement of the photon numbers: measurement strength against runs, rebuilding the projective record from the weak one, and experiment time against accuracy. | about 20–40 min |
| `Dudas_Python_Weak.py` | An audit: the remake of their code, the weak measurement and the thesis changes in one file, with checks against their published results. Writes nothing. | about 7 min |
| `dudas_export.jl` | Exports the saved results of Dudas et al. from their JLD2 files to `dudas_data/`. Only needed if `dudas_data/` is lost; see the notes at the end of the file. | not timed |

**Data:** `dudas_data/` (their saved results as CSV).

**Saved results:** `qcml102426_fill_in_table.csv` and `qcml102426_prediction_table.csv` (the fill-in table and the first-readout table of the thesis), and `qcml102426_2_*.csv`.

**Copied from `3_QCML_Plus`** for the audit's comparisons: `qcml102426_3_missing_data.csv` and `qcml110226_noise_weights.csv`.
