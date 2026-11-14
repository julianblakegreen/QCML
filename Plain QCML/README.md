# 1. Plain QCML

The model that every other folder builds on.

**Nothing here needs to be run.** The scripts in the other folders import the model from this folder.

| File | What it does |
|---|---|
| `qcml.py` | The `QCML` class (learned Hermitian observables, the error Hamiltonian of a data point, its ground state as the encoding, the read-back or Bloch map, the fluctuation, the loss, the reconstruction and the quantum metric) and a `train` routine. |
| `hermitian.py` | `HermitianMatrices`: a set of n × n matrices that are Hermitian by construction, so they stay Hermitian at every optimizer step. |

Other folders import the model with `from qcml import QCML`, after adding this folder to Python's search path in the set-up lines near the top of each script.
