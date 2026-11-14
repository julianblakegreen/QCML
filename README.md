# Quantum cognition machine learning: thesis code

##  To reproduce, do the following:

Install the requirements:

```bash
pip install -r requirements.txt
```

Each result in the thesis comes from one script. Run the script for the result you want, from any directory; it writes its CSV tables and plots into its own folder.

| Thesis result | Run | Runs in |
|---|---|---|
| Table 3.1: breast cancer classification | `python3 6_WDBC_Breast_Cancer/QCML110426.py` | about 2 min |
| Figure 3.1 data: breast cancer with features removed | `python3 6_WDBC_Breast_Cancer/QCML111326_2.py` | about 10 min on 8 cores |
| Table 3.2: the Mackey–Glass benchmark | `python3 2_Supervised_QCML/QCML102326_2.py` | about 20 min |
| Tables 3.3 and 3.4: filling in and the first readouts on the reservoir | `python3 5_Dudas_Reservoir_Simulation/QCML102426.py` | about 20 min |
| Table 3.5: the final comparison with QCML+, and the uncertainty check | `python3 3_QCML_Plus/QCML110226.py` | about 10 min on 8 cores |
| Table 3.6 and the data for Figures 3.3 and 3.4: accuracy against runs, shots and experiment time | `python3 3_QCML_Plus/QCML111326.py` | about 45 min on 8 cores |
| Figures 3.1 to 3.4, drawn from the saved results | `python3 3_QCML_Plus/QCML111326_3.py` | a few seconds |
| Figure 1.1, the egg with missing data | `python3 7_Egg/QCML110426_2.py` | a few seconds |
| Figures 1.2, 2.3 and 2.4: the egg tests, the diffusion coordinates and a learned breast cancer observable | `python3 7_Egg/QCML102826.py` | about 30 s |

- The repository already holds every saved result, so the tables can be read directly from the CSV files and `QCML111326_3.py` redraws all the results figures without rerunning anything. Two of the CSV files it reads are copies from other folders (`qcml111326_2_removal.csv` from `6_WDBC_Breast_Cancer` and `qcml102426_2_cost.csv` from `5_Dudas_Reservoir_Simulation`), so after rerunning the scripts that make them, copy the new files into `3_QCML_Plus` before redrawing.
- Every other script is an earlier step of the work, kept so that each change can be traced, and the Methods chapter cites several of them. None is needed to reproduce the results above.

## Folders

| Folder | What it holds | Thesis |
|---|---|---|
| `1_Plain_QCML` | The QCML model itself (`qcml.py`) and its learned Hermitian matrices (`hermitian.py`). Every other folder imports it from here. | Methods, Section 2.1 |
| `2_Supervised_QCML` | Supervised QCML with a target observable (Di Caro et al., 2025), on the breast cancer data and the Mackey–Glass forecasting benchmark. | Section 2.2; Results, Mackey–Glass benchmark |
| `3_QCML_Plus` | The improved inputs, QCML+, the final comparison of reservoir readouts, the sweeps over runs and shots, and the results figures. | Section 2.3; Results, Sections 3.3.3–3.3.6 |
| `4_Baseline_Methods` | QCML against PCA and the diffusion map, and against every classical reservoir readout (ridge, diffusion map, kernel ridge regression, neural network). | Section 2.4 |
| `5_Dudas_Reservoir_Simulation` | The reservoir of Dudas et al. rebuilt in Python, fewer values and fewer shots, weak continuous measurement, and an audit of the remake. | Section 2.7; Results, Sections 3.3.1–3.3.2 |
| `6_WDBC_Breast_Cancer` | QCML on the Wisconsin Diagnostic Breast Cancer data: classification, diagnostic measures and missing features. | Section 2.6; Results, Section 3.1 |
| `7_Egg` | The egg-shaped curve that teaches QCML, and the figures of the introduction and methods. | Introduction; Section 2.1 |

Each folder has its own README listing its scripts, what each produces and how long it takes.

## How the scripts run

Every script moves to its own folder when it starts, so it can be launched from anywhere. It reads its data from that folder, writes its CSV results and plots there, and imports QCML from `1_Plain_QCML`. The scripts were run with Python 3.13 on an 8-core laptop, where they take from a few seconds to about 45 minutes. The longest use every core.

## How the scripts are named and laid out

- Scripts are named `QCML<MMDDYY>.py` by the date they were written, with `_2`, `_3` for later versions of the same day. Each builds on an earlier one, named in its header, and later scripts supersede earlier ones. The folder READMEs say which script produced each thesis result.
- Each file puts its code first. Long explanations sit in an **Explanation** section at the end of the file, and a tag such as `[note 3]` beside a section heading in the code points to the note that explains that section.
- Each folder includes the saved CSV results that the thesis reports, so the figure scripts run without repeating the long computations. A few of these CSVs are copies of results from another folder, listed in that folder's README.

## Data

- `wdbc.data` and `wdbc.names`: the Wisconsin Diagnostic Breast Cancer data (UCI Machine Learning Repository, dataset 17; Street et al., 1993).
- `5_Dudas_Reservoir_Simulation/dudas_data/`: the saved results of Dudas et al. (2023), *Quantum reservoir computing implementation on coherently coupled quantum oscillators*, npj Quantum Information 9, 64, exported to CSV by `dudas_export.jl`. The reservoir records themselves are simulated by the Python scripts.
