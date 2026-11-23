# Uncertainty Estimation for Anomaly Detection in IoT Data

Code for the paper *"Uncertainty Estimation for Anomaly Detection in IoT Data"*
(Soullami and Hornyák). Autoencoders with MC Dropout and Conformal Prediction on SKAB and NAB.

## The paper's experiments: R01–R06

| Notebook | What it does |
|---|---|
| `R01_Data_Splits` | Train / val / calibration / test splits |
| `R02_Grid_Search` | Hyperparameter search |
| `R03_Main_Experiments` | Main results (all methods, 5 seeds) |
| `R04_Dropout_Sweeps` | Dropout-rate sweeps |
| `R05_Window_Size` | Window-size sensitivity |
| `R06_Confidence_Intervals` | Bootstrap confidence intervals |

The notebooks are saved with their outputs. Figures are in `figures_rev/`, tables in `tables_rev/`.

Notebooks `01`–`08` are earlier exploratory work and do not reproduce the paper.

## Run

```bash
pip install -r requirements.txt
git clone --depth 1 https://github.com/waico/SKAB.git
git clone --depth 1 https://github.com/numenta/NAB.git
bash scripts/run_all_notebooks.sh
```
