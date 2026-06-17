# SKAB — Skoltech Anomaly Benchmark

**Raw data source:** https://github.com/waico/SKAB

34 CSV files (8 sensor features) from a water-pump IIoT testbed.
Clone the repo into this folder or let notebook `01_SKAB_EDA_Preprocessing.ipynb` do it automatically.

**Preprocessed arrays (sliding windows, train/val/test splits):**
https://huggingface.co/datasets/ayyoubsoullami/skab-anomaly-detection

The preprocessing notebook pushes arrays to the HF Hub; all downstream notebooks
pull from there at runtime — no raw CSVs are needed after preprocessing.
