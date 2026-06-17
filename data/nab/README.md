# NAB — Numenta Anomaly Benchmark

**Raw data source:** https://github.com/numenta/NAB

58 univariate CSV files across multiple IoT/IT domains (AWS, ad-exchange, traffic, tweets, …).
Clone the repo into this folder or let notebook `02_NAB_EDA_Preprocessing.ipynb` do it automatically.

**Preprocessed arrays (sliding windows, train/val/test splits):**
https://huggingface.co/datasets/ayyoubsoullami/nab-anomaly-detection

The preprocessing notebook pushes arrays to the HF Hub; all downstream notebooks
pull from there at runtime — no raw CSVs are needed after preprocessing.
