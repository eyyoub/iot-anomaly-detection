# IoT Anomaly Detection with Uncertainty Estimation

Research on anomaly detection in IoT time-series data using autoencoders combined with two uncertainty quantification methods: MC Dropout and Conformal Prediction.

Datasets: SKAB (multivariate, 34 files, 8 sensor features) and NAB (58 univariate files across multiple IoT domains).

---

## Notebooks

| Notebook | Description |
|---|---|
| 01_SKAB_EDA_Preprocessing | Load, explore, and preprocess the SKAB dataset |
| 02_NAB_EDA_Preprocessing | Load, explore, and preprocess the NAB dataset |
| 03_Baselines | Statistical baselines (z-score, IQR) |
| 04_Dense_AE_MCD | Dense autoencoder with MC Dropout |
| 05_LSTM_AE_MCD | LSTM autoencoder with MC Dropout |
| 05b_LSTM_AE_MCD_low_dropout | LSTM autoencoder variant with lower dropout rate |
| 06_Conformal_Prediction | Conformal Prediction on top of Dense and LSTM AE |
| 07_Results_Comparison | Side-by-side comparison of all models |
| 08_DenseDropout_Model | Dense model with Dropout architecture variant |

---

## Data

Raw datasets are not included in this repository. The preprocessing notebooks clone the source repos automatically and push the processed arrays to Hugging Face.

| Dataset | Raw source | Preprocessed (HF Hub) |
|---|---|---|
| SKAB | https://github.com/waico/SKAB | https://huggingface.co/datasets/ayyoubsoullami/skab-anomaly-detection |
| NAB | https://github.com/numenta/NAB | https://huggingface.co/datasets/ayyoubsoullami/nab-anomaly-detection |

All training and evaluation notebooks pull directly from the HF Hub at runtime, so no local data is required after preprocessing.

---

## Setup

### Local

```bash
git clone https://github.com/eyyoub/iot-anomaly-detection.git
cd iot-anomaly-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your credentials:

```
HF_USERNAME=your_username
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key   # optional, only needed during training
```

### Google Colab

Open any notebook directly in Colab. The notebooks detect the Colab environment and pull data from the HF Hub automatically. No local setup needed.

Install dependencies at the top of the notebook:

```python
!pip install -r requirements.txt
```

Then set your credentials as Colab secrets (the key icon in the left sidebar) or paste them temporarily into the config cell at the top.

### RunPod

1. Launch a pod with a PyTorch template (CUDA 11.8+).
2. Clone the repo and install dependencies:

```bash
git clone https://github.com/eyyoub/iot-anomaly-detection.git
cd iot-anomaly-detection
pip install -r requirements.txt
```

3. Set environment variables:

```bash
export HF_USERNAME=your_username
export HF_TOKEN=your_token
export WANDB_API_KEY=your_key
```

4. Launch Jupyter and open any training notebook (04, 05, 06).

---

## Experiment Tracking (W&B)

Training notebooks log metrics to Weights and Biases when `WANDB_API_KEY` is set. Set `USE_WANDB = False` in the config cell at the top of any notebook to disable it.

Metrics logged: train/val loss per epoch, F1, AUC-PR, and threshold on the test set.

---

## Key Design Decisions

- Chronological splits only - no random shuffling on time-series data.
- Scaler fitted on training data only, applied to val and test.
- Sliding window applied after splitting to prevent data leakage.
- No dataset balancing - class imbalance is intentional for unsupervised anomaly detection.
- Primary metrics: F1 and AUC-PR (not accuracy).
