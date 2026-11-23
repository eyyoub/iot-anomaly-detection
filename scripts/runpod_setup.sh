#!/usr/bin/env bash
# One-time setup on a fresh RunPod pod (PyTorch template). Run from the repo
# root after the code has been copied to the pod:   bash scripts/runpod_setup.sh
set -euo pipefail

PIP="pip install -q --break-system-packages"   # disposable pod: system python is fine
$PIP numpy pandas scipy scikit-learn matplotlib nbformat nbconvert ipykernel

# raw datasets (not in git)
[ -d SKAB ] || git clone --depth 1 https://github.com/waico/SKAB.git
[ -d NAB ]  || git clone --depth 1 https://github.com/numenta/NAB.git

command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux rsync)

python - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
EOF
echo "setup done"
