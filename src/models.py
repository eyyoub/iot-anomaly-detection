"""Models, training and MC Dropout inference.

Architectures are the same as in the exploratory notebooks:
  DenseAE            nb 03/04 (B3 with p=0, M1 with p>0)
  LSTMAE             nb 05    (M2/M2b with p>0, M4 backbone with p=0)
  DenseDropoutModel  nb 08    (M5)

Training keeps the whole dataset on the GPU and batches by index, which is
much faster than a DataLoader for models this small. The optimisation
itself (Adam, MSE, batch size, early stopping on val MSE) is unchanged.
"""
import copy

import numpy as np
import torch
from torch import nn


class DenseAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32, dropout_p=0.0):
        super().__init__()
        p = dropout_p
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p),
            nn.Linear(64, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Dropout(p),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, input_dim), nn.Sigmoid(),
        )
        self.flat = True

    def forward(self, x):
        return self.decoder(self.encoder(x))


class LSTMAE(nn.Module):
    def __init__(self, n_features, hidden_dim=64, latent_dim=32, dropout_p=0.05):
        super().__init__()
        self.enc_lstm = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.enc_drop = nn.Dropout(dropout_p)
        self.enc_fc = nn.Linear(hidden_dim, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, hidden_dim)
        self.dec_drop = nn.Dropout(dropout_p)
        self.dec_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dec_out = nn.Linear(hidden_dim, n_features)
        self.flat = False

    def forward(self, x):
        steps = x.size(1)
        _, (h_n, _) = self.enc_lstm(x)
        z = self.enc_fc(self.enc_drop(h_n.squeeze(0)))
        d = self.dec_drop(self.dec_fc(z)).unsqueeze(1).expand(-1, steps, -1)
        out, _ = self.dec_lstm(d)
        return torch.sigmoid(self.dec_out(out))


class DenseDropoutModel(nn.Module):
    def __init__(self, input_dim, n_units=256, dropout_p=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, n_units), nn.Sigmoid(), nn.Dropout(dropout_p),
            nn.Linear(n_units, n_units // 4), nn.Sigmoid(),
            nn.Linear(n_units // 4, input_dim),
        )
        self.flat = True

    def forward(self, x):
        return self.net(x)


def build(arch, window_size, n_features, latent_dim=32, dropout_p=0.0,
          hidden_dim=64, n_units=256):
    d = window_size * n_features
    if arch == "dense":
        return DenseAE(d, latent_dim, dropout_p)
    if arch == "lstm":
        return LSTMAE(n_features, hidden_dim, latent_dim, dropout_p)
    if arch == "densedrop":
        return DenseDropoutModel(d, n_units, dropout_p)
    raise ValueError(arch)


def _prep(model, X, device):
    t = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=device)
    return t.reshape(len(t), -1) if model.flat else t


def _val_loss(model, Xv, bs=4096):
    model.eval()
    with torch.no_grad():
        tot = sum(((model(Xv[i:i + bs]) - Xv[i:i + bs]) ** 2).mean().item()
                  * len(Xv[i:i + bs]) for i in range(0, len(Xv), bs))
    return tot / len(Xv)


def train(model, X_train, X_val, lr=1e-3, epochs=100, patience=10,
          batch_size=256, min_delta=1e-7, seed=0, device="cuda"):
    """Adam + MSE, early stopping on deterministic val MSE, best weights."""
    torch.manual_seed(seed)
    model.to(device)
    Xt, Xv = _prep(model, X_train, device), _prep(model, X_val, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator(device=device).manual_seed(seed)
    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=device, generator=g)
        for i in range(0, len(Xt), batch_size):
            xb = Xt[perm[i:i + batch_size]]
            loss = ((model(xb) - xb) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        vl = _val_loss(model, Xv)
        hist.append(vl)
        if vl < best - min_delta:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_val": best, "epochs": len(hist), "val_curve": hist}


@torch.no_grad()
def det_scores(model, X, device="cuda", bs=4096):
    """Deterministic per-window reconstruction MSE (dropout off)."""
    model.eval()
    Xt = _prep(model, X, device)
    dims = tuple(range(1, Xt.dim()))
    return torch.cat([((model(Xt[i:i + bs]) - Xt[i:i + bs]) ** 2).mean(dim=dims)
                      for i in range(0, len(Xt), bs)]).cpu().numpy()


@torch.no_grad()
def mc_scores(model, X, T=50, device="cuda", bs=4096, seed=0):
    """T stochastic passes with dropout on.

    Returns (n_windows, T) matrix of per-pass reconstruction MSE. From it:
      mean over passes = MCD anomaly score
      std over passes  = MCD uncertainty (the paper's definition: std of the
                         per-pass *reconstruction error*, not of the outputs)
      per-pass decisions vs a threshold = vote disagreement
    """
    torch.manual_seed(seed)
    model.train()
    Xt = _prep(model, X, device)
    dims = tuple(range(1, Xt.dim()))
    out = []
    for i in range(0, len(Xt), bs):
        xb = Xt[i:i + bs]
        out.append(torch.stack([((model(xb) - xb) ** 2).mean(dim=dims)
                                for _ in range(T)], dim=1).cpu())
    model.eval()
    return torch.cat(out).numpy()
