"""Step 3.3: a small 1D convolutional neural network (1D-CNN) that reads 10 s of
raw skin color (R, G, B) and predicts the heart rate in bpm.

It gets the colors only -- no POS formula -- so it has to learn by itself how to
mix the three colors and how to read the rhythm.
"""
import numpy as np
import torch
from torch import nn

import signal_proc as sp
import windows

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # Colab's GPU, if it's switched on


def prepare(rgb_windows):
    """(n, 300, 3) raw colors -> (n, 3, 300) network input.

    1. relative change per color (removes skin tone and overall brightness)
    2. the same 0.7-4 Hz bandpass that green and POS use (removes drift and noise)
    3. one common scale per window for all three colors, so the RATIOS between
       the colors survive -- that's the information POS relies on
    4. reorder to (colors, time): PyTorch wants the channels before time
    """
    x = rgb_windows / rgb_windows.mean(axis=1, keepdims=True) - 1
    x = np.stack([sp.bandpass(w, windows.FPS_OUT) for w in x])
    x = x / (x.std(axis=(1, 2), keepdims=True) + 1e-8)
    return np.transpose(x, (0, 2, 1)).astype(np.float32)


def speed_change(x, bpm, rng, out_len=224, slowest=0.75, fastest=1.3):
    """Augmentation: play each window slower or faster by a random factor r, and
    multiply its heart rate by r. This creates examples at every heart rate (about
    50-165 bpm), so the network learns "faster rhythm = higher bpm" instead of
    playing safe with values near the average. Output windows are 224 samples."""
    n, c, length = x.shape
    r = rng.uniform(slowest, fastest, n)
    out = np.empty((n, c, out_len), np.float32)
    for k in range(n):
        start = rng.uniform(0, length - 1 - out_len * r[k])   # random crop position
        pos = start + np.arange(out_len) * r[k]                # read every r-th step
        for ch in range(c):
            out[k, ch] = np.interp(pos, np.arange(length), x[k, ch])
    return out, bpm * r


class PulseCNN(nn.Module):
    """The network itself: 4 convolution blocks, then a small dense 'head'."""

    def __init__(self):
        super().__init__()

        def block(c_in, c_out):
            # 7-step filters (~0.23 s) slide along time; BatchNorm keeps the numbers
            # in a stable range; ReLU lets the network build non-linear patterns
            return nn.Sequential(nn.Conv1d(c_in, c_out, kernel_size=7, padding=3),
                                 nn.BatchNorm1d(c_out), nn.ReLU())

        self.features = nn.Sequential(
            block(3, 16), nn.MaxPool1d(2),     # 300 -> 150 time steps
            block(16, 32), nn.MaxPool1d(2),    # 150 -> 75
            block(32, 64), nn.MaxPool1d(2),    # 75 -> 37
            block(64, 64),
            nn.AdaptiveAvgPool1d(1))           # average over time -> 64 numbers
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(32, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


class CNNRegressor:
    """PulseCNN with .fit() and .predict(), like a scikit-learn model, so that
    evaluate.loso_predict trains and tests it exactly like any other model."""

    def __init__(self, epochs=40, batch_size=32, lr=1e-3, seed=0):
        self.epochs, self.batch_size, self.lr, self.seed = epochs, batch_size, lr, seed

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        x = prepare(X)
        self.mean, self.std = y.mean(), y.std()   # learn in standard units, convert back later
        self.net = PulseCNN().to(DEVICE)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.SmoothL1Loss()               # like MAE, but smooth near zero
        self.losses = []                          # average training loss per epoch
        for epoch in range(self.epochs):
            self.net.train()
            batches = np.array_split(rng.permutation(len(x)), max(1, len(x) // self.batch_size))
            total = 0.0
            for b in batches:
                xb, yb = speed_change(x[b], y[b], rng)                        # 1. speed
                xb = torch.tensor(xb)
                xb = xb * (1 + 0.15 * (2 * torch.rand(len(b), 3, 1) - 1))    # 2. color balance
                xb = xb + 0.1 * torch.randn_like(xb)                          # 3. noise
                flip = torch.rand(len(b)) < 0.5
                xb[flip] = xb[flip].flip(-1)                                  # 4. backwards
                target = torch.tensor(((yb - self.mean) / self.std).astype(np.float32))
                opt.zero_grad()
                loss = loss_fn(self.net(xb.to(DEVICE)), target.to(DEVICE))
                loss.backward()
                opt.step()
                total += loss.item()
            self.losses.append(total / len(batches))
        return self

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.tensor(prepare(X)).to(DEVICE))
        return out.cpu().numpy() * self.std + self.mean      # back from standard units to bpm
