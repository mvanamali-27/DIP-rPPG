"""Step 3.4: robustness -- how does each method cope when the video gets worse?

We damage the dataset's raw color windows in three realistic ways, then ask every
method for the heart rate again. The CNN is NOT retrained on damaged windows: like
a real user, it only meets the problems at test time.
"""
import numpy as np

import evaluate
import signal_proc as sp
import windows

FPS = windows.FPS_OUT


def add_flicker(rgb, amount=0.002, bpm=60):
    """A lamp that flickers by +-amount (0.2% = invisible), `bpm` times a minute.
    It brightens R, G and B equally, like real lighting does."""
    t = np.arange(rgb.shape[1]) / FPS
    return rgb * (1 + amount * np.sin(2 * np.pi * bpm / 60 * t))[None, :, None]


def add_noise(rgb, level=0.3, seed=0):
    """Camera noise: each frame's average skin color jitters randomly by about
    `level` brightness steps (the pulse itself is only ~0.2 steps)."""
    rng = np.random.default_rng(seed)
    return rgb + rng.normal(0, level, rgb.shape)


def add_movement(rgb, amount=0.03, seed=0):
    """A head turn or passing shadow: a ~1 s brightness bump of `amount` (3%) at a
    random moment in each window, hitting all three colors together -- like the
    bump we saw in subject_11."""
    rng = np.random.default_rng(seed)
    t = np.arange(rgb.shape[1]) / FPS
    centers = rng.uniform(2, 8, len(rgb))
    bump = amount * np.exp(-0.5 * ((t[None, :] - centers[:, None]) / 0.3) ** 2)
    return rgb * (1 + bump)[:, :, None]


def green_bpm(rgb):
    """Green + FFT, one window at a time."""
    return np.array([sp.hr_fft(sp.green_pulse(w, FPS), FPS) for w in rgb])


def pos_bpm(rgb):
    """POS + FFT, one window at a time."""
    return np.array([sp.hr_fft(sp.pos_pulse(w, FPS), FPS) for w in rgb])


def cnn_bpm(versions, clean, y, subjects, make_model):
    """Leave-One-Subject-Out for the CNN, tested on every version at once.

    versions: dict of name -> (n, 300, 3) windows (clean and damaged copies).
    For each person, ONE network is trained on everyone else's CLEAN windows, then
    asked about every version of that person's windows. Training once per person
    (instead of once per version) keeps it fair and 4x faster.
    """
    out = {name: np.zeros(len(y)) for name in versions}
    people = np.unique(subjects)
    for k, (s, train, test) in enumerate(evaluate.loso_splits(subjects), 1):
        model = make_model().fit(clean[train], y[train])
        for name, X in versions.items():
            out[name][test] = model.predict(X[test])
        print(f"  [{k}/{len(people)}] {s} done")
    return out
