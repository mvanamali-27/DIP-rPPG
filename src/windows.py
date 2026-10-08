"""Step 3.1 (shared by both models): turn every subject's saved signals into ONE
dataset of 10-second windows, each labelled with its true heart rate.

Both models (and the classical baselines) use exactly these windows, so their
scores are directly comparable.
"""
import numpy as np

import config
import data
import signal_proc as sp

FPS_OUT = 30.0                                  # every signal is resampled to exactly 30 frames/s
WINDOW_LEN = int(config.WINDOW_SEC * FPS_OUT)   # so every window is exactly 300 samples long
DATASET_PATH = config.DATA_DIR / "dataset.npz"  # saved in the shared Drive folder


def resample(signal, fps_in, fps_out=FPS_OUT):
    """Re-draw a signal at a new frame rate (e.g. 29.79 -> 30 frames/s) by drawing
    straight lines between neighbouring samples. Works on shape (T,) or (T, 3)."""
    signal = np.asarray(signal, dtype=float)
    t_in = np.arange(len(signal)) / fps_in
    t_out = np.arange(0, t_in[-1], 1 / fps_out)
    if signal.ndim == 1:
        return np.interp(t_out, t_in, signal)
    return np.stack([np.interp(t_out, t_in, signal[:, c])
                     for c in range(signal.shape[1])], axis=1)


def window_starts(n_samples, fps=FPS_OUT, window_sec=config.WINDOW_SEC,
                  stride_sec=config.STRIDE_SEC, margin_sec=1):
    """Where each window starts: the same windows signal_proc.hr_per_window uses
    (10 s long, 1 s apart, skipping the first and last second)."""
    win = int(round(window_sec * fps))
    hop = int(round(stride_sec * fps))
    margin = int(round(margin_sec * fps))
    return np.arange(margin, n_samples - win - margin + 1, hop)


def subject_windows(signals):
    """One subject's saved signals (from extract_all.load) -> per-window arrays."""
    fps = signals["fps"]
    rgb = resample(signals["rgb"], fps)
    pulse = resample(signals["pulse"], fps)
    truth_wave = sp.bandpass(resample(signals["ppg"], fps) - signals["ppg"].mean(), FPS_OUT)
    green_wave = sp.green_pulse(rgb, FPS_OUT)

    starts = window_starts(len(rgb))
    cut = lambda x: np.stack([x[s:s + WINDOW_LEN] for s in starts])
    bpm_of = lambda waves: np.array([sp.hr_fft(w, FPS_OUT) for w in waves])
    return {
        "rgb": cut(rgb).astype(np.float32),        # (n, 300, 3) raw skin colors -> the CNN's input
        "pulse": cut(pulse).astype(np.float32),    # (n, 300) POS pulse -> the Random Forest's input
        "bpm": bpm_of(cut(truth_wave)),            # (n,) TRUE heart rate = the label
        "bpm_pos": bpm_of(cut(pulse)),             # (n,) classical baseline: POS + FFT
        "bpm_green": bpm_of(cut(green_wave)),      # (n,) classical baseline: green + FFT
        "t": (starts + WINDOW_LEN / 2) / FPS_OUT,  # (n,) time of each window's center (s)
    }


def build_dataset(path=None):
    """Window every subject that has a signals.npz, stack them all, and save.

    'subject' records which person each window came from -- that's what lets us
    test on people the model has never seen (Leave-One-Subject-Out).
    """
    import extract_all   # imported here, so loading the dataset never needs MediaPipe
    parts = []
    for sid, video, gt in data.find_subjects():
        if not extract_all.cache_path(video).exists():
            print(f"{sid}: no signals.npz yet, skipping")
            continue
        w = subject_windows(extract_all.load(video))
        w["subject"] = np.array([sid] * len(w["bpm"]))
        parts.append(w)
    dataset = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    np.savez(path or DATASET_PATH, **dataset)
    print(f"saved {len(dataset['bpm'])} windows from {len(parts)} subjects -> {path or DATASET_PATH}")
    return dataset


def load_dataset(path=None):
    """The saved dataset as a dict of arrays. Needs only numpy -- no MediaPipe."""
    with np.load(path or DATASET_PATH) as z:
        return {key: z[key] for key in z.files}
