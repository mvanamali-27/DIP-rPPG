"""Step 2g: run the slow image processing ONCE per subject and save the result
next to each video, so nobody ever has to process the videos again.

Each subject folder gets a `signals.npz` holding:
    rgb               (T, 3)  mean skin R, G, B per frame, all regions pooled
    rgb_forehead      (T, 3)  \
    rgb_cheek_left    (T, 3)   > the same, for each region separately
    rgb_cheek_right   (T, 3)  /
    pulse             (T,)    POS pulse wave (step 2f), clean and bandpassed
    ppg               (T,)    the finger clip's waveform = the ground truth
    t                 (T,)    time of each frame, in seconds
    fps               ()      the video's frame rate
The device's heart-rate row is deliberately NOT saved: it's capped at 127 bpm.
"""
import os
import time
from pathlib import Path

import numpy as np

import data
import signal_proc as sp

CACHE_NAME = "signals.npz"


def cache_path(video_path):
    """Where a subject's saved signals live: right next to its video."""
    return Path(video_path).parent / CACHE_NAME


def process_subject(video_path, gt_path):
    """One video + its ground truth -> dict of arrays. This is the slow part."""
    import roi   # MediaPipe lives in roi, so it's only needed when processing videos
    traces, fps = roi.extract_rgb(video_path)
    ppg, _, _ = data.load_ground_truth(gt_path)          # device HR row: not used
    T = min(len(traces["all"]), len(ppg))                # trim both to the same length
    rgb = sp.fill_gaps(traces["all"][:T])
    return {
        "rgb": rgb,
        "rgb_forehead": sp.fill_gaps(traces["forehead"][:T]),
        "rgb_cheek_left": sp.fill_gaps(traces["cheek_left"][:T]),
        "rgb_cheek_right": sp.fill_gaps(traces["cheek_right"][:T]),
        "pulse": sp.pos_pulse(rgb, fps),
        "ppg": ppg[:T],
        "t": np.arange(T) / fps,
        "fps": fps,
    }


def build_all(data_dir=None, overwrite=False):
    """Process every subject that isn't saved yet.

    Safe to re-run: finished subjects are skipped, so if Colab disconnects halfway
    you just run it again. Each file is written under a temporary name and renamed
    only when complete, so a half-written file can never look finished.
    """
    subjects = data.find_subjects(data_dir)
    for i, (sid, video, gt) in enumerate(subjects, 1):
        out = cache_path(video)
        if out.exists() and not overwrite:
            print(f"[{i}/{len(subjects)}] {sid}: already saved, skipping")
            continue
        t0 = time.time()
        partial = out.with_name("signals.partial.npz")
        np.savez(partial, **process_subject(video, gt))
        os.replace(partial, out)
        print(f"[{i}/{len(subjects)}] {sid}: saved ({time.time() - t0:.0f} s)")


def load(video_path):
    """One subject's saved signals, as a dict of arrays (fps as a plain number)."""
    with np.load(cache_path(video_path)) as z:
        signals = {key: z[key] for key in z.files}
    signals["fps"] = float(signals["fps"])
    return signals
