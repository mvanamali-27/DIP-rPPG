"""Signal processing: turn the raw RGB skin trace into a clean pulse wave.

Step 2d: fill gaps, normalize, bandpass filter.   Step 2e (next): pulse -> bpm.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt

import config


def fill_gaps(trace):
    """Fill NaN rows (frames where no face was found) by drawing a straight line
    between the neighbouring good frames. Works on shape (T,) or (T, 3)."""
    trace = np.array(trace, dtype=float)              # a copy: never change the input
    flat = trace.reshape(len(trace), -1)
    t = np.arange(len(flat))
    for c in range(flat.shape[1]):
        bad = np.isnan(flat[:, c])
        if bad.any() and not bad.all():
            flat[bad, c] = np.interp(t[bad], t[~bad], flat[~bad, c])
    return flat.reshape(trace.shape)


def normalize(trace):
    """Express each channel as relative change around its own average: x / mean - 1.

    0.003 means "this frame is 0.3% brighter than this person's average". That
    removes overall brightness and skin tone, so every video is on the same scale.
    """
    return trace / np.mean(trace, axis=0) - 1


def bandpass(signal, fps, band=config.BAND_HZ, order=3):
    """Keep only the speeds a human pulse can have: 0.7-4 Hz = 42-240 bpm by default.

    Anything slower (lighting drift, slow head motion) or faster (camera and
    compression noise) is removed. It's a Butterworth filter (smooth, no ripples),
    and sosfiltfilt runs it forward and then backward so the wave is NOT shifted
    in time -- every peak stays exactly where it was.
    """
    sos = butter(order, band, btype="bandpass", fs=fps, output="sos")
    return sosfiltfilt(sos, signal, axis=0)


def green_pulse(rgb, fps):
    """Simplest rPPG method: the pulse is the flipped green channel.

    rgb: array (T, 3) of mean skin R, G, B per frame. Returns a 1-D pulse of length T.
    Flipped because more blood absorbs more green, so less green reaches the camera.
    """
    rgb = fill_gaps(rgb)
    green = -normalize(rgb)[:, 1]
    return bandpass(green, fps)
