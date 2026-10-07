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



# ---------------------------------------------------------- 2e: pulse -> bpm
def hr_fft(window, fps, band=config.BAND_HZ, n_fft=4096):
    """Heart rate (bpm) of one window: the frequency of the tallest peak in its
    spectrum, searching only the human pulse band.

    Two standard tricks:
      - np.hanning fades the window's edges in and out, so the hard cut at its
        ends doesn't create fake frequencies.
      - n_fft pads with zeros. A 10 s window can only separate frequencies 0.1 Hz
        (6 bpm) apart; padding lets us read the peak's position more finely.
    """
    x = (window - window.mean()) * np.hanning(len(window))
    n = max(n_fft, len(x))
    freqs = np.fft.rfftfreq(n, d=1 / fps)
    power = np.abs(np.fft.rfft(x, n=n)) ** 2
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    return 60 * freqs[in_band][np.argmax(power[in_band])]


def hr_per_window(pulse, fps, window_sec=config.WINDOW_SEC,
                  stride_sec=config.STRIDE_SEC, margin_sec=1):
    """Slide a window along a pulse wave and measure the heart rate in each one.

    margin_sec skips the first and last second, where the filter's edge effect is.
    Returns a dict of two arrays, one entry per window:
        t    -- time of the window's center (s)
        bpm  -- heart rate measured in that window
    """
    win = int(round(window_sec * fps))
    hop = int(round(stride_sec * fps))
    margin = int(round(margin_sec * fps))
    starts = np.arange(margin, len(pulse) - win - margin + 1, hop)
    return {"t": (starts + win / 2) / fps,
            "bpm": np.array([hr_fft(pulse[s:s + win], fps) for s in starts])}


def true_hr_per_window(ppg, fps, **window_args):
    """The REAL heart rate in each window, measured from the finger clip's pulse
    WAVEFORM with exactly the same method we use on the video.

    Why not the heart-rate row of the ground-truth file? On several subjects that
    number is capped at 127 bpm and turns into garbage (values like 1) whenever the
    real rate goes above it. The waveform itself is fine, so we measure from it.
    """
    return hr_per_window(bandpass(ppg - np.mean(ppg), fps), fps, **window_args)



# ------------------------------------------------- 2f: POS (a smarter pulse)
def pos_pulse(rgb, fps, window_sec=1.6):
    """POS rPPG method (Wang et al., 2017, "Algorithmic Principles of Remote PPG").

    Green-only struggles when lighting changes or the head moves, because those
    change all three colors at once. POS uses R, G and B together:
      1. In short windows (1.6 s -- at least one heartbeat, too short for the
         lighting to drift much), express each color as relative change.
      2. Mix the three colors into two signals with fixed weights:
             S1 = G - B          S2 = G + B - 2R
         Each row of weights adds up to zero, so a plain brightness change
         (which scales R, G and B equally) cancels out completely.
      3. Combine them as S1 + (std S1 / std S2) * S2. Motion and lighting noise
         show up in both signals in a matching way, so scaling them to the same
         size and adding cancels most of it, while the pulse survives.
      4. Overlap-add the short windows into one long signal, bandpass it, and
         flip it so it points the same way as the finger clip (like green_pulse).
    Returns a 1-D pulse of length T, the same shape as green_pulse.
    """
    rgb = fill_gaps(rgb)
    T = len(rgb)
    win = int(round(window_sec * fps))
    P = np.array([[0, 1, -1],
                  [-2, 1, 1]])
    pulse = np.zeros(T)
    for start in range(T - win + 1):
        C = rgb[start:start + win].T                      # shape (3, win)
        Cn = C / C.mean(axis=1, keepdims=True)            # relative change per color
        S1, S2 = P @ Cn                                   # two mixed signals
        h = S1 + (S1.std() / (S2.std() + 1e-12)) * S2
        pulse[start:start + win] += h - h.mean()          # overlap-add
    return -bandpass(pulse, fps)
