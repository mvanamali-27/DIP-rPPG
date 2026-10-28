"""Phase 4: live heart-rate demo. Run from the repo root, on a laptop:

    ~/venvs/rppg/bin/python src/live.py                      # your webcam
    ~/venvs/rppg/bin/python src/live.py --video some.mp4     # a video file instead

L1: show the camera with the face's skin regions drawn on it.
L2: measure the heart rate live (POS over the last 10 seconds). Press q to quit.
L3: a side panel with the big bpm, a warm-up bar and the pulse wave scrolling by.
"""
import argparse
import time
from collections import deque

import cv2
import numpy as np

import roi
import signal_proc as sp

WINDOW_SEC = 10        # heart rate is measured over the last 10 seconds, like the dataset
FPS = 30.0             # the buffer is re-drawn at exactly 30 frames/s, like the dataset
WAVE_SEC = 6           # how many seconds of pulse wave the panel shows
PANEL_W = 320          # width of the side panel, in pixels
# OpenCV draws in BGR order, so these are (blue, green, red)
REGION_COLORS = {"forehead": (0, 0, 255), "cheek_left": (0, 200, 0), "cheek_right": (255, 120, 0)}
WHITE, GREY, PULSE_COLOR = (255, 255, 255), (140, 140, 140), (80, 80, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class PulseMeter:
    """Collects the skin color of every frame and turns the last 10 s into a heart rate."""

    def __init__(self):
        self.times, self.colors = deque(), deque()
        self.recent = deque(maxlen=3)              # the last few readings, for a steadier display

    def add(self, t, rgb):
        self.times.append(t)
        self.colors.append(rgb)
        while self.times[0] < t - WINDOW_SEC - 1:  # keep only the last ~11 seconds
            self.times.popleft()
            self.colors.popleft()

    def seconds_collected(self):
        return self.times[-1] - self.times[0] if len(self.times) > 1 else 0.0

    def pulse(self, seconds):
        """POS pulse wave of the last `seconds`, on an exact 30-per-second grid."""
        t, c = np.array(self.times), np.array(self.colors)
        # webcams don't deliver frames evenly, so re-draw the colors on an exact 30-per-second grid
        grid = np.arange(t[-1] - seconds, t[-1], 1 / FPS)
        rgb = np.stack([np.interp(grid, t, c[:, k]) for k in range(3)], axis=1)
        return sp.pos_pulse(rgb, FPS)

    def estimate(self):
        """Heart rate (bpm) over the last 10 s, or None while still warming up."""
        if self.seconds_collected() < WINDOW_SEC:
            return None
        self.recent.append(sp.hr_fft(self.pulse(WINDOW_SEC), FPS))
        return float(np.median(self.recent))

    def wave(self):
        """The last few seconds of pulse, for drawing (None until there is enough).
        The newest second is left out: the filter and POS need a little data on BOTH
        sides of a moment, so the wave's very end is still squashed flat."""
        have = self.seconds_collected()
        if have < WAVE_SEC + 1:
            return None
        p = self.pulse(min(have, WINDOW_SEC))
        return p[-int((WAVE_SEC + 1) * FPS):-int(FPS)]


def skin_color(frame_rgb, pts):
    """Average R, G, B of all clean skin pixels (forehead + cheeks), or None."""
    masks = roi.skin_masks(frame_rgb, roi.roi_masks(pts, frame_rgb.shape))
    skin = masks["forehead"] | masks["cheek_left"] | masks["cheek_right"]
    return frame_rgb[skin].mean(axis=0) if skin.any() else None


def draw_panel(height, bpm, meter, fps):
    """The dark side panel: big bpm (or a warm-up bar), the pulse wave, the frame rate."""
    panel = np.full((height, PANEL_W, 3), 30, np.uint8)
    cv2.putText(panel, "HEART RATE", (20, 40), FONT, 0.7, GREY, 2)

    if bpm is None:                                   # warming up: show how far along we are
        done = min(meter.seconds_collected() / WINDOW_SEC, 1.0) if meter.times else 0.0
        cv2.putText(panel, "--", (20, 120), FONT, 2.5, GREY, 5)
        cv2.rectangle(panel, (20, 150), (PANEL_W - 20, 165), GREY, 1)
        cv2.rectangle(panel, (20, 150), (20 + int(done * (PANEL_W - 40)), 165), WHITE, -1)
        cv2.putText(panel, f"measuring... {WINDOW_SEC * (1 - done):.0f} s left", (20, 190), FONT, 0.55, GREY, 1)
    else:
        cv2.putText(panel, f"{bpm:.0f}", (20, 120), FONT, 3.0, WHITE, 6)
        cv2.putText(panel, "bpm", (210, 120), FONT, 0.9, GREY, 2)

    # the pulse wave: newest on the right, so it scrolls to the left as time passes
    top, box_h = 240, 120
    cv2.putText(panel, f"pulse, last {WAVE_SEC} s", (20, top - 10), FONT, 0.5, GREY, 1)
    cv2.rectangle(panel, (20, top), (PANEL_W - 20, top + box_h), (70, 70, 70), 1)
    wave = meter.wave() if meter.times else None
    if wave is not None:
        wave = (wave - wave.min()) / (np.ptp(wave) + 1e-9)          # squeeze into 0..1
        xs = np.linspace(25, PANEL_W - 25, len(wave))
        ys = top + box_h - 10 - wave * (box_h - 20)                  # 0 at the bottom, 1 at the top
        cv2.polylines(panel, [np.stack([xs, ys], axis=1).astype(np.int32)], False, PULSE_COLOR, 2, cv2.LINE_AA)

    cv2.putText(panel, f"{fps:4.1f} fps    q = quit", (20, height - 20), FONT, 0.5, GREY, 1)
    return panel


def open_source(video_path=None):
    """The webcam (default) or a video file. Returns (capture, fps_or_None)."""
    if video_path:
        cap = cv2.VideoCapture(video_path)
        return cap, cap.get(cv2.CAP_PROP_FPS)
    cap = cv2.VideoCapture(0)                        # 0 = the built-in camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)           # same size as the UBFC videos
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap, None                                 # webcam: timing comes from the clock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="use this video file instead of the webcam")
    parser.add_argument("--snapshot", help="(testing) save the last frame here and quit")
    parser.add_argument("--max-frames", type=int, default=0, help="(testing) stop after N frames")
    args = parser.parse_args()

    cap, file_fps = open_source(args.video)
    if not cap.isOpened():
        raise SystemExit("could not open the camera -- check System Settings > Privacy & Security > Camera")
    landmarker = roi.make_landmarker(video_mode=True)
    meter = PulseMeter()

    start, last_ms, n, shown = time.monotonic(), -1, 0, None
    bpm, last_estimate = None, 0.0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        # every frame gets a time stamp: from the clock (webcam) or the frame count (file)
        t = n / file_fps if file_fps else time.monotonic() - start
        ms = max(int(t * 1000), last_ms + 1)          # MediaPipe needs strictly increasing times
        last_ms, n = ms, n + 1

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pts = roi.face_landmarks(landmarker, frame_rgb, timestamp_ms=ms)
        view = frame_bgr.copy()
        if pts is not None:
            for name, idx in roi.ROI_LANDMARKS.items():
                cv2.polylines(view, [cv2.convexHull(pts[idx])], True, REGION_COLORS[name], 2)
            color = skin_color(frame_rgb, pts)
            if color is not None:
                meter.add(t, color)
        if not file_fps:
            view = cv2.flip(view, 1)                  # webcam: mirror it, so it feels like a mirror
        if pts is None:
            cv2.putText(view, "no face", (20, 40), FONT, 1, (0, 0, 255), 2)

        if t - last_estimate >= 1.0 and meter.times:  # a new reading once per second
            last_estimate = t
            bpm = meter.estimate()
            if bpm is not None:
                print(f"{t:6.1f} s   {bpm:5.1f} bpm", flush=True)

        shown = np.hstack([view, draw_panel(view.shape[0], bpm, meter, n / max(t, 1e-6))])

        if args.snapshot or args.max_frames:
            if args.max_frames and n >= args.max_frames:
                break
            continue                                  # testing mode: no window
        cv2.imshow("rPPG live", shown)
        if cv2.waitKey(int(1000 / file_fps) if file_fps else 1) & 0xFF == ord("q"):
            break

    if args.snapshot and shown is not None:
        cv2.imwrite(args.snapshot, shown)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
