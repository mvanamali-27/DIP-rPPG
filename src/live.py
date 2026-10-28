"""Phase 4: live heart-rate demo. Run from the repo root, on a laptop:

    ~/venvs/rppg/bin/python src/live.py                      # your webcam
    ~/venvs/rppg/bin/python src/live.py --video some.mp4     # a video file instead

L1: show the camera with the face's skin regions drawn on it.
L2: measure the heart rate live (POS over the last 10 seconds). Press q to quit.
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
# OpenCV draws in BGR order, so these are (blue, green, red)
REGION_COLORS = {"forehead": (0, 0, 255), "cheek_left": (0, 200, 0), "cheek_right": (255, 120, 0)}


class PulseMeter:
    """Collects the skin color of every frame and turns the last 10 s into a heart rate."""

    def __init__(self):
        self.times, self.colors = deque(), deque()
        self.recent = deque(maxlen=3)              # the last few readings, for a steadier display
        self.pulse = None                          # the latest pulse wave (for drawing in L3)

    def add(self, t, rgb):
        self.times.append(t)
        self.colors.append(rgb)
        while self.times[0] < t - WINDOW_SEC - 1:  # keep only the last ~11 seconds
            self.times.popleft()
            self.colors.popleft()

    def seconds_collected(self):
        return self.times[-1] - self.times[0] if len(self.times) > 1 else 0.0

    def estimate(self):
        """Heart rate (bpm) over the last 10 s, or None while still warming up."""
        if self.seconds_collected() < WINDOW_SEC:
            return None
        t, c = np.array(self.times), np.array(self.colors)
        # webcams don't deliver frames evenly, so re-draw the colors on an exact 30-per-second grid
        grid = np.arange(t[-1] - WINDOW_SEC, t[-1], 1 / FPS)
        rgb = np.stack([np.interp(grid, t, c[:, k]) for k in range(3)], axis=1)
        self.pulse = sp.pos_pulse(rgb, FPS)
        self.recent.append(sp.hr_fft(self.pulse, FPS))
        return float(np.median(self.recent))


def skin_color(frame_rgb, pts):
    """Average R, G, B of all clean skin pixels (forehead + cheeks), or None."""
    masks = roi.skin_masks(frame_rgb, roi.roi_masks(pts, frame_rgb.shape))
    skin = masks["forehead"] | masks["cheek_left"] | masks["cheek_right"]
    return frame_rgb[skin].mean(axis=0) if skin.any() else None


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
        shown = frame_bgr.copy()
        if pts is None:
            cv2.putText(shown, "no face", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            for name, idx in roi.ROI_LANDMARKS.items():
                cv2.polylines(shown, [cv2.convexHull(pts[idx])], True, REGION_COLORS[name], 2)
            color = skin_color(frame_rgb, pts)
            if color is not None:
                meter.add(t, color)

        if t - last_estimate >= 1.0 and meter.times:  # a new reading once per second
            last_estimate = t
            bpm = meter.estimate()
            if bpm is not None:
                print(f"{t:6.1f} s   {bpm:5.1f} bpm", flush=True)

        if bpm is None:
            wait = max(0, WINDOW_SEC - meter.seconds_collected()) if meter.times else WINDOW_SEC
            cv2.putText(shown, f"measuring... {wait:.0f} s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.putText(shown, f"{bpm:.0f} bpm", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
        cv2.putText(shown, f"{n / max(t, 1e-6):4.1f} fps", (20, shown.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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
