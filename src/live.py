"""Phase 4: live heart-rate demo. Run from the repo root, on a laptop:

    ~/venvs/rppg/bin/python src/live.py                      # your webcam
    ~/venvs/rppg/bin/python src/live.py --video some.mp4     # a video file instead

Step L1: show the camera with the face's skin regions drawn on it. Press q to quit.
"""
import argparse
import time

import cv2
import numpy as np

import roi

# OpenCV draws in BGR order, so these are (blue, green, red)
REGION_COLORS = {"forehead": (0, 0, 255), "cheek_left": (0, 200, 0), "cheek_right": (255, 120, 0)}


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

    start, last_ms, n, shown = time.monotonic(), -1, 0, None
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
