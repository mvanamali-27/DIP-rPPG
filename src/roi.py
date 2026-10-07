"""Video -> per-frame RGB trace: read frames, find the face, average skin color.

This version is step 2a only: reading the video. Face detection (2b) and
color averaging (2c) get added to this same file next.
"""
import cv2


def video_info(video_path):
    """Basic facts from the video's header: fps, frame count, size, duration."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "n_frames": n, "width": w, "height": h,
            "duration_s": n / fps if fps else None}


def iter_frames(video_path):
    """Yield the video's frames one at a time, as RGB arrays of shape (H, W, 3).

    Why one at a time: one 640x480 frame is ~0.9 MB, so a 2,000-frame video held
    all at once is ~1.8 GB of RAM. A generator (`yield`) keeps only the current
    frame in memory.

    Why the color flip: OpenCV decodes frames as BGR (blue first). We convert to
    RGB here, once, so nothing downstream has to remember.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"could not open {video_path}")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:                      # no more frames -> end of video
                break
            yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()                       # always close the file, even on error
