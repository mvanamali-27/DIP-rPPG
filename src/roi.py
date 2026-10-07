"""Video -> per-frame RGB trace: read frames, find the face, average skin color.

Step 2a: reading the video.   Step 2b: finding the face + skin regions.
Step 2c (next): averaging the skin color in every frame.
"""
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


# ---------------------------------------------------------------- 2a: reading


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


# ---------------------------------------------------- 2b: face + skin regions
# MediaPipe's face model finds 478 numbered points ("landmarks") on a face.
# It's downloaded once per Colab session (~3.7 MB).
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = Path("face_landmarker.task")

# Landmark numbers outlining each skin region. The numbering is fixed (point 1 is
# always the nose tip), so these mean the same patch of skin on every face.
# Checked on several UBFC subjects: clear of glasses, eyebrows, stubble, nose.
ROI_LANDMARKS = {
    "forehead":    [107, 66, 69, 109, 10, 338, 299, 296, 336, 9],
    "cheek_left":  [50, 36, 203, 206, 207, 187],
    "cheek_right": [280, 266, 423, 426, 427, 411],
}


def make_landmarker(video_mode=False):
    """Create a MediaPipe face landmarker, downloading its model file if needed.

    video_mode=True tracks the face from frame to frame (faster and steadier for
    a whole video), and every call then needs a timestamp. False = single images.
    """
    if not MODEL_PATH.exists():
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    mode = vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE
    options = vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mode,
        num_faces=1)
    return vision.FaceLandmarker.create_from_options(options)


def face_landmarks(landmarker, frame_rgb, timestamp_ms=None):
    """The face's 478 landmarks in pixel coordinates, shape (478, 2), or None if
    no face was found. Pass timestamp_ms only to a video-mode landmarker."""
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    if timestamp_ms is None:
        result = landmarker.detect(image)
    else:
        result = landmarker.detect_for_video(image, timestamp_ms)
    if not result.face_landmarks:
        return None
    h, w = frame_rgb.shape[:2]
    # MediaPipe gives positions as fractions of the image (0..1): convert to pixels.
    return np.array([(p.x * w, p.y * h) for p in result.face_landmarks[0]],
                    dtype=np.int32)


def roi_masks(landmarks, frame_shape):
    """One True/False mask per skin region, the same size as the frame.

    Each region is the convex hull of its landmarks -- the tightest "rubber band"
    around those points -- so the order of the points doesn't matter.
    """
    h, w = frame_shape[:2]
    masks = {}
    for name, idx in ROI_LANDMARKS.items():
        outline = cv2.convexHull(landmarks[idx])
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [outline], 1)
        masks[name] = mask.astype(bool)
    return masks
