"""Central config: paths + constants. Every module imports this."""
from pathlib import Path

# ---- where the dataset lives ----------------------------------------------
# We try a few likely locations and use the first that EXISTS, so the SAME file
# works on Colab (Drive) and locally. You can always override by passing a path
# to data.find_subjects(...). If your Drive folder is named differently, edit
# COLAB_DATA below to match.
COLAB_DATA = Path("/content/drive/MyDrive/DIP_rPPG_data/rppg dataset")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_candidates = [
    COLAB_DATA,
    PROJECT_ROOT / "data" / "rppg dataset",
    PROJECT_ROOT / "rppg dataset",
]
DATA_DIR = next((c for c in _candidates if c.exists()), COLAB_DATA)

# ---- artifacts (cached signals, model weights) — kept out of git ----------
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SIGNALS_DIR = ARTIFACTS_DIR / "signals"

# ---- signal / heart-rate constants ----------------------------------------
FPS = 30.0                  # UBFC videos are ~30 fps
HR_MIN_BPM = 42             # 0.7 Hz — lowest plausible human pulse
HR_MAX_BPM = 240            # 4.0 Hz — highest
BAND_HZ = (HR_MIN_BPM / 60.0, HR_MAX_BPM / 60.0)   # bandpass edges in Hz
WINDOW_SEC = 10             # analysis window length (seconds)
STRIDE_SEC = 1             # hop between windows (seconds)
