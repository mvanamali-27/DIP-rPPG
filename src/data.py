"""Find subject folders and load their video path + ground truth."""
from pathlib import Path
import numpy as np
import config


def find_subjects(data_dir=None):
    """List (subject_id, video_path, gt_path) for every subject folder.

    Filenames are inconsistent (3.mp4, ground_truth3.txt, ground_truth_15.txt,
    a 'sibject_16' typo), so we glob for 'an mp4' and 'a txt' instead of
    hardcoding names.
    """
    data_dir = Path(data_dir or config.DATA_DIR)
    subjects = []
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        videos = list(folder.glob("*.mp4"))
        gts = list(folder.glob("*.txt"))
        if not videos or not gts:
            print(f"  skip {folder.name}: {len(videos)} mp4, {len(gts)} txt")
            continue
        subjects.append((folder.name, videos[0], gts[0]))
    return subjects


def load_ground_truth(gt_path):
    """Parse a UBFC 'DATASET_2' ground-truth file into 3 arrays.

    The file is exactly 3 rows of numbers:
        row 0 -> PPG pulse waveform (the reference signal)
        row 1 -> heart rate in bpm at each sample
        row 2 -> timestamp in seconds at each sample
    """
    rows = np.loadtxt(gt_path)                 # shape (3, N)
    assert rows.shape[0] == 3, f"expected 3 rows, got {rows.shape}"
    return rows[0], rows[1], rows[2]           # ppg, hr_bpm, time_s
