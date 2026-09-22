# dip-rppg — Contactless Heart-Rate Estimation (rPPG)

DIP course project: estimate heart rate from webcam video (no contact) using
remote photoplethysmography (rPPG) plus a small neural network.
Dataset: **UBFC-rPPG**. Goal: decent accuracy + understanding the whole pipeline.

## How the project is organized
- `src/` — the real code, as small importable modules (`config`, `data`, ...).
- `notebooks/` — thin, visual notebooks (one per person) that import from `src/`.
- `data/` — videos live here locally; **git-ignored** (they live in shared Drive).
- `artifacts/` — cached signals + model weights; **git-ignored**.

## Setup (each teammate, once)
**Data (Google Drive):** get Editor access to the shared `DIP_rPPG_data` folder,
then in Drive → *Shared with me* → right-click the folder → *Organize → Add
shortcut to Drive → My Drive*. It then appears at
`/content/drive/MyDrive/DIP_rPPG_data` for everyone.

**Code (Colab):** every notebook starts with these two cells:
```python
from google.colab import drive
drive.mount('/content/drive')
```
```python
!git clone https://github.com/OWNER/dip-rppg.git      # <- replace OWNER
import sys; sys.path.append("dip-rppg/src")
import config, data
```
Grab teammates' latest changes with `!cd dip-rppg && git pull`.

## Quick check (loads all subjects)
```python
subs = data.find_subjects()
for sid, vid, gt in subs:
    ppg, hr, t = data.load_ground_truth(gt)
    print(sid, len(t), "samples", f"{hr.mean():.1f} bpm")
```

## Team rules
1. **One notebook per person** (avoids merge conflicts).
2. **Shared logic goes in `src/*.py`**, not copied between notebooks.
3. **`git pull` before you start; commit small.**

## Roadmap
1. Load & explore ✅
2. Video → pulse signal (face ROI → RGB)
3a. Classical FFT baseline (bandpass + FFT, POS/CHROM)
3b. 1D-CNN, evaluated Leave-One-Subject-Out
4. Live webcam demo (stretch)
5. Report (MAE/RMSE, plots)
