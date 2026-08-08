# RSNA Knee Abnormality — Image Model (Phase 2B)
# Per-study 12-label classification from knee MRI DICOMs.
# Strategy: sagittal fluid-sensitive series → middle slices → 2D CNN (EfficientNet-B0)
# → slice logits mean-pooled per study → BCE on mined soft labels.
# Same validation split as the text model (seed 42, 10%) for clean fusion later.
#
# How to run on Kaggle:
#   1. New Notebook → File > Import Notebook → paste this file
#   2. Add Input → search "RSNA Knee Abnormality Detection" → add the competition dataset
#   3. Settings: Accelerator = GPU T4 x2, Internet = ON
#   4. Run all. Preprocessing ~20-40 min (cached), training ~30-60 min.

# 🔴 GPU SELF-HEALING — Kaggle's GPU lottery hands out P100/K80 (sm_60) which
# the default PyTorch build can't run. Detect and install a compatible build.
import os, subprocess, sys
def _gpu_compat():
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            name = torch.cuda.get_device_name(0)
            print(f"GPU detected: {name} (sm_{cap[0]}{cap[1]})", flush=True)
            if cap[0] < 7:
                print("Old GPU — installing compatible PyTorch 2.0.1 (cu117)...", flush=True)
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                    "torch==2.0.1", "--index-url", "https://download.pytorch.org/whl/cu117"])
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                    "transformers==4.36.2", "tokenizers==0.15.2", "accelerate==0.25.0",
                    "datasets==2.16.1", "timm", "pydicom", "opencv-python-headless"])
                os.environ["PYTHONPATH"] = ""
                print("Compatible stack installed.", flush=True)
        else:
            print("No GPU — running on CPU (slow but works)", flush=True)
    except ImportError:
        pass
_gpu_compat()

import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

import pydicom
import cv2

LABELS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
          'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

# 🔴 AUTO-DETECT competition data dir (slug varies; never hardcode)
print("Mounted inputs:", os.listdir("/kaggle/input"))
_candidates = glob.glob("/kaggle/input/*/train_series.csv")
assert _candidates, ("Competition data NOT found. Add Input → 'RSNA Knee Abnormality "
                     "Detection' (trophy icon) and check it contains train_series.csv")
DATA_DIR = os.path.dirname(_candidates[0])
print("Using data dir:", DATA_DIR)
SERIES_DIR = os.path.join(DATA_DIR, "train_series")
CACHE_DIR = "/kaggle/working/img_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
N_SLICES = 12          # middle slices per study
IMG_SIZE = 224

# ── 1. LABELS (mined) ───────────────────────────────────────────
URL = "https://raw.githubusercontent.com/messaynew-cyber/rsna-kaggle/main/rsna_labels_full.csv"
df = pd.read_csv(URL)
df = df[df[LABELS].notna().all(axis=1)].reset_index(drop=True)
print("Labeled studies:", len(df))

# ── 2. SERIES SELECTION: sagittal + fluid-sensitive ─────────────
series = pd.read_csv(os.path.join(DATA_DIR, "train_series.csv"))
series = series[(series["Anatomical_Plane"] == "Sagittal") & (series["Fluid_Sensitive"] == 1)]
# one series per study (largest = most slices)
series["n_files"] = series["SeriesInstanceUID"].map(
    lambda s: len(glob.glob(os.path.join(SERIES_DIR, "*", s, "*.dcm"))))
series = series.sort_values("n_files", ascending=False).drop_duplicates("StudyInstanceUID")
print("Studies with sagittal fluid series:", len(series))

df = df[df["StudyInstanceUID"].isin(series["StudyInstanceUID"])].reset_index(drop=True)
print("Final trainable studies:", len(df))

# ── 3. SPLIT — MUST MATCH TEXT MODEL (seed 42, test 0.1, same stratify) ──
y_bin = (df[LABELS] >= 0.5).astype(int)
strat = y_bin.sum(axis=1).clip(upper=5).astype(str) + "_" + (y_bin.sum(axis=1) == 0).astype(str)
train_idx, val_idx = train_test_split(
    np.arange(len(df)), test_size=0.1, random_state=42, stratify=strat)
df["fold"] = "train"
df.loc[val_idx, "fold"] = "val"
print("train:", (df.fold == "train").sum(), "val:", (df.fold == "val").sum())

# ── 4. DICOM → SLICES (cached to /kaggle/working/img_cache) ─────
def load_slices(study_uid):
    """Return (N, 224, 224) float32 array of middle slices, 0-1 normalized."""
    cache_path = os.path.join(CACHE_DIR, f"{study_uid}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)
    sel = series[series["StudyInstanceUID"] == study_uid]
    if len(sel) == 0:
        return None
    series_uid = sel.iloc[0]["SeriesInstanceUID"]
    files = sorted(glob.glob(os.path.join(SERIES_DIR, study_uid, series_uid, "*.dcm")))
    if len(files) == 0:
        return None
    arrays = []
    for f in files:
        try:
            dcm = pydicom.dcmread(f, force=True)
            arr = dcm.pixel_array.astype(np.float32)
            # percentile normalize
            lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
            if hi - lo < 1e-3:
                continue
            arr = np.clip((arr - lo) / (hi - lo), 0, 1)
            arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            arrays.append(arr)
        except Exception:
            continue
    if len(arrays) == 0:
        return None
    arrays = np.stack(arrays)
    # middle slices
    n = len(arrays)
    mid = n // 2
    start = max(0, mid - N_SLICES // 2)
    end = min(n, mid + N_SLICES // 2)
    if end - start < N_SLICES:
        idxs = np.linspace(0, n - 1, N_SLICES).astype(int)
        out = arrays[idxs]
    else:
        out = arrays[start:end]
    np.save(cache_path, out)
    return out

# Preprocess all (parallel-ish, with progress)
valid_uids = []
for uid in tqdm(df["StudyInstanceUID"], desc="Preprocessing DICOMs"):
    sl = load_slices(uid)
    if sl is not None:
        valid_uids.append(uid)
df = df[df["StudyInstanceUID"].isin(valid_uids)].reset_index(drop=True)
print("Studies with usable slices:", len(df))
print(f"Cache size: {sum(os.path.getsize(os.path.join(CACHE_DIR,f)) for f in os.listdir(CACHE_DIR))/1e9:.2f} GB")

# ── 5. DATASET ──────────────────────────────────────────────────
class KneeDataset(Dataset):
    def __init__(self, df):
        self.uids = df["StudyInstanceUID"].values
        self.labels = df[LABELS].astype(float).values

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        slices = np.load(os.path.join(CACHE_DIR, f"{self.uids[i]}.npy"))
        # 3-channel for ImageNet pretrained
        imgs = np.repeat(slices[:, None, :, :], 3, axis=1).astype(np.float32)
        return torch.from_numpy(imgs), torch.from_numpy(self.labels[i].astype(np.float32))

tr_ds = KneeDataset(df[df.fold == "train"])
va_ds = KneeDataset(df[df.fold == "val"])
print("Dataset sizes:", len(tr_ds), len(va_ds))

# ── 6. MODEL: EfficientNet-B0 + 12 heads ────────────────────────
import timm
model = timm.create_model("tf_efficientnet_b0", pretrained=True, num_classes=12)
model = model.cuda()

# ImageNet normalization
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

def study_forward(model, imgs):
    """imgs: (S, 3, 224, 224) → mean-pooled study logits (12,)"""
    with torch.no_grad():
        imgs = (imgs - IMAGENET_MEAN) / IMAGENET_STD
        logits = []
        B = 32
        for i in range(0, len(imgs), B):
            logits.append(model(imgs[i:i+B]))
        return torch.stack(logits).mean(0)

# ── 7. TRAIN (custom loop — BCE on soft targets) ────────────────
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)
loss_fn = nn.BCEWithLogitsLoss()
EPOCHS = 5

for ep in range(EPOCHS):
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(range(len(tr_ds)), desc=f"Epoch {ep+1}/{EPOCHS}")
    for i in pbar:
        imgs, labels = tr_ds[i]
        imgs = ((imgs - IMAGENET_MEAN) / IMAGENET_STD).cuda()
        labels = labels.unsqueeze(0).cuda()
        logits = model(imgs)                      # (1, 12)
        loss = loss_fn(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item(); n += 1
        pbar.set_postfix(loss=total / n)
    sched.step()

    # validation (mean AUC)
    model.eval()
    all_logits, all_labels = [], []
    for i in range(len(va_ds)):
        imgs, labels = va_ds[i]
        logits = study_forward(model, imgs.cuda()).cpu()
        all_logits.append(logits.numpy())
        all_labels.append(labels.numpy())
    probs = 1 / (1 + np.exp(-np.array(all_logits)))
    y_true = (np.array(all_labels) >= 0.5).astype(int)
    aucs = []
    for j, l in enumerate(LABELS):
        if len(np.unique(y_true[:, j])) < 2:
            aucs.append(0.5); continue
        aucs.append(roc_auc_score(y_true[:, j], probs[:, j]))
    print(f"Epoch {ep+1} — val mean_auc: {np.mean(aucs):.4f} | " +
          ", ".join(f"{l}={a:.3f}" for l, a in zip(LABELS, aucs)))

# ── 8. SAVE PREDICTIONS ─────────────────────────────────────────
model.eval()
def predict_df(d):
    outs = []
    for i in range(len(d)):
        imgs, _ = d[i]
        logits = study_forward(model, imgs.cuda()).cpu().numpy()
        outs.append(logits)
    return np.array(outs)

tr_logits = predict_df(tr_ds)
va_logits = predict_df(va_ds)

tr_preds = pd.DataFrame(1 / (1 + np.exp(-tr_logits)), columns=LABELS)
tr_preds.insert(0, "StudyInstanceUID", df[df.fold == "train"]["StudyInstanceUID"].values)
va_preds = pd.DataFrame(1 / (1 + np.exp(-va_logits)), columns=LABELS)
va_preds.insert(0, "StudyInstanceUID", df[df.fold == "val"]["StudyInstanceUID"].values)

tr_preds.to_csv("image_preds_train.csv", index=False)
va_preds.to_csv("image_preds_val.csv", index=False)
print("✅ Saved image_preds_train.csv:", tr_preds.shape, "| image_preds_val.csv:", va_preds.shape)

torch.save(model.state_dict(), "image_model_efficientnet_b0.pt")
print("✅ Model saved: image_model_efficientnet_b0.pt")
print("\nNOTE: download image_preds_*.csv + model from the Output tab —")
print("these feed the Phase 3 fusion notebook.")
