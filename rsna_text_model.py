# RSNA Knee Abnormality — Text Model (XLM-R)
# Phase 2A: train a multilingual text model on (radiology report → 12 mined labels)
# Handles EN/DE/FR/RU/ES natively. Output: per-study soft probabilities (text tower).
#
# How to run on Kaggle:
#   1. New Notebook → "File > Import Notebook" → paste this file (or GitHub import)
#   2. Settings: Accelerator = GPU T4 x2 (or P100), Internet = ON
#   3. Run all. ~15-25 min with T4. Saves text_preds.csv + model to /kaggle/working

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
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from datasets import Dataset

LABELS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
          'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

# ── 1. DATA ─────────────────────────────────────────────────────
# Mined labels (58 official + 4,345 LLM-mined soft labels)
URL = "https://raw.githubusercontent.com/messaynew-cyber/rsna-kaggle/main/rsna_labels_full.csv"
df = pd.read_csv(URL)
print("Loaded:", df.shape, "| labeled:", df[LABELS].notna().all(axis=1).sum())

# Optional: official train.csv from competition input (same reports, ground truth labels)
if os.path.exists("/kaggle/input/rsna-knee-abnormality-detection/train.csv"):
    official = pd.read_csv("/kaggle/input/rsna-knee-abnormality-detection/train.csv")
    print("Official train.csv found:", official.shape)

df = df[df[LABELS].notna().all(axis=1)].reset_index(drop=True)
print("Training rows:", len(df))

# Language sanity (report length)
df["rlen"] = df["Report"].astype(str).str.len()
print(df["rlen"].describe())

# ── 2. SPLIT (stratified by positive-label count) ───────────────
y_bin = (df[LABELS] >= 0.5).astype(int)
strat = y_bin.sum(axis=1).clip(upper=5).astype(str) + "_" + (y_bin.sum(axis=1) == 0).astype(str)
train_idx, val_idx = train_test_split(
    np.arange(len(df)), test_size=0.1, random_state=42, stratify=strat)
tr, va = df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)
print(f"train={len(tr)} val={len(va)}")

# ── 3. TOKENIZE ─────────────────────────────────────────────────
MODEL_NAME = "xlm-roberta-base"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def prep(d):
    enc = tok(d["Report"].tolist(), padding=True, truncation=True, max_length=384)
    enc["labels"] = d[LABELS].astype(float).values.tolist()
    return enc

tr_ds = Dataset.from_dict(prep(tr))
va_ds = Dataset.from_dict(prep(va))
collator = DataCollatorWithPadding(tokenizer=tok)

# ── 4. MODEL ────────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(LABELS), problem_type="multi_label_classification")

# ── 5. METRICS (mean AUC — matches competition metric) ───────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    y_true = (labels >= 0.5).astype(int)  # threshold soft targets for AUC
    aucs = []
    for i, l in enumerate(LABELS):
        if len(np.unique(y_true[:, i])) < 2:
            aucs.append(0.5)  # single class in fold
            continue
        aucs.append(roc_auc_score(y_true[:, i], probs[:, i]))
    return {"mean_auc": float(np.mean(aucs)),
            **{f"auc_{l}": round(a, 4) for l, a in zip(LABELS, aucs)}}


# 🔴 CUSTOM TRAINER — explicit BCEWithLogitsLoss.
# HF's built-in multi-label path silently breaks with soft float targets
# (constant predictions → AUC 0.5). This computes the loss directly.
class SoftTargetTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        return (loss, outputs) if return_outputs else loss

args = TrainingArguments(
    output_dir="./text_model",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=True,
    max_grad_norm=1.0,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="mean_auc",
    report_to=[],
)

trainer = SoftTargetTrainer(
    model=model,
    args=args,
    train_dataset=tr_ds,
    eval_dataset=va_ds,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

# ── 6. TRAIN ────────────────────────────────────────────────────
trainer.train()
print("\n=== FINAL VALIDATION ===")
print(trainer.evaluate())

# 🔴 Sanity check: prediction variance must be > 0 (constant outputs = broken)
import numpy as _np
_sanity = _np.std(trainer.predict(va_ds).predictions)
print(f"Sanity — pred logits std: {_sanity:.4f} (must be clearly > 0)")

# ── 7. PREDICT ALL TRAIN + TEST ─────────────────────────────────
def predict(df_in):
    ds = Dataset.from_dict(tok(df_in["Report"].tolist(), padding=True,
                                truncation=True, max_length=384))
    out = trainer.predict(ds)
    probs = 1 / (1 + np.exp(-out.predictions))
    res = pd.DataFrame(probs, columns=LABELS)
    res.insert(0, "StudyInstanceUID", df_in["StudyInstanceUID"].values)
    return res

train_preds = predict(df)
train_preds.to_csv("text_preds_train.csv", index=False)
print("Saved text_preds_train.csv:", train_preds.shape)

# Test set — if it has reports, predict; else note it
test_path = "/kaggle/input/rsna-knee-abnormality-detection/test.csv"
if os.path.exists(test_path):
    te = pd.read_csv(test_path)
    if "Report" in te.columns:
        test_preds = predict(te)
        test_preds.to_csv("text_preds_test.csv", index=False)
        print("Saved text_preds_test.csv:", test_preds.shape)
    else:
        print("⚠️ test.csv has no Report column — text predictions not possible on test (image model will carry submission)")

# ── 8. SAVE MODEL ───────────────────────────────────────────────
trainer.save_model("./text_model_final")
tok.save_pretrained("./text_model_final")
print("✅ Model saved to ./text_model_final (download from Output tab)")
