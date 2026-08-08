# RSNA Knee Abnormality Detection — Lab Notebook

Full pipeline for the $77K Kaggle competition.

## Status
- ✅ **Data mined**: 4,407 studies fully labeled (58 official + 4,345 LLM-mined) → [`rsna_labels_full.csv`](rsna_labels_full.csv)
- ✅ Mining validated: 88% label agreement, clinically sane prevalence
- 🔄 **Phase 2A — text model**: [`rsna_text_model.py`](rsna_text_model.py) (XLM-R, multilingual, 12-head BCE)
- ⏳ Phase 2B — image model (DICOM → 2D slices, trained on mined soft labels)
- ⏳ Phase 3 — multimodal fusion + ensemble

## Quick start (Kaggle)
1. Create notebook → import `rsna_text_model.py`
2. Accelerator: GPU T4 x2, Internet ON
3. Run — ~20 min, outputs `text_preds_train.csv` + model

## Key facts
- 4,407 studies · 12 binary findings · mean AUC metric
- Only 1.3% officially labeled → the competition IS report label-mining
- Reports in EN (59%) / DE (19%) / FR (17%) / RU (5%) / ES (0.5%)
- Test set currently 3-study placeholder, no reports → image model carries submission
