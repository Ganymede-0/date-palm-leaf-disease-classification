# Date Palm Leaf Disease Classification — Swin Transformer vs ConvNeXt-Tiny

A computer vision project that fine-tunes two modern image classifiers — **Swin Transformer (`swin_tiny_patch4_window7_224`)** and **ConvNeXt-Tiny (`convnext_tiny`)** — to identify diseases on date palm (*Phoenix dactylifera*) leaves from RGB images.

The project also implements a **leakage-aware data split** built on top of self-supervised image embeddings, so visually near-duplicate leaf images cannot be split across train / validation / test.
 
-----

## Motivation

Date palm cultivation is a major agricultural sector in Saudi Arabia and the wider MENA region. Several common leaf disorders — fungal infections, pest damage, and nutritional deficiencies — visibly affect the leaves long before fruit yield is impacted. An accurate image-based classifier can support early field-level diagnosis using nothing more than a smartphone photo.

This project explores two complementary architectures:

- **Swin Transformer** — hierarchical vision transformer with shifted-window self-attention.
- **ConvNeXt-Tiny** — a pure-convolutional network modernised with transformer-era design choices.

Both are trained under the same data pipeline so their results are directly comparable.

---

## Dataset

Link: https://data.mendeley.com/datasets/g684ghfxvg/2

This work uses the publicly published **Infected Date Palm Leaves Dataset** (Namoun et al., 2024 — *Data in Brief*).

- **Classes (9):** potassium deficiency, manganese deficiency, magnesium deficiency, black scorch, leaf spots, fusarium wilt, rachis blight, *Parlatoria blanchardi* (pest), and healthy.
- **Origin:** 10 date farms in the Madinah region, Saudi Arabia.
- **Size used here:** the *Processed* subset of the dataset (~3,089 images after cropping/augmentation by the dataset authors).
- **Image formats supported:** `.jpg`, `.jpeg`, `.png`, `.heic` / `.HEIC` (HEIC is decoded via `pillow-heif`, which matters because many of the source photos came from iPhones).

---

## Method

The full pipeline lives across the two notebooks. At a high level:

**1. Build the file index**
Recursively scan the `Processed/` folder; each immediate subfolder name becomes a class label. Produce a DataFrame with columns `path`, `label`, `filename`.

**2. Extract DINOv2 embeddings**
Use a self-supervised vision transformer (`vit_small_patch14_dinov2.lvd142m` via `timm`, no classification head) to produce a feature vector for every image at 518×518.

**3. Group near-duplicates per class**
For each class, compute pairwise cosine similarity between embeddings and assign images to embedding groups using a **0.95 similarity threshold**. This catches augmented variants of the same source photo, slight crops, or near-identical shots.

**4. Group-safe split**
Apply `sklearn.model_selection.GroupShuffleSplit` twice, using the embedding-group ID as the grouping key:
- 70% train / 30% temp
- temp → 50/50 → 15% validation / 15% test

A leakage check confirms zero group overlap between any two splits. The resulting CSVs (`train_processed_embedding_split.csv`, etc.) are saved so both training notebooks consume **identical splits**.

**5. Training**
Both models are fine-tuned from ImageNet-pretrained weights via `timm`, with:

| Setting              | Value                                                                  |
|----------------------|------------------------------------------------------------------------|
| Image size           | 224 × 224                                                              |
| Normalisation        | ImageNet mean / std                                                    |
| Train augmentations  | `RandomResizedCrop(0.75–1.0)`, horizontal flip, ±15° rotation, ColorJitter |
| Optimizer            | AdamW                                                                  |
| Loss                 | CrossEntropyLoss (optionally weighted, optionally with label smoothing) |
| LR scheduler         | CosineAnnealingLR                                                      |
| Batch size           | 16                                                                     |
| Model selection      | Best epoch by validation macro-F1 (ConvNeXt) / validation accuracy (Swin) |

**6. Imbalance handling**
Because some classes are noticeably under-represented, both notebooks run additional experiments using `sklearn.utils.class_weight.compute_class_weight("balanced", ...)`, passed as the `weight` argument to `CrossEntropyLoss`. ConvNeXt adds a further run combining class weights with `label_smoothing=0.1` and a lower LR / higher weight decay.

**7. Evaluation**
On the held-out test split:
- Overall accuracy
- Macro F1 (the headline metric, because the classes are imbalanced)
- Full `classification_report` (per-class precision / recall / F1 / support)
- Confusion matrix (Seaborn heatmap)
- Qualitative panels of correct vs. incorrect predictions

---


## Results

> Performance Comparison Before and After Class Weights

| Model | Configuration | Accuracy | Precision | Recall | F1 Score |
|-------|---------------|----------|-----------|--------|----------|
| Swin Transformer | No class weights | 95.33% | 89% | 93% | 91% |
| ConvNeXt-Tiny | No class weights | 96.88% | 94% | 92% | 93% |
| Swin Transformer | Class weights | 96.67% | 94% | 96% | 95% |
| ConvNeXt-Tiny | Class weights | 97.1% | 94% | 97% | 95% |

A confusion matrix and a panel of correct/incorrect predictions are produced at the end of each notebook.
---

The notebooks are written for **Google Colab + Google Drive**. Paths like `/content/drive/MyDrive/Computer Vision Project/...` appear throughout — change these to your own paths (Drive or local) before running.

---

## Requirements

Python 3.10+ is recommended.

The full pinned environment (loose pins — these are the libraries the notebooks actually import) is in [`requirements.txt`](requirements.txt). Install with:

```bash
pip install -r requirements.txt
```

For GPU training, install a PyTorch build that matches your CUDA version from the official selector at <https://pytorch.org/get-started/locally/> instead of the generic `torch` line in the requirements file.

---

## How to Run

### Option A — Google Colab (matches the notebooks as-written)

1. Upload the dataset to your Google Drive, e.g. at
   `MyDrive/Diseases of date palm leaves dataset/Infected Date Palm Leaves Dataset/Processed/`.
2. Open `ComputerVisionProject_SwinTransformer.ipynb` in Colab.
3. Run the cells top-to-bottom. The first cells will:
   - mount Drive,
   - extract DINOv2 embeddings,
   - build the leakage-safe splits,
   - save the CSVs to `MyDrive/Computer Vision Project/`,
   - then fine-tune the Swin Transformer.
4. Open `ConvNeXt_Tiny.ipynb` and run it. It loads the **same** CSVs produced in step 3.

### Option B — Local machine

1. Clone or download this repository.
2. Create a virtual environment and install the requirements.
3. In each notebook, replace the Drive paths (`/content/drive/MyDrive/...`) with local paths (for example `./data/...` and `./splits/...`).
4. Skip the `drive.mount(...)` cell.
5. Run the notebooks in the same order: **Swin first** (it builds the splits), **ConvNeXt second**.

---

---

## Notes & Limitations

- **Dataset size.** The Processed subset is small (~3k images). Heavy data augmentation and pretrained backbones are essential — training from scratch is not advisable.
- **Class imbalance.** Some disease classes are noticeably under-represented. Use macro-F1 (not accuracy) as the primary metric.
- **Group-safe splitting.** Without grouping near-duplicates, the standard random split would optimistically inflate test scores. The DINOv2-embedding-based grouping at threshold 0.95 was chosen empirically — feel free to sweep this value.
- **Reproducibility.** `random_state=42` is fixed in the splits, but cuDNN nondeterminism, DataLoader shuffling, and float16 ops mean exact metric reproduction across machines isn't guaranteed.
- **HEIC files.** The dataset contains HEIC images; `pillow-heif` is imported and registered at the top of both notebooks. Don't remove that import or HEIC loading will silently fail.

---


## License

This repository is released under the MIT License — see `LICENSE` for details. The dataset is governed by the license stated by its original authors; consult the dataset publication before redistribution.
