# Align Where the Words Look: Cross-Attention-Guided Patch Alignment with Contrastive and Transport Regularization for Bengali Captioning

## Description

This repo hosts the official implementation of our Bengali captioning pipeline built around a tri-loss objective: Patch-Alignment Loss (PAL) + InfoNCE + Sinkhorn-based Optimal Transport (OT). PAL reuses decoder cross-attention to pool caption-relevant patches and align real–synthetic evidence; InfoNCE adds global instance discrimination between pooled real/synthetic features; OT enforces balanced, fine-grained patch correspondences under the same text weights. The result is stronger grounding and more accurate Bengali captions under limited supervision.

The project integrates:

* Translation pipeline: automated English→Bengali captioning with LaBSE validation
* Synthetic image generation: bilingual prompts with Kandinsky 2.1 to pair each caption with a synthetic image
* Architecture: frozen MaxViT vision encoder + linear+LayerNorm bridge + mBART-50 Bengali decoder
* Patch-Alignment Loss (PAL): cross-attention–guided real↔synthetic alignment at caption-attended patches
* Contrastive and transport regularization: InfoNCE for global margins; entropic OT for fine-grained patch matching
* Multi-scale training: progressive decoder unfreezing and optional multi-scale PAL
* Evaluation: BLEU (B1–B4), METEOR, CIDEr, ROUGE-L, and BERTScore

The system addresses scarce Bengali image–text pairs by coupling synthetic data with text-conditioned alignment, improving both lexical precision and semantic faithfulness.


## Execution Flow

The pipeline consists of 5 sequential stages:

### Stage 1: Caption Translation (`stage-1-translate-captions.ipynb`)
- Loads MSCOCO English captions
- Translates to Bengali using `shhossain/opus-mt-en-to-bn`
- Validates semantic alignment using LaBSE embeddings
- Filters translations with similarity threshold ≥ 0.75
- Outputs: `secure_translated_bn_valid.csv`

### Stage 2: Synthetic Image Generation (`stage-2-synthetic-images-from-translated-captions.ipynb`)
- Loads translated Bengali captions
- Generates synthetic images using Kandinsky 2.1
- Creates joint English-Bengali prompts for better generation
- Saves images with metadata (JSON + PNG)
- Outputs: `synthetic_bengali_captioned_images.csv`

### Stage 3: Model Training (`stage-3-train-model-with-split-decoder-attention.ipynb`)
- Pairs real COCO images with synthetic images
- Trains MaxViT + mBART model with PAL
- Implements progressive unfreezing schedule
- PAL, InfoNCE and Optimal Transport losses
- Outputs: `maxvit_mbart_bn_checkpoint.pt`

### Stage 4: PAL Analysis (`stage-4-pal-analysis.ipynb`)
- Compares models trained with/without PAL
- Generates loss curves and embedding visualizations
- Creates t-SNE/UMAP plots of real vs synthetic embeddings
- Evaluates on Flickr30k and test sets
- Outputs: Loss comparison PDFs and metrics CSVs

### Stage 5: Metrics Ablation (`stage-5-metrics-ablation.ipynb`)
- Performs hyperparameter ablation study
- Tests different PAL weights, attention temperatures, and top-k ratios
- Evaluates impact on BLEU, CIDEr, METEOR scores
- Outputs: `pal_ablation_results.csv` and visualization plots

## Directory Structure & Paths

### Input Paths (Kaggle)
```
/kaggle/input/
├── coco-image-caption/
│   ├── annotations_trainval2014/annotations/
│   │   └── captions_train2014.json
│   └── train2014/train2014/
│       └── COCO_train2014_*.jpg
├── synthetic-bic-pairs/
│   └── synthetic-bengali-images/
│       ├── *.png (synthetic images)
│       └── *.json (metadata)
├── flickr30k/
│   ├── captions.txt
│   └── flickr30k_images/
└── Various model checkpoints (*/transformers/default/*)
```

### Output Paths
```
/kaggle/working/
├── secure_translated_bn_valid.csv
├── synthetic_bengali_images/
├── synthetic_bengali_captioned_images.csv
├── maxvit_mbart_bn_checkpoint.pt
├── maxvit_mbart_captioning_model.pth
├── history_*.csv
├── loss_pal_vs_ce.pdf
├── embeddings_ce_vs_pal.pdf
├── flickr30k_bengali_caption_results.csv
├── df_test_generated_bestofN_scores.csv
└── pal_ablation_results.csv
```

## Instructions to Run in Kaggle

### Prerequisites
1. Create a new Kaggle notebook
2. Enable GPU (P100 recommended)
3. Add the following datasets:
   - `coco-image-caption` (MSCOCO 2014)
   - Your translated captions CSV (after Stage 1)
   - Your synthetic images dataset (after Stage 2)
   - (Optional) Flickr30k dataset for evaluation

### Step-by-Step Execution

#### Stage 1: Translation
```python
# 1. Run all cells in stage-1-translate-captions.ipynb
# 2. Adjust batch_size (default: 128) and split_points if needed
# 3. Monitor translation progress (uses 5th part of 10 by default)
# 4. Save output: secure_translated_bn_valid.csv
```

#### Stage 2: Synthetic Image Generation
```python
# 1. Install required packages:
!pip install -q diffusers torch torchvision transformers accelerate

# 2. Run stage-2-synthetic-images-from-translated-captions.ipynb
# 3. Adjust image slice (default: indices 2500:2700)
# 4. Monitor GPU memory (Kandinsky requires ~12GB)
# 5. Save outputs to /kaggle/working/synthetic_bengali_images/
```

#### Stage 3: Model Training
```python
# 1. Install dependencies:
!pip install -q -U transformers accelerate timm torch datasets
!pip install -q sacremoses sentencepiece geomloss

# 2. Configure training parameters:
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 4
NUM_EPOCHS = 6
INITIAL_PATCH_ALIGNMENT_WEIGHT = 0.5

# 3. Run training cells
# 4. Monitor loss curves (CE, PAL, InfoNCE, OT)
# 5. Save checkpoint after each epoch
```

#### Stage 4: PAL Analysis
```python
# 1. Set environment variables for comparison:
%env CE_ONLY_CKPT_IN=/path/to/ce_only.pt
%env WITH_PAL_CKPT_IN=/path/to/with_pal.pt

# 2. Run analysis cells
# 3. Generate comparison plots
# 4. Evaluate on test sets
```

#### Stage 5: Ablation Study
```python
# 1. Configure ablation grid:
PATCH_WEIGHTS = [0.1, 0.3, 0.5, 0.8]
ATTN_TEMPS = [0.7, 1.0, 1.3]
TOPK_RATIOS = [0.1, 0.3, 0.5]

# 2. Run ablation (takes ~2-3 hours)
# 3. Analyze results in CSV and plots
```


### Acknowledgments

- MSCOCO dataset for base English captions
- Hugging Face for transformer models
- Kandinsky team for image generation model
- LaBSE for semantic similarity validation
