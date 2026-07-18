# CropIQ: Fruit & Vegetable Classification

A TensorFlow/Keras project for classifying fruits and vegetables using the Fruits-360 dataset. Trains and evaluates MobileNetV2 and EfficientNetB0 models with proper per-model preprocessing.

## Project Structure

```
Tensorflow/
├── model.ipynb          # Complete training notebook (Google Colab)
├── test.md              # Previous evaluation results (pre-fix)
├── .gitignore
└── CropIQ/              # Dataset subset (Test/Validation images)
    ├── README.md        # Fruits-360 dataset documentation
    ├── Test/
    │   ├── Apple 19/
    │   └── Apple Braeburn 1/
    └── Validation/
        └── apple_pink_lady_1/
```

## Dataset

**Fruits-360** (Version 2026.5.12.0) - 102,551 images across 145 classes (fruits, vegetables, nuts, seeds).

This project uses a **13-class subset** + background:
- Apple, Banana, Cabbage, Carrot, Cucumber, Eggplant, Grape, Onion, Orange, Papaya, Pepper, Strawberry, Tomato
- Background (from COCO unlabeled2017)

Split: 50% Training / 25% Validation / 25% Test (preserves original Fruits-360 specimen-level splits)

## Models Trained

| Model | Preprocessing | Parameters |
|-------|---------------|------------|
| **MobileNetV2** | `mobilenet_v2.preprocess_input` (→ [-1, 1]) | ~3.5M |
| **EfficientNetB0** | `efficientnet.preprocess_input` (→ [0, 255] internal) | ~5.3M |

## Key Pipeline Fixes Applied

1. **Per-model preprocessing** (critical): Both models now receive raw [0, 255] images from generators. Each model has a `Lambda` preprocessing layer:
   - MobileNetV2: scales to [-1, 1] via `mobilenet_v2.preprocess_input`
   - EfficientNetB0: passes through unchanged; model's internal layers handle scaling

2. **Removed `rescale=1./255`** from all `ImageDataGenerator` instances — was causing double-scaling for EfficientNetB0

3. **BatchNorm freezing during fine-tuning**: BN running statistics frozen (`layer.trainable = False`) on unfrozen layers to prevent corruption from small batch sizes

4. **Original dataset splits preserved**: Uses Fruits-360's original Training/Validation/Test folders (specimen-level split by k, k+1, k+2, k+3 rule), not random image-level shuffle

## Training Details

### MobileNetV2
- Phase 1 (Head): 20 epochs, LR 1e-3 → 1e-5 (cosine decay + warmup)
- Phase 2 (Fine-tune last 30 layers): 20 epochs, LR 5e-5 → 1e-6

### EfficientNetB0
- Phase 1 (Head): 20 epochs, LR 1e-3 → 1e-5
- Phase 2 (Fine-tune last 30 layers): 25 epochs, LR 5e-5 → 1e-7

Shared: AdamW (weight_decay=1e-4), Label Smoothing (0.1), Class Weights (balanced), Top-3 Accuracy metric

## Requirements

```bash
pip install tensorflow opencv-python scikit-learn matplotlib tqdm
```

## Usage

1. Open `model.ipynb` in Google Colab
2. Mount Google Drive with dataset (`CropIQ.zip` containing Fruits-360)
3. Run all cells sequentially
4. Models saved to Drive: `best_mobilenet.keras`, `best_efficientnet.keras`
5. TFLite exports: `mobilenet_model.tflite`, `efficientnet_model.tflite`, `labels.txt`
6. **Results file**: `results.json` — contains test accuracy, per-class precision/recall/F1, confusion matrices, training history, and inference benchmarks for both models. Share this file for analysis.

## Known Limitations

- **Background domain shift**: Training uses random background augmentation; validation/test use original white studio backgrounds. Real-world deployment (phone camera) will have varied backgrounds.
- **External test set needed**: 100% accuracy on studio photos ≠ real-world performance. Collect phone photos for true evaluation.
- **EfficientNetB0 LR schedule**: May need independent tuning vs MobileNetV2 recipe

## License

Dataset: CC BY-SA 4.0 (Mihai Oltean, Fruits-360)
Code: MIT License