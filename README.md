# CropIQ: Fruit & Vegetable Classification

A TensorFlow/Keras project for classifying fruits and vegetables using the Fruits-360 dataset. Trains and evaluates MobileNetV2 and EfficientNetB0 models with proper per-model preprocessing.

## Project Structure

```
Tensorflow/
├── model.ipynb          # Complete training notebook (Google Colab) — 17 code cells + markdown headers
├── README.md
├── .gitignore
├── .gitattributes       # Git LFS tracking for model artifacts
├── models/              # Trained artifacts (Git LFS tracked)
│   ├── final_mobilenet.keras
│   ├── final_efficientnet.keras
│   ├── mobilenet_model.tflite
│   ├── mobilenet_model_fp16.tflite
│   ├── efficientnet_model.tflite
│   ├── efficientnet_model_fp16.tflite
│   ├── labels.txt
│   ├── labels.json
│   └── results.json
├── CropIQ/              # Full Fruits-360 dataset (102,551 images)
│   ├── README.md        # Fruits-360 dataset documentation
│   ├── Training/
│   ├── Validation/
│   └── Test/
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
| **MobileNetV2** | `Rescaling(1./127.5, offset=-1.0)` (→ [-1, 1]) | ~3.5M |
| **EfficientNetB0** | `EfficientNetPreprocess` (ImageNet torch-style: $(x/255 - \mu)/\sigma$) | ~5.3M |

## Key Pipeline Fixes Applied

1. **Per-model serializable preprocessing**: Replaced fragile `Lambda` layer function references with serializable layers:
   - MobileNetV2: scales [0, 255] to [-1, 1] via native `tf.keras.layers.Rescaling(1./127.5, offset=-1.0)`
   - EfficientNetB0: normalized via serializable `EfficientNetPreprocess` layer implementing exact ImageNet torch-style $(x/255 - \mu)/\sigma$ ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$), separate from internal backbone operations. Prevents deserialization crashes and enables clean TFLite export.

2. **Removed `rescale=1./255`** from all `ImageDataGenerator` instances — was causing double-scaling for models with pipeline normalization.

3. **Model-specific BatchNorm strategies in fine-tuning**:
   - MobileNetV2: BN layers in the unfrozen tail are **frozen** (`layer.trainable = False`) to prevent running statistics drift at the low fine-tuning learning rate ($5 \times 10^{-5}$).
   - EfficientNetB0: BN layers in the unfrozen tail are **trained** (batch size 64 keeps statistics stable, allowing depthwise feature adaptation).
   - Both models: BN in the early frozen stem remains completely frozen.

4. **Original dataset splits preserved**: Uses Fruits-360's original Training/Validation/Test folders (specimen-level split by k, k+1, k+2, k+3 rule), not random image-level shuffle.

5. **Folder→class mapping fixed with separator normalization**: Merges Fruit-360 variety folders into their parent classes using separator normalization (`low.replace('_', ' ')`) and prefix matching (e.g. `Apple Braeburn 1` / `apple_golden_1` → `Apple`, `eggplant_long_1` → `Eggplant`, `cabbage_white_1` → `Cabbage`), accurately mapping 76 variety folders across all splits without missing multi-word or snake_case classes.

6. **Selective Background Acquisition**: Downloads lightweight annotation JSON (~4.7 MB) and directly fetches 600 background images (~30 MB total instead of full 19 GB zip archive). A `.extraction_complete` sentinel file ensures re-runs skip completed downloads and cleans incomplete attempts.

7. **Background replacement as a gated layer**: `RandomBackgroundReplace` is a `Layer` subclass with a `training` argument and `get_config()` — augmentation runs only during `fit()`, passes through at inference, and keeps `tf.random.uniform` out of the exported TFLite graph.

## Scalability Optimizations

- **Mixed precision (`mixed_float16`)**: ~2x faster training on T4 with lower VRAM usage
- **Batch size 64**: more stable BatchNorm stats, better GPU utilization
- **Callbacks on `val_accuracy`** (not `val_loss`): early stopping + best-weights checkpointing align with the actual goal; `TerminateOnNaN` guards against divergence
- **GPU memory growth + device detection** at startup
- **Ensemble evaluation**: results.json includes averaged-probability ensemble accuracy (typically +1-3% over best single model)

## Training Details

Target accuracy: **90-95% on the test set** (single models); the ensemble in results.json typically lands at the top of that range.

### MobileNetV2
- Phase 1 (Head): 20 epochs, LR 1e-3 → 1e-5 (cosine decay + warmup)
- Phase 2 (Fine-tune last 30 layers): 20 epochs, LR 5e-5 → 1e-6

### EfficientNetB0
- Phase 1 (Head): 20 epochs, LR 1e-3 → 1e-5
- Phase 2 (Fine-tune last 30 layers): 25 epochs, LR 5e-5 → 1e-7

Shared: AdamW (weight_decay=1e-4), Label Smoothing (0.1), Class Weights (balanced), Top-3 Accuracy metric, mixed float16

## Requirements

```bash
pip install tensorflow opencv-python scikit-learn matplotlib tqdm
```

## Usage

### Training (Google Colab)

1. Open `model.ipynb` in Google Colab (GPU runtime)
2. Upload `CropIQ.zip` to Drive: `MyDrive/CropIQ/CropIQ.zip`
3. Run cells top-to-bottom (17 code cells with markdown headers) — handles extraction, class merging, background download, training both models, evaluation, ensemble, and export
4. Models saved to Drive: `best_mobilenet.keras`, `best_efficientnet.keras` (+ `_finetuned` variants)
5. TFLite exports: `mobilenet_model.tflite` (+ `_fp16` variant with `SELECT_TF_OPS`), `efficientnet_model.tflite` (+ `_fp16` variant with `SELECT_TF_OPS`), `labels.txt`, `labels.json` (ordered JSON array where array index = class index)
6. **Results file**: `results.json` — contains test accuracy, per-class precision/recall/F1, confusion matrices, training history, inference benchmarks, ensemble accuracy, and `tflite_parity` metrics for both standard and FP16 models.

### Model Artifacts

Trained artifacts are stored in `models/` (Git LFS tracked):

```bash
# After training, copy from Drive to repo:
cp "/content/drive/MyDrive/CropIQ/*.keras" models/
cp "/content/drive/MyDrive/CropIQ/*.tflite" models/
cp "/content/drive/MyDrive/CropIQ/labels.txt" models/
cp "/content/drive/MyDrive/CropIQ/labels.json" models/
cp "/content/drive/MyDrive/CropIQ/results.json" models/
```

Then commit (Git LFS handles large files):

```bash
git add models/
git commit -m "Add trained models + artifacts"
git push
```

### Android Integration

For the CropIQ Android app, copy only the TFLite model + labels to `app/src/main/assets/`:

```
app/src/main/assets/
├── mobilenet_model.tflite      (or efficientnet_model.tflite, or _fp16 variants)
└── labels.txt                  (or labels.json — ordered JSON array: labels[index])
```

Use the provided `CropIQClassifier` Kotlin class (supports GPU/NNAPI delegates, Flex ops):
```kotlin
// labels.json is a JSON array: ["Apple", "Banana", ...]
val className = labelsJsonArray.getString(predictedIndex)
```

### Branch

Active development on `cropiq-android-integration` branch:

```bash
git checkout cropiq-android-integration
```

## Known Limitations

- **Background domain shift**: Training uses random background augmentation; validation/test use original white studio backgrounds. Real-world deployment (phone camera) will have varied backgrounds.
- **External test set needed**: 100% accuracy on studio photos ≠ real-world performance. Collect 50–100 phone photos under natural conditions as a holdout set before trusting deployment metrics.
- **TFLite models require Flex delegate** (`SELECT_TF_OPS`) due to mixed-precision training. For pure TFLite, retrain with `float32` policy.
- **App-side gating required**: never display predictions below ~50% confidence; treat `Background` top-1 as "no fruit detected," not as an answer.

## License

Dataset: CC BY-SA 4.0 (Mihai Oltean, Fruits-360)
Code: MIT License
