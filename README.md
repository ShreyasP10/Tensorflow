# CropIQ: Fruit & Vegetable Classification

A production-ready TensorFlow/Keras computer vision project for classifying fruits and vegetables using the Fruits-360 dataset. Implements transfer learning with **MobileNetV2** and **EfficientNetB0**, canonical `tf.data` input pipelines, in-model GPU data augmentation, and exports 100% pure `TFLITE_BUILTINS` and full INT8 quantized edge models.

---

## Project Structure

```
Tensorflow/
├── model.ipynb                   # Complete training & export notebook (Local & Colab compatible)
├── TFLITE_DEVELOPER_REFERENCE.md  # Complete 23-point architecture & edge deployment guide
├── README.md
├── .gitignore
├── .gitattributes                # Git LFS tracking for trained model artifacts
├── models/                       # Trained artifacts (Git LFS tracked)
│   ├── final_mobilenet.keras
│   ├── final_efficientnet.keras
│   ├── mobilenet_model.tflite          # 100% Pure TFLite Builtins (~3.1 MB)
│   ├── mobilenet_model_fp16.tflite     # FP16 Quantized (~2.5 MB)
│   ├── mobilenet_model_int8.tflite     # Full INT8 Quantized (~3.5 MB)
│   ├── efficientnet_model.tflite       # 100% Pure TFLite Builtins (~4.8 MB)
│   ├── efficientnet_model_fp16.tflite  # FP16 Quantized (~4.8 MB)
│   ├── efficientnet_model_int8.tflite  # Full INT8 Quantized (~5.2 MB)
│   ├── labels.txt                      # Newline-separated class labels
│   ├── labels.json                     # JSON array of ordered classes
│   ├── class_indices.json              # Class name to integer index mapping
│   └── results.json                    # Full evaluation metrics & camera input contract
├── CropIQ/                       # Source Fruits-360 dataset (102,551 images)
│   ├── README.md
│   ├── Training/
│   ├── Validation/
│   └── Test/
└── merged_dataset/               # Preprocessed 13-class + Background split directory
    ├── Training/
    ├── Validation/
    └── Test/
```

---

## Dataset

**Fruits-360** (Version 2026.5.12.0) - 102,551 images across 145 varieties.

This project trains on a **13-class subset** + background:
- **13 Produce Classes**: Apple, Banana, Cabbage, Carrot, Cucumber, Eggplant, Grape, Onion, Orange, Papaya, Pepper, Strawberry, Tomato
- **Background Class**: Non-produce background images sourced directly from COCO unlabeled2017

**Splits**: Preserves Fruits-360 specimen-level partition across Training, Validation, and Test sets (no data leakage).

---

## Models Trained

| Model | Internal Preprocessing | Input Pixel Contract | Parameters | Pure TFLite Size | Full INT8 Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MobileNetV2** | `Rescaling(1./127.5, offset=-1.0)` | Raw `[0, 255]` float32 RGB | ~3.5M | ~3.12 MB | ~3.5 MB |
| **EfficientNetB0** | Native Backbone `Rescaling(1./255.0)` | Raw `[0, 255]` float32 RGB | ~5.3M | ~4.87 MB | ~5.2 MB |

---

## Key Pipeline Fixes & Architecture Highlights

1. **Canonical `tf.data` Input Pipeline**:
   Replaced deprecated `ImageDataGenerator` with `tf.keras.utils.image_dataset_from_directory`. Features memory caching (`.cache()`) and background asynchronous prefetching (`.prefetch(buffer_size=tf.data.AUTOTUNE)`).

2. **In-Model GPU Data Augmentation**:
   Image augmentations (`RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomTranslation`) are implemented as native Keras preprocessing layers. They execute in-graph on the GPU during training and automatically act as an identity pass-through during inference and evaluation.

3. **Unified Input Pixel Contract & Zero Double-Normalization**:
   Stripped all external preprocessing layers that previously caused double-normalization. Both MobileNetV2 and EfficientNetB0 handle rescaling internally. External callers (Android CameraX, web APIs, inference scripts) feed raw `[0, 255]` float32 RGB pixels.

4. **100% Pure `TFLITE_BUILTINS` Export (Zero Flex Ops)**:
   Dedicated inference serving models strip training-only data augmentation layers prior to TFLite conversion. Both MobileNetV2 and EfficientNetB0 convert into standard `TFLITE_BUILTINS` without any `SELECT_TF_OPS` Flex dependencies.

5. **Full INT8 Post-Training Quantization (PTQ)**:
   Includes a representative dataset generator to quantize weights and activations to INT8. Provides maximum acceleration on mobile NNAPI and edge NPUs while preserving classification accuracy.

6. **Variety-to-Class Mapping & Case-Insensitive Matching**:
   Normalized folder matching (`.lower().strip().replace('_', ' ')`) accurately maps all 76 Fruits-360 variety folders into the 13 target parent classes (including `eggplant_long_1` → `Eggplant`, restoring 160 previously skipped training images). Varietal file copies prepend source folder names (`f"{entry.name}_{f_entry.name}"`) to prevent silent overwrite collisions.

7. **Stratified Multi-Class Parity Verification**:
   Validates numerical parity ($\text{MAE} < 0.01$) between Keras and TFLite across a stratified sample covering all 13 produce classes and background, rather than testing a single batch of one class.

8. **Data Integrity & Sentinel Checks**:
   Automatically purges `merged_dataset/` before rebuilding to prevent stale split contamination. Corrupt and truncated images are removed with `PIL.Image.verify()`. Hard assertions halt execution if any class has 0 images in any split.

---

## Scalability & Training Strategies

- **Mixed Precision (`mixed_float16`)**: Automatically detected and enabled on supported GPUs for ~2x faster training; falls back cleanly to `float32` on CPU.
- **BatchNorm Freezing**: Backbone layers use `base_model(x, training=False)` to keep ImageNet running mean and variance frozen throughout transfer learning and fine-tuning.
- **Learning Rate Warmup & Cosine Decay**: Warmup cosine decay callback stabilizes initial convergence and smoothly anneals the learning rate down to a specified floor.
- **Class Weights**: Inferred dynamically with `compute_class_weight('balanced', ...)` to balance smaller classes (Carrot, Cabbage, Eggplant) against large classes (Apple, Tomato).

---

## Developer Reference & Architecture Guide

For deep-dive documentation on all 23 core engineering principles across Foundation, Data Quality, Transfer Learning, Mixed Precision, Export, Android Integration, and Ops:
👉 **[TFLITE_DEVELOPER_REFERENCE.md](TFLITE_DEVELOPER_REFERENCE.md)**

---

## Requirements

```bash
pip install tensorflow opencv-python scikit-learn matplotlib tqdm certifi gdown pillow
```

---

## Usage

### Local Workstation Execution (`local-run` branch)

The `local-run` branch is preconfigured to run locally (Windows, macOS, Linux with CPU or GPU).

#### 1. Environment Setup
```bash
# Clone and switch to local-run branch
git clone https://github.com/ShreyasP10/Tensorflow.git
cd Tensorflow
git checkout local-run

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install tensorflow opencv-python scikit-learn matplotlib tqdm certifi gdown pillow
```

#### 2. Dataset Setup
- **Existing Local Dataset**: If `CropIQ/` (or `CropIQ.zip`) exists in the project root, the notebook detects it automatically.
- **Google Drive Auto-Download**: If missing, Cell 1 and Cell 3 use `gdown` to download `CropIQ.zip` directly into the project directory via `GDRIVE_DATASET_ID`.

#### 3. Run the Notebook
Open `model.ipynb` in VS Code, JupyterLab, or Cursor and execute cells sequentially:
- **Cell 1**: Configures determinism (seed 42), local paths (`./models`), and GPU/CPU precision.
- **Cell 3**: Resolves or auto-downloads and extracts `CropIQ.zip`.
- **Cell 5**: Partitions varieties into parent classes inside `./merged_dataset/` with corrupt image filtering.
- **Cell 7**: Downloads lightweight COCO annotation JSON (~4.7 MB) and 600 background images (~30 MB) into `./coco_unlabeled/`.
- **Cell 9**: Builds native `tf.data` datasets with caching and prefetching.
- **Cells 13–19**: Builds and trains MobileNetV2 and EfficientNetB0 (head training + fine-tuning).
- **Cells 20–29**: Evaluates models, verifies multi-class parity, and exports all TFLite models and metadata into `./models/`.

---

### Google Colab Execution

1. Open `model.ipynb` in Google Colab (select T4 GPU runtime).
2. Upload `CropIQ.zip` to your Google Drive at `MyDrive/CropIQ/CropIQ.zip`.
3. Run cells top-to-bottom. The notebook detects Colab, mounts Google Drive, and exports models to `/content/drive/MyDrive/CropIQ/`.

---

### Android CameraX Integration

For the CropIQ Android app, copy the exported model (`mobilenet_model.tflite` or `mobilenet_model_int8.tflite`) and `labels.json` into `app/src/main/assets/`:

```
app/src/main/assets/
├── mobilenet_model.tflite          (or mobilenet_model_int8.tflite)
└── labels.json
```

#### Camera Frame Input Contract:
- **Tensor Shape**: `[1, 224, 224, 3]`
- **Color Format**: RGB
- **Pixel Values**: Raw `[0.0, 255.0]` float32 (the model normalizes internally)
- **Aspect Ratio**: Center-crop 1:1 before resizing to 224x224
- **Runtime**: Standard TensorFlow Lite runtime with GPU or NNAPI delegate (zero Flex delegate required)

---

## Known Limitations & Best Practices

- **White Studio vs. Real-World Backgrounds**: Fruits-360 images feature controlled studio backgrounds. In real-world camera usage, confidence gating (threshold $\ge 60\%$) and treating the `Background` class as "No item detected" is required.
- **Holdout Validation**: Validate exported models with a test set of 50–100 natural smartphone camera photos before full field deployment.

---

## License

- Dataset: CC BY-SA 4.0 (Mihai Oltean, Fruits-360)
- Code: MIT License
