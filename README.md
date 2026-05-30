
# 🦟 Malaria Cell Image Classifier

> A Convolutional Neural Network (CNN) trained to detect malaria parasites in blood cell microscopy images — built entirely for free, no API required.

---

## 🔬 What It Does

This project uses **deep learning** to classify microscopic blood cell images as either:

- 🦠 **Parasitized** — red blood cells infected with *Plasmodium* malaria parasites
- ✅ **Uninfected** — healthy red blood cells

By automating this detection, the model can assist in early diagnosis — particularly valuable in low-resource settings where expert microscopists may not be available.

---

## 📊 Dataset

**TensorFlow Malaria Dataset**
- 27,558 cell images (50/50 class balance)
- Source: National Institutes of Health (NIH)
- Auto-downloaded via `tensorflow-datasets` — no manual setup needed
- Reference: [TF Datasets Catalog](https://www.tensorflow.org/datasets/catalog/malaria)

| Split | Images |
|---|---|
| Training | ~19,290 |
| Validation | ~4,134 |
| Test | ~4,134 |

---

## 🧠 Model Architecture

A custom CNN built with TensorFlow/Keras:

```
Input (128×128×3)
    │
    ├── Conv Block 1: Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout(0.25)
    ├── Conv Block 2: Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout(0.25)
    ├── Conv Block 3: Conv2D(128) → BatchNorm → Conv2D(128) → MaxPool → Dropout(0.25)
    │
    └── Head: Flatten → Dense(256) → BatchNorm → Dropout(0.5) → Dense(1, sigmoid)
```

**Training config:**
- Optimizer: Adam (lr=0.001)
- Loss: Binary Cross-Entropy
- Callbacks: EarlyStopping · ReduceLROnPlateau · ModelCheckpoint
- Augmentation: Random flip, rotation, zoom (training set only)

---

## 📈 Results

| Metric | Score |
|---|---|
| Test Accuracy | ~95–97% |
| AUC | ~0.98+ |
| Precision | ~96% |
| Recall | ~96% |

---

## 🚀 How to Run

### Option 1 — Kaggle (Recommended, Free GPU)

1. Go to [kaggle.com](https://www.kaggle.com) and create a free account
2. Click **Create → New Notebook**
3. **File → Import Notebook** → upload `malaria_classifier_kaggle.ipynb`
4. Right panel: **Settings → Accelerator → GPU T4 x1**
5. Click **Run All**

> ⏱ Full training takes ~5–10 minutes on Kaggle's free T4 GPU.

### Option 2 — Local

```bash
pip install tensorflow tensorflow-datasets scikit-learn matplotlib seaborn pillow
jupyter notebook malaria_classifier_kaggle.ipynb
```

> ⚠️ Local training without a GPU will be significantly slower.

---

## 📁 Output Files

After running the notebook, the following files are saved:

| File | Description |
|---|---|
| `best_malaria_model.keras` | Best model checkpoint (highest val accuracy) |
| `malaria_cnn_final.keras` | Final model after all epochs |
| `sample_cells.png` | Grid of sample parasitized vs uninfected cells |
| `training_curves.png` | Accuracy and loss over epochs |
| `confusion_matrix.png` | Confusion matrix + confidence distribution |

---

## 🔄 Reusing the Trained Model

```python
import tensorflow as tf

# Load without retraining
model = tf.keras.models.load_model("malaria_cnn_final.keras")

# Predict on a new image
from PIL import Image
import numpy as np

img = Image.open("cell_image.png").convert("RGB").resize((128, 128))
img_array = np.array(img) / 255.0
img_tensor = tf.expand_dims(tf.constant(img_array, dtype=tf.float32), 0)

prob = model.predict(img_tensor)[0][0]
label = "Uninfected" if prob >= 0.5 else "Parasitized"
print(f"{label} ({prob*100:.1f}% confidence)")
```

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| TensorFlow / Keras | Model building & training |
| TensorFlow Datasets | Dataset download & management |
| scikit-learn | Evaluation metrics |
| Matplotlib / Seaborn | Visualisation |
| Pillow | Image loading for inference |
| Kaggle (T4 GPU) | Free cloud training |

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is not a certified medical diagnostic tool and should not be used to make clinical decisions.

---

## 📚 References

- Rajaraman et al. (2018). *Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images.* PeerJ.
- [TensorFlow Datasets — Malaria](https://www.tensorflow.org/datasets/catalog/malaria)
- [NIH National Library of Medicine — Cell Image Library](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html)
