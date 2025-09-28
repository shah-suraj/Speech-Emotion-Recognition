
# Speech Emotion Recognition (SER)


## 🎯 Objective

Recognize human emotions (e.g., anger, happiness, sadness) from speech audio using extracted acoustic features and a deep learning model.

---

## 📁 Dataset

- **Samples**: 1140 audio clips  
- **Speakers**: ~19 speakers  
- **Emotions**:  
  - Neutral  
  - Happy  
  - Sad  
  - Angry  
  - Fearful  
  - Disgust  
  - Surprised  
  - Calm  

> Note: Minority classes like "neutral" were upsampled to ensure class balance.

---

## 🔍 Feature Extraction

### Pipeline:
1. **Trim Silence**
2. **Pre-Emphasis**
3. **Resample to 22.05kHz**
4. **Feature Extraction**
5. **Normalization using `StandardScaler`**

### Extracted Features:
- MFCCs (20 coefficients)
- MFCC Δ and ΔΔ
- Spectral Contrast
- Spectral Centroid
- Spectral Bandwidth
- Spectral Rolloff
- Chroma
- Zero Crossing Rate (ZCR)

> **Tool used**: `librosa` library for Python

---

## 🧠 Model Architecture

Convolutional Neural Network (CNN) with the following layers:

1. **Input Layer**  
   - Shape: `82 × 64 × 1` (Features × Time Steps × Channel)

2. **Conv2D Layers**  
   - Learn both low-level (pitch, timbre) and high-level (tone patterns) audio features

3. **Batch Normalization**

4. **ReLU Activation**

5. **Max Pooling**

6. **Dropout (20–40%)**

7. **Residual Connections**

8. **Global Average Pooling**

9. **Dense Layers**  
   - 256 and 128 neurons

10. **Output Layer (Softmax)**  
    - Predicts probabilities for the 8 emotion classes

---

## 🏆 Performance

- **Validation Accuracy**: **71.88%**

---

## 📦 Dependencies

- Python 3.8+
- NumPy
- Librosa
- Scikit-learn
- TensorFlow / Keras
- Matplotlib (for visualization, if needed)

---

## 📌 Run Instructions

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train_model.py

# Evaluate or test
python evaluate_model.py
```

---

## 📈 Why This Works?

- **CNNs** can effectively capture spatial-temporal features in spectrogram-like audio representations.
- Residual connections and dropout improve generalization and help the network retain essential low-level features.
- Feature-rich preprocessing pipeline ensures robust input representation.
