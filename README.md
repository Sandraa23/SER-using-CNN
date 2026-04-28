#  Speech Emotion Recognition using CNN

**Binary Classification (Happy vs Sad) using RAVDESS + CREMA-D**

---

##  Overview

This project implements a **Speech Emotion Recognition (SER)** system using a **Convolutional Neural Network (CNN)** to classify emotions from audio signals.

The model performs **binary classification (Happy vs Sad)** and evaluates the impact of different **activation functions** on performance using a consistent training pipeline.

---

##  Objective

* Build a CNN model for speech-based emotion classification
* Convert audio signals into **Mel Spectrograms**
* Apply **data augmentation on training data only**
* Compare multiple **activation functions under identical conditions**
* Evaluate models using accuracy, loss, and classification metrics

---

##  Dataset

The model uses a **combination of two datasets**:

* RAVDESS
* CREMA-D

###  Emotion Filtering

Only the following emotions are used:

* Happy
* Sad

###  Label Mapping

* Happy → 0
* Sad → 1

---

##  Methodology

###  1. Audio Preprocessing

* Audio loaded using `librosa`
* Standardized with:

  * Sampling rate: **22050 Hz**
  * Fixed duration: **3 seconds** (padding/trimming applied)
* Converted into **Mel Spectrograms**
* Resized to **128 × 128 × 1**
* Min-max normalization applied per sample

---

###  2. Data Augmentation (Training Only)

Augmentation is applied **only to training data**:

* Time Stretch:

  * 0.9×
  * 1.1×

* Pitch Shift:

  * +1 step
  * -1 step

* Gaussian Noise:

  * Small random noise added

* Volume Scaling:

  * 0.9×
  * 1.1×

---

###  3. Train-Test Split

* Stratified split using `train_test_split`
* Test size: **15%**
* Random seed: **42**

---

###  4. Model Architecture (CNN)

The model consists of:

* 4 Convolutional Blocks:

  * Conv2D → BatchNorm → Activation → MaxPooling → Dropout

* Increasing filters:

  * 32 → 64 → 128 → 256

* Regularization:

  * L2 Regularization
  * Dropout (0.2 → 0.5)

* Final layers:

  * GlobalAveragePooling
  * Dense (128 units)
  * Output layer:

    * 1 neuron with **sigmoid activation**

---

###  5. Activation Functions Compared

The same CNN architecture is trained using:

* ReLU
* Tanh
* Leaky ReLU (α = 0.1)
* GELU
* Swish

---

###  6. Training Strategy

* Optimizer: Adam
* Learning rate: **0.0002**
* Epochs: **60**
* Batch size: **32**
* Validation split: **15%**

### Callbacks Used:

* EarlyStopping (patience = 8)
* ModelCheckpoint (best model saved)
* ReduceLROnPlateau (adaptive learning rate)

---

## Evaluation

Each model is evaluated using:

* Test Accuracy
* Test Loss
* Classification Report
* Confusion Matrix

---

##  Visualizations

The project generates:

* Confusion matrices for all activations
* Training curves (accuracy & loss) for each model
* Combined validation curves
* Comparison table of all models
* Bar chart comparing accuracy and loss

---

##  How to Run

1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Upload datasets to Google Drive:

```
/content/drive/MyDrive/SER_Project/data/
```

4. Update paths if needed in notebook

5. Run the notebook in Jupyter or Google Colab

---

##  Technologies Used

* Python
* TensorFlow / Keras
* Librosa
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

##  Key Highlights

* Binary classification: Happy vs Sad
* Uses **Mel Spectrograms as CNN input**
* Applies **multiple augmentation techniques**
* Compares **5 activation functions in identical conditions**
* Includes **full evaluation + visualization pipeline**

---

##  Future Improvements

* Extend to multi-class emotion recognition
* Use raw audio or spectrogram-based deep models
* Try hybrid CNN + LSTM architectures
* Deploy as a real-time application

---

##  Notes

* Datasets are not included due to size constraints
* Project uses **Google Drive paths (Colab-based setup)**
* Ensure correct directory structure before running
