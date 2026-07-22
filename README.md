# Real-Time Indian Sign Language Recognition

A real-time **Indian Sign Language (ISL) Recognition System** built using **MediaPipe**, **TensorFlow**, **BiLSTM**, and **OpenCV**. The system recognizes dynamic Indian Sign Language gestures from live webcam input by extracting hand landmarks and modeling temporal gesture sequences using a Bidirectional Long Short-Term Memory (BiLSTM) network.

## Overview

Communication between hearing-impaired individuals and people unfamiliar with Indian Sign Language remains a significant challenge. This project aims to bridge that gap by recognizing commonly used ISL gestures in real time without requiring sensor gloves or specialized hardware.

The system uses **MediaPipe** for efficient hand landmark extraction and a **Bidirectional LSTM (BiLSTM)** network to learn temporal dependencies in dynamic gestures. Several preprocessing and prediction stabilization techniques are incorporated to improve recognition accuracy and robustness.

---

## Features

- Real-time gesture recognition using a standard webcam
- MediaPipe-based 3D hand landmark extraction
- Bidirectional LSTM (BiLSTM) for temporal sequence modeling
- Custom dataset containing 26 Indian Sign Language gesture classes
- Feature normalization and temporal smoothing
- Gaussian noise augmentation
- Dropout and L2 regularization
- Sliding window inference
- Confidence threshold filtering
- Majority voting for stable predictions
- Real-time gesture display with confidence score
- CPU-based implementation (No GPU required)

---

## Tech Stack

- Python
- TensorFlow
- Keras
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

## Dataset

A custom dataset was created using a webcam by following reference gestures from the **Indian Sign Language Research and Training Centre (ISLRTC)**.

### Dataset Statistics

- Total recorded videos: **553**
- Valid gesture sequences: **507**
- Gesture classes: **26**
- Sequence length: **45 frames**
- Features per frame: **133**

### Gesture Classes

```
Hello
Bye
Thanks
Sorry
Welcome
Please
Namaste
I
You
He
She
What
Where
Man
Woman
Teacher
Go
Good
Correct
Sign
Language
Deaf
Today
Home
Food
```

---

## Methodology

### 1. Data Collection

- Webcam-based gesture recording
- Multiple samples per gesture
- Different lighting conditions
- Different hand orientations
- Multiple gesture speeds

### 2. Feature Extraction

MediaPipe extracts **21 hand landmarks** for each detected hand.

Features include:

- Normalized 3D landmark coordinates
- Wrist-relative normalization
- Scale normalization
- Geometric distance features

Each frame is represented using a **133-dimensional feature vector**.

---

### 3. Data Preprocessing

The preprocessing pipeline includes:

- Feature normalization
- Temporal smoothing
- Gaussian noise augmentation
- Invalid sequence removal
- Sequence validation

---

### 4. Model Architecture

```
Input (45 × 133)

↓

Gaussian Noise

↓

Bidirectional LSTM (48 Units)

↓

Dropout (0.5)

↓

Bidirectional LSTM (24 Units)

↓

Dropout (0.5)

↓

Dense (ReLU)

↓

Batch Normalization

↓

Softmax Output (26 Classes)
```

Training Configuration

- Optimizer: Adam
- Learning Rate: 0.0008
- Batch Size: 16
- Epochs: 100
- Loss Function: Sparse Categorical Crossentropy
- Early Stopping
- ReduceLROnPlateau

Regularization Techniques

- Dropout
- L2 Regularization
- Gaussian Noise
- Temporal Smoothing

---

## Real-Time Recognition Pipeline

```
Webcam
    │
    ▼
Frame Capture
    │
    ▼
MediaPipe Hand Detection
    │
    ▼
Landmark Extraction
    │
    ▼
Feature Normalization
    │
    ▼
Sliding Window (45 Frames)
    │
    ▼
BiLSTM Prediction
    │
    ▼
Confidence Filtering
    │
    ▼
Majority Voting
    │
    ▼
Gesture Prediction
```

---

## Performance

| Metric | Score |
|---------|-------|
| Accuracy | **86.27%** |
| Precision | **90.74%** |
| Recall | **86.27%** |
| F1 Score | **86.05%** |

The model demonstrated stable performance during live webcam testing with consistent predictions across different lighting conditions and gesture variations.

---

## Project Structure

```
Indian-Sign-Language-Recognition/
│
├── dataset/
├── models/
├── notebooks/
├── preprocessing/
├── training/
├── realtime/
├── utils/
├── outputs/
├── requirements.txt
└── README.md
```

---

## Results

The developed system successfully:

- Recognizes 26 commonly used Indian Sign Language gestures
- Performs real-time recognition using webcam input
- Operates without wearable sensors
- Maintains stable predictions using temporal buffering and majority voting
- Runs efficiently on CPU

---

## Future Improvements

- Expand dataset with more gesture classes
- Continuous sentence-level recognition
- Sign-to-Speech translation
- Facial expression and body pose integration
- Transformer-based sequence models
- Mobile and Edge deployment
- Multilingual sign language recognition

---

