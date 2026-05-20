# 🚦 RADAR: Road Accident Detection and Analysis using Deep Learning

## 📌 Overview
RADAR is a deep learning-based system designed to automatically detect road accidents from CCTV video data.  
The system combines spatial and temporal modeling using VGG16 and GRU, along with TimeGAN for data augmentation.

---

## 🎯 Features
- Automatic accident detection from video sequences
- Spatio-temporal analysis using CNN–GRU architecture
- Synthetic data generation using TimeGAN
- Real-time prediction with probability output
- Frame capture at accident occurrence
- Visualization with bounding boxes for impact area

---

## 🧠 Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas, Matplotlib
- Scikit-learn

---

## 🧩 Models Used
- **VGG16** – Spatial feature extraction from frames  
- **GRU (Gated Recurrent Unit)** – Temporal sequence learning  
- **CNN–GRU Hybrid Model** – Combined spatio-temporal modeling  
- **TimeGAN** – Synthetic data generation for class imbalance  

---

## 📂 Dataset
- Input: Accident and non-accident videos
- Frames extracted from video using OpenCV
- Labels provided via `mapping.csv`

---

## ⚙️ Workflow
1. Video input
2. Frame extraction
3. Preprocessing (resize, normalize)
4. Feature extraction (VGG16)
5. Sequence generation
6. Data augmentation (TimeGAN)
7. Model training (GRU)
8. Prediction and visualization

---

## 🛠️ Installation

```bash
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn
