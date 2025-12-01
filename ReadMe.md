# 🌱 Plant Seedling Classification — CNN with Transfer Learning

A deep learning project for classifying 12 species of crop and weed seedlings using TensorFlow/Keras and MobileNetV2.  
This notebook demonstrates the full machine-learning workflow: preprocessing, augmentation, model construction, fine-tuning, evaluation, and model saving.

---

## 📊 Project Overview
This project uses a convolutional neural network (CNN) built on top of **MobileNetV2** to classify plant seedlings.  
Key steps include:

- Loading and preprocessing the dataset  
- Data augmentation to increase generalization  
- Transfer learning with MobileNetV2  
- Fine-tuning the deeper layers  
- Training and validation  
- Evaluating accuracy and generating a confusion matrix  
- Saving trained model + class labels  

---

## 🚀 Features

### ✔ Transfer Learning
Uses MobileNetV2 with ImageNet weights as a feature extractor.

### ✔ Fine-tuning
Unfreezes deeper layers to increase classification accuracy.

### ✔ Data Augmentation
Random flip, rotation, zoom, brightness adjustments.

### ✔ Performance Visualization
Plots include:
- Training/validation accuracy  
- Training/validation loss  
- Confusion matrix  

### ✔ Model Persistence
The notebook exports:
- `plant_seedling_model.h5` — Saved Keras model  
- `label_classes.npy` — Encoded class names  

---

## 📂 Folder / File Structure

```
📁 project/
   ├── PlantDetection.ipynb
   ├── plant_seedling_model.h5
   ├── label_classes.npy
   ├── README.md
   └── dataset/ (if applicable)
```

---

## ⚙️ How to Run

### 1. Install Dependencies
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

### 2. Launch Jupyter Notebook
```bash
jupyter notebook PlantDetection.ipynb
```

### 3. Train the Model
Follow the sequential notebook cells.

---

## 📈 Model Performance

- Final Test Accuracy: **~85%** (varies based on randomness + dataset split)
- Strong performance boost after fine-tuning
- Model successfully distinguishes 12 plant species  

The final confusion matrix and performance charts are shown in the notebook output.

---

## 🧠 Techniques Used

| Technique | Purpose |
|----------|----------|
| Transfer Learning (MobileNetV2) | Speeds up training, improves accuracy |
| CNN Classification Head | Adds dense layers for prediction |
| Data Augmentation | Prevents overfitting |
| Fine-tuning | Improves feature extraction |
| Softmax Output | Multi-class classification |
| Categorical Cross-Entropy | Loss function |

---

## 💾 Output Artifacts

The notebook saves:

- `plant_seedling_model.h5` — trained deep learning model  
- `label_classes.npy` — mapping of class index → species  

These files can be reloaded for inference or deployment.
