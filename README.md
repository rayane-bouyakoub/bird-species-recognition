# 🐦 Bird Species Recognition

A deep learning project that uses transfer learning with **VGG16** to recognize bird species from images and provide their corresponding bird songs. The system includes a **PyQt5 desktop application** for interactive use.

---

## 🎯 Project Overview

This project applies **transfer learning** to classify bird species with high accuracy. The trained model is integrated into a desktop interface, enabling users to upload bird images, get predictions, and hear the corresponding bird song.

---

## ✨ Key Features

- **Transfer Learning with VGG16**: Fine-tuned last convolutional block for robust feature extraction  
- **High Accuracy**: Achieved **100% test accuracy after 10 epochs** using Adam optimizer and categorical cross-entropy loss  
- **Model Deployment**: Serialized trained model for efficient loading and inference in production  
- **Desktop Interface**: Built with **PyQt5**, allowing users to:  
  - Upload bird images  
  - Classify species  
  - Play the corresponding bird song  

---

## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow  
- **ML Tools**: scikit-learn, NumPy  
- **Visualization**: matplotlib, seaborn  
- **UI Framework**: PyQt5  