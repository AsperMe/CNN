#Image Classification using CNN (TensorFlow/Keras)

##  Problem Statement
The goal of this project is to build a Convolutional Neural Network (CNN) capable of classifying images from the **CIFAR-10 dataset** into 10 categories, including airplanes, cars, birds, cats, and more.  
This project demonstrates practical deep learning techniques for image recognition and overfitting control.

---

##  Dataset
**CIFAR-10** is a benchmark dataset containing:
- 60,000 colour images** of size 32×32 pixels  
- 10 classes**, with 6,000 images per class  
- Training set: 50,000 images  
- Test set: 10,000 images  

It is automatically loaded using TensorFlow/Keras (`tf.keras.datasets.cifar10`).

---

## Model Architecture
A custom **CNN** model was designed with:
- Multiple **Convolution + BatchNormalization + ReLU + MaxPooling** blocks  
- **Dropout layers (0.3)** after each block to reduce overfitting  
- **Dense (Fully Connected)** layers with softmax output  

**Optimizer:** Adam  
**Loss:** Sparse Categorical Crossentropy  
**Metrics:** Accuracy  
**Learning Rate Scheduler:** ReduceLROnPlateau  
**EarlyStopping:** patience = 10  

---

## Training Configuration
- **Epochs:** 50  
- **Batch Size:** 64  
- **Data Augmentation:** Enabled (random flips, rotations, shifts)  
- **Validation Split:** 20%  
- **Best model checkpoint:** `save_model/best_model.keras`

---

## Evaluation
The model achieved:
- **Validation Accuracy:** 91.18%  
- **Test Accuracy:** 90.55%  

Visualizations:
- `results/accuracy.png` → Training/Validation Accuracy curve  
- `results/loss.png` → Training/Validation Loss curve  
- `results/confusion_matrix.png` → Confusion Matrix  

---

##  How to Run
1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/cnn-image-classifier.git
   cd cnn-image-classifier
   
2.Create virtual environment

'''bash
python3 -m venv .venv
source .venv/bin/activate
Install dependencies

'''bash
pip install -r requirements.txt
Train the model

'''bash
python3 train.py
Evaluate the model

'''bash
python3 evaluate.py
