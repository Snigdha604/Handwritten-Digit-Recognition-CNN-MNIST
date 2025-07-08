# Handwritten-Digit-Recognition-CNN-MNIST

## 📝 Overview
This project demonstrates the use of a Convolutional Neural Network (CNN) for handwritten digit recognition using the MNIST dataset. It reads raw binary data files (`.ubyte` format), processes them using NumPy, and builds a CNN model using TensorFlow/Keras.
The model learns to classify digits (0 to 9) from 28x28 grayscale images and achieves high accuracy on the test set. The project also includes visualization of predictions to better understand how the model performs.
This is an ideal beginner-to-intermediate deep learning project that showcases model architecture, data preprocessing, training, evaluation, and result interpretation — all in one.

## 📌 Features
- Reads raw `.ubyte` MNIST dataset format (no CSVs)
- Implements a multi-layer CNN using TensorFlow
- Trains, validates, and tests the model
- Visualizes predictions vs. ground truth using matplotlib
- Achieves over **98% accuracy**
- Easily extensible for other image classification tasks

## 📁 Dataset
You’ll need to download the MNIST dataset files (in `.ubyte` format) and place them in the project folder
- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`
📥 Download from: [https://www.kaggle.com/datasets/hojjatk/mnist-dataset]

## Clone the Repository
```bash
git clone https://github.com/Snigdha604/mnist-digit-recognition-cnn.git
cd mnist-digit-recognition-cnn
```
## Install Requirements
```bash
pip install -r requirements.txt
```
##  Run the Model
```bash
python mnist_cnn.py
```
##  Model Architecture
Input: 28x28 grayscale image (1 channel)

→ Conv2D (32 filters, 3x3, ReLU)
→ MaxPooling2D (2x2)

→ Conv2D (64 filters, 3x3, ReLU)
→ MaxPooling2D (2x2)

→ Flatten
→ Dense (128 units, ReLU)
→ Dropout (0.3)
→ Dense (10 units, Softmax)

## Results
✅ Achieved ~98% accuracy on the MNIST test set
🖼️ Visualizes predictions vs actual labels for first 5 test images

## Technologies Used
Python
TensorFlow / Keras
NumPy
Matplotlib
Scikit-learn

## License
This project is licensed under the MIT License.
You are free to use, modify, and distribute this code.



