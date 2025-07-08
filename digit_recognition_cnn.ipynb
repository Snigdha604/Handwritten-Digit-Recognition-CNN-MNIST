# Import required libraries
import numpy as np
import struct
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Loading IDX Files
def load_images(filepath):
    with open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28, 28, 1)
    return images / 255.0  # Normalize

def load_labels(filepath):
    with open(filepath, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Load Dataset (Update paths as needed)
train_images = load_images("train-images-idx3-ubyte")
train_labels = load_labels("train-labels-idx1-ubyte")

test_images = load_images("t10k-images-idx3-ubyte")
test_labels = load_labels("t10k-labels-idx1-ubyte")

# One-hot encode labels
y_train = to_categorical(train_labels, 10)
y_test = to_categorical(test_labels, 10)

# Split training into train/val
X_train, X_val, y_train, y_val = train_test_split(train_images, y_train, test_size=0.1)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

# Compile and Train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=128, validation_data=(X_val, y_val))

# Evaluation on test set
loss, acc = model.evaluate(test_images, y_test)
print(f"\nTest accuracy: {acc:.4f}")

# Predict and Visualize
predictions = model.predict(test_images[:5])
for i in range(5):
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[i])} | Actual: {np.argmax(y_test[i])}")
    plt.axis('off')
    plt.show()
