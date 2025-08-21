import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import cv2  # for custom image prediction

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Normalize data (0-255 → 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN (28x28 → 28x28x1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 digits (0–9)
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
print("Training model...")
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Plot sample predictions
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.binary)
    pred = np.argmax(model.predict(x_test[i].reshape(1, 28, 28, 1)))
    plt.xlabel(f"Pred: {pred}, True: {y_test[i]}")
plt.show()

# Save model
model.save("digit_model.h5")
print("Model saved as digit_model.h5")

# ---- OPTIONAL: Predict your own image ----
def predict_custom_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # load in grayscale
    img = cv2.resize(img, (28, 28))                  # resize to 28x28
    img = 255 - img                                  # invert (white background → black digit)
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    prediction = np.argmax(model.predict(img))
    print(f"Predicted digit for {img_path}: {prediction}")

# Example usage:
# predict_custom_image("my_digit.png")  # put your own digit image here
