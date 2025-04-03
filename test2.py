import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

# Simulated Terrain Data (Height Maps)
terrain_data = np.random.rand(100, 50, 50)  # 100 samples, 50x50 terrain maps

# CNN Model for Terrain Prediction
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 1)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Predict safe (1) or rough terrain (0)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Select a Sample Terrain Map for Visualization
terrain_sample = terrain_data[0]

# **Plot Raw Terrain Map**
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(terrain_sample, cmap="gray")
plt.title("Raw Terrain Map")
plt.colorbar()

# Apply CNN Model to Predict Terrain Classification
terrain_prediction = model.predict(terrain_sample.reshape(1, 50, 50, 1))

# **Plot AI-Predicted Terrain Classification**
plt.subplot(1, 2, 2)
plt.imshow(terrain_sample, cmap="coolwarm")  # Change color to highlight AI classification
plt.title("AI-Classified Terrain Map")
plt.colorbar()

plt.show()

print("Terrain Prediction & Visualization Completed!")
