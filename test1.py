import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense

# Simulated EEG Data (100 Time Steps, 64 Channels)
eeg_data = np.random.rand(100, 64)

#  Fourier Transform for Frequency Analysis
freq_domain_signal = np.abs(np.fft.fft(eeg_data, axis=0))

# CNN-LSTM Model for EEG Classification
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(64, 1)),
    LSTM(50, return_sequences=True),
    LSTM(50),
    Dense(10, activation='softmax')  # 10 movement classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# **Plot EEG Raw vs. Filtered Signals**
plt.figure(figsize=(12, 6))

# Plot Raw EEG Signal
plt.subplot(1, 2, 1)
plt.plot(eeg_data[:, 0], color='blue', alpha=0.7)
plt.title("Raw EEG Signal")
plt.xlabel("Time Steps")
plt.ylabel("Amplitude")

# Plot Fourier Transformed EEG Signal
plt.subplot(1, 2, 2)
plt.plot(freq_domain_signal[:, 0], color='red', alpha=0.7)
plt.title("Filtered EEG Signal (Fourier Transform)")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")

plt.show()

print("EEG Processing & Visualization Completed! ")
