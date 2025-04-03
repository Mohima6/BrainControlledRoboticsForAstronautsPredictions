import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import pywt  # Import PyWavelets for Wavelet Transform
from sklearn.decomposition import PCA

# Generate Simulated EEG Data (Random Noise)
np.random.seed(42)
eeg_signal = np.random.rand(500, 10)  # 500 samples, 10 EEG channels

# Apply Fourier Transform for Frequency Analysis
freq_domain_signal = np.abs(np.fft.fft(eeg_signal, axis=0))

# Apply Wavelet Transform using PyWavelets
coeffs, freqs = pywt.cwt(eeg_signal[:, 0], np.arange(1, 31), 'gaus1')  # Using the first EEG channel

# Apply PCA for Noise Reduction
pca = PCA(n_components=5)  # Reduce to 5 key components
eeg_signal_reduced = pca.fit_transform(eeg_signal)

# **Plot EEG Signal Processing Results**
plt.figure(figsize=(12, 6))

# Plot Raw EEG Signal (First Channel)
plt.subplot(1, 3, 1)
plt.plot(eeg_signal[:, 0], color='blue', alpha=0.7)
plt.title("Raw EEG Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")

# Plot Fourier Transform Signal (First Channel)
plt.subplot(1, 3, 2)
plt.plot(freq_domain_signal[:, 0], color='red', alpha=0.7)
plt.title("Fourier Transform (Frequency Domain)")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")

# Plot PCA-Filtered EEG Signal (First Component)
plt.subplot(1, 3, 3)
plt.plot(eeg_signal_reduced[:, 0], color='green', alpha=0.7)
plt.title("PCA-Filtered EEG Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

print("EEG Signal Processing & Visualization Completed! ")
