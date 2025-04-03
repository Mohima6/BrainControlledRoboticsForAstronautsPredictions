import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm

# Simulated EEG Signal Data (Mocked for Movement Prediction)
np.random.seed(42)
sequence_length = 100
eeg_signal = np.random.choice([0, 1, 2], size=sequence_length, p=[0.4, 0.4, 0.2])  # 0=Idle, 1=Move, 2=Adjust

# Reshape for HMM Training
eeg_sequence = eeg_signal.reshape(-1, 1)

# Define HMM Model (3 states: Idle, Move, Adjust)
model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)
model.fit(eeg_sequence)

#  Predict Movement States
predicted_states = model.predict(eeg_sequence)

#  Plot EEG Signal & Predicted Movement States
plt.figure(figsize=(10, 5))
plt.plot(range(sequence_length), eeg_signal, marker='o', linestyle='-', color='blue', alpha=0.6, label="Original EEG Signal")
plt.plot(range(sequence_length), predicted_states, marker='s', linestyle='-', color='red', alpha=0.6, label="Predicted Movement States")
plt.xlabel("Time Steps")
plt.ylabel("Signal / State")
plt.title("HMM-Based Prediction of Astronaut Mobility Using EEG")
plt.legend()
plt.grid(True)
plt.show()

print(" HMM-Based Astronaut Movement Prediction Completed Successfully!")
