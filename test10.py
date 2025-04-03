import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.ar_model import AutoReg
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
import torch

# ==== Autoregression: Predicting Astronaut Adaptation Trends ====
mission_days = np.arange(1, 21)
mobility_efficiency = np.array([85, 86, 84, 83, 81, 82, 80, 78, 76, 75, 74, 73, 71, 70, 69, 68, 66, 65, 64, 63])

# Train autoregression model
model = AutoReg(mobility_efficiency, lags=3)
model_fit = model.fit()
predictions = model_fit.predict(start=20, end=25)  # Predict astronaut mobility for future days

# ==== Gaussian Mixture Model (GMM): Risk Clustering ====
data = np.array([[85, 30], [90, 25], [78, 45], [88, 40], [92, 22], [80, 50]])  # Mobility Efficiency, Stress Level
gmm_model = GaussianMixture(n_components=3, covariance_type="diag", random_state=42)
gmm_model.fit(data)
labels = gmm_model.predict(data)

# ====  Confusion Matrix: AI Mobility Prediction Accuracy ====
true_labels = ["Walk", "Stay", "Robotic Chair", "Walk", "Stay", "Robotic Chair"]
predicted_labels = ["Walk", "Stay", "Robotic Chair", "Stay", "Robotic Chair", "Walk"]

# Compute confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=["Stay", "Walk", "Robotic Chair"])

# ==== Plotting Results ====
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot autoregression: astronaut adaptation prediction
axes[0].plot(mission_days, mobility_efficiency, label="Observed Efficiency", marker="o")
axes[0].plot(np.arange(20, 26), predictions, label="Predicted Efficiency", linestyle="dashed", color="red")
axes[0].set_title("Autoregression: Astronaut Adaptation Prediction")
axes[0].set_xlabel("Mission Days")
axes[0].set_ylabel("Mobility Efficiency (%)")
axes[0].legend()

# Plot GMM: astronaut risk segmentation
scatter = axes[1].scatter(data[:, 0], data[:, 1], c=labels, cmap="coolwarm", marker="o")
axes[1].set_title("Gaussian Mixture Model: Risk Clustering")
axes[1].set_xlabel("Mobility Efficiency (%)")
axes[1].set_ylabel("Stress Level")
plt.colorbar(scatter, ax=axes[1])

# Plot Confusion Matrix: AI mobility accuracy
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Stay", "Walk", "Robotic Chair"], yticklabels=["Stay", "Walk", "Robotic Chair"], ax=axes[2])
axes[2].set_title("Confusion Matrix: AI Mobility Classification")
axes[2].set_xlabel("Predicted Labels")
axes[2].set_ylabel("True Labels")

plt.tight_layout()
plt.show()
