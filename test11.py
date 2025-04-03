import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from pykalman import KalmanFilter

# ==== Physics-Informed Neural Network (PINNs): Mechanical Adaptation ====
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Simulated astronaut terrain adaptation data (terrain steepness & gravity influence)
terrain_data = torch.tensor([[0.1, 9.8], [0.3, 9.8], [0.5, 9.8], [0.2, 3.7], [0.4, 3.7], [0.6, 3.7]], dtype=torch.float32)
mobility_efficiency = torch.tensor([[95], [90], [85], [80], [75], [70]], dtype=torch.float32)

# Train PINN model
pinn_model = PINN()
optimizer = optim.Adam(pinn_model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(300):
    optimizer.zero_grad()
    output = pinn_model(terrain_data)
    loss = criterion(output, mobility_efficiency)
    loss.backward()
    optimizer.step()

# ==== Kalman Filter: AI Movement Tracking Optimization ====
kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1], initial_state_mean=mobility_efficiency[0])
filtered_state_means, _ = kf.filter(mobility_efficiency.numpy())

# ==== Plotting AI-Based Mechanical & Mobility Adaptation ====
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot PINN-Based Terrain Adaptation
terrain_labels = ["Flat Surface", "Rocky Terrain", "Sloped Path", "Lunar Flat", "Lunar Rocky", "Lunar Sloped"]
axes[0].plot(terrain_labels, mobility_efficiency.numpy().flatten(), marker="o", label="Observed Mobility", color="blue")
axes[0].plot(terrain_labels, pinn_model(terrain_data).detach().numpy().flatten(), linestyle="dashed", label="Predicted Mobility", color="red")
axes[0].set_title("Physics-Informed Neural Networks: Terrain Adaptation")
axes[0].set_xlabel("Mission Terrain")
axes[0].set_ylabel("Mobility Efficiency (%)")
axes[0].legend()

# Plot Kalman Filter-Based AI Movement Optimization
axes[1].plot(mobility_efficiency.numpy(), marker="o", label="Observed Movement", color="purple")
axes[1].plot(filtered_state_means, linestyle="dashed", label="Kalman Filter Prediction", color="green")
axes[1].set_title("Kalman Filter: AI Movement Optimization")
axes[1].set_xlabel("Observation Steps")
axes[1].set_ylabel("Mobility Efficiency (%)")
axes[1].legend()

plt.tight_layout()
plt.show()
