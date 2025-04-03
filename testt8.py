import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf

# ==== LSTM Model for Astronaut Adaptation Prediction ====
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Extract final time-step output
        return x

# Simulated astronaut adaptation trends on Moon & Mars (efficiency over time)
adaptation_data = torch.tensor([[0.85], [0.86], [0.84], [0.80], [0.78], [0.75], [0.73], [0.70]], dtype=torch.float32)
model_lstm = LSTMModel()
optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop (dummy training for demonstration)
for epoch in range(100):
    optimizer_lstm.zero_grad()
    output = model_lstm(adaptation_data.unsqueeze(0))
    loss = criterion(output, adaptation_data[-1])
    loss.backward()
    optimizer_lstm.step()

# ==== Bayesian Optimization for Movement Strategy Enhancements ====
# Define astronaut movement optimization parameters (Moon: 0.17G, Mars: 0.38G)
moon_mars_gravity = torch.tensor([[0.17], [0.38]], dtype=torch.float32)
mobility_efficiency = torch.tensor([[85], [78]], dtype=torch.float32)

# Gaussian Process Model for Bayesian Optimization
gp = SingleTaskGP(moon_mars_gravity, mobility_efficiency)
optimizer_gp = ExpectedImprovement(gp, best_f=mobility_efficiency.max())

# Optimize movement efficiency across gravity environments
bounds = torch.tensor([[0.1], [0.5]])  # Gravity range (Moon & Mars)
optimal_point, _ = optimize_acqf(optimizer_gp, bounds=bounds, q=1, num_restarts=5, raw_samples=20)

# ==== Plotting Astronaut Adaptation & Movement Optimization ====
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot astronaut adaptation efficiency across planets
planets = ["Moon", "Mars"]
sns.barplot(x=planets, y=mobility_efficiency.numpy().flatten(), palette="Blues", ax=axes[0])
axes[0].set_title("Astronaut Adaptation Efficiency on Lunar & Martian Terrain")
axes[0].set_xlabel("Space Environment")
axes[0].set_ylabel("Efficiency Score (%)")

# Plot astronaut adaptation trends predicted by LSTM
epochs = np.arange(1, len(adaptation_data)+1)
axes[1].plot(epochs, adaptation_data.numpy(), marker="o", label="Observed")
axes[1].plot(epochs[-1]+1, output.detach().numpy(), "ro", label="Predicted")
axes[1].set_title("LSTM Prediction: Astronaut Mobility Trend")
axes[1].set_xlabel("Time (Mission Days)")
axes[1].set_ylabel("Adaptation Score")
axes[1].legend()

plt.tight_layout()
plt.show()

# Print optimal mobility efficiency improvement
print(f" Optimal mobility strategy improvement (Bayesian Optimization): {optimal_point.item()}")
