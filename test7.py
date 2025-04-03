import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Enable GPU for faster training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Custom Energy Optimization Environment
class EnergyOptimizationEnv(gym.Env):
    def __init__(self):
        super(EnergyOptimizationEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(3)  # 0=Reduce Power, 1=Maintain, 2=Increase
        self.observation_space = gym.spaces.Box(low=10, high=90, shape=(1,), dtype=np.float32)
        self.energy_level = 50  # Initial power level
        self.history = []  #  Track optimization over time

    def step(self, action):
        if action == 0:  # Reduce power
            self.energy_level -= np.random.uniform(5, 15)
        elif action == 2:  # Increase power
            self.energy_level += np.random.uniform(5, 15)

        self.energy_level = np.clip(self.energy_level, 10, 90)  # Keep within safe limits
        reward = -abs(self.energy_level - 50)  # Reward closer to optimal 50%

        terminated = False  #  Continuous optimization
        truncated = False  #  Stable-Baselines3 compatibility
        info = {}  #  Additional info (can stay empty)

        self.history.append(self.energy_level)  #  Store energy levels for plotting
        return np.array([self.energy_level], dtype=np.float32), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  #  Ensure compatibility
        self.energy_level = 50
        self.history.clear()  #  Clear history for new run
        return np.array([self.energy_level], dtype=np.float32), {}


# Initialize RL Environment
env = EnergyOptimizationEnv()

#  Train AI using PPO on GPU
model = PPO("MlpPolicy", env, verbose=1, device=device)
model.learn(total_timesteps=10000)

# Test AI Energy Optimization & Store Data for Plotting
obs, _ = env.reset()
energy_levels = []
for _ in range(20):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    energy_levels.append(obs[0])  # Store optimized energy levels

# Plot AI Energy Optimization Over Time
plt.figure(figsize=(10, 5))
plt.plot(range(1, 21), energy_levels, marker='o', linestyle='-', color='blue', label="Optimized Energy Level")
plt.axhline(y=50, color='r', linestyle='--', label="Ideal Energy Level")
plt.xlabel("Steps")
plt.ylabel("Energy Level")
plt.title("AI-Powered Energy Efficiency Optimization Over Time")
plt.legend()
plt.grid(True)
plt.show()

print(" AI-Powered Energy Optimization Completed Successfully!")
