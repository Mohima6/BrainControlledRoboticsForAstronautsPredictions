import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.ar_model import AutoReg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error

# Simulated Time-Series Data for IoT-Based Synchronization
np.random.seed(42)
data_size = 5000
df = pd.DataFrame({
    "timestamp": pd.date_range(start="2025-01-01", periods=data_size, freq="T"),  # Time-based index
    "sync_latency": np.random.uniform(0.1, 1.5, data_size),  # Transmission delay (sec)
    "mobility_speed": np.random.uniform(0.5, 2.5, data_size),  # Robot movement speed (m/s)
    "signal_quality": np.random.uniform(0.2, 0.9, data_size),  # Data transmission quality
    "sync_status": np.random.randint(0, 3, data_size)  # Sync Status: 0=Normal, 1=Delayed, 2=Lost Signal
})
df.set_index("timestamp", inplace=True)  #  Set timestamp as index for time-series analysis

#  Train Autoregression Model for Sync Latency Prediction
train_size = int(len(df) * 0.8)
train_data, test_data = df["sync_latency"][:train_size], df["sync_latency"][train_size:]

#  Fit AutoRegression Model
ar_model = AutoReg(train_data, lags=10)  #  Using previous 10 timestamps for prediction
ar_model_fit = ar_model.fit()
y_pred_ar = ar_model_fit.predict(start=len(train_data), end=len(df) - 1)

mse_ar = mean_squared_error(test_data, y_pred_ar)
print(f"Autoregression Model - MSE: {mse_ar:.4f}")

#  Train Classification Model to Detect Sync Errors
X = df.drop(columns=["sync_status"])
y = df["sync_status"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

clf_model = SVC(kernel="rbf")
clf_model.fit(X_train, y_train)
y_pred_clf = clf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_clf)
print(f"Classification Model - Test Accuracy: {accuracy * 100:.2f}%")

#  Train Clustering Model to Group Sync Patterns
clustering_model = KMeans(n_clusters=3, random_state=42)
df["cluster"] = clustering_model.fit_predict(X_scaled)

#  Create a Single Figure with Three Subplots for Combined Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Single figure with 3 subplots

#  Plot Autoregression Predictions for Sync Latency
axes[0].plot(df.index[train_size:], test_data, label="Actual Latency", color="blue")
axes[0].plot(df.index[train_size:], y_pred_ar, label="Predicted Latency", color="red", linestyle="dashed")
axes[0].set_xlabel("Timestamp")
axes[0].set_ylabel("Sync Latency (Sec)")
axes[0].set_title("Autoregression-Based Sync Latency Prediction")
axes[0].legend()
axes[0].grid(True)

# Plot Classification Accuracy
sns.histplot(y_test, color="blue", alpha=0.5, label="Actual Labels", bins=3, ax=axes[1])
sns.histplot(y_pred_clf, color="red", alpha=0.5, label="Predicted Labels", bins=3, ax=axes[1])
axes[1].set_xlabel("Sync Status (0=Normal, 1=Delayed, 2=Lost Signal)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Classification Model Performance")
axes[1].legend()
axes[1].grid(True)

# Plot Clustering Groups
sns.scatterplot(x=df["mobility_speed"], y=df["signal_quality"], hue=df["cluster"], palette="viridis", ax=axes[2])
axes[2].set_xlabel("Mobility Speed (m/s)")
axes[2].set_ylabel("Signal Quality")
axes[2].set_title("Clustering-Based Sync Pattern Analysis")
axes[2].grid(True)

plt.tight_layout()  #Adjust layout for better spacing
plt.show()
