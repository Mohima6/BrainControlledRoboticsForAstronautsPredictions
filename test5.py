import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

#  Simulated EEG Signal Dataset
np.random.seed(42)
data_size = 5000
df = pd.DataFrame({
    "eeg_signal_1": np.random.rand(data_size),
    "eeg_signal_2": np.random.rand(data_size),
    "eeg_signal_3": np.random.rand(data_size),
    "motion_intent": np.random.randint(0, 5, data_size)  # Simulated movement intent classes (0-4)
})

#  Preprocess Data
X = df.drop(columns=["motion_intent"])
y = df["motion_intent"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#  Train SVM Model with RBF Kernel
svm_model = SVC(kernel="rbf", C=1.0, gamma="scale")
svm_model.fit(X_train, y_train)

#  Make Predictions
y_pred = svm_model.predict(X_test)

#  Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")  # Display accuracy percentage
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#  Visualize Model Performance
plt.figure(figsize=(10, 6))
plt.hist(y_test, alpha=0.6, label="True Labels", bins=np.arange(-0.5, 5.5, 1))
plt.hist(y_pred, alpha=0.6, label="Predicted Labels", bins=np.arange(-0.5, 5.5, 1))
plt.xlabel("Motion Intent Class")
plt.ylabel("Frequency")
plt.title("SVM-Based EEG Motion Prediction – Comparison of True vs. Predicted")
plt.legend()
plt.grid(True)
plt.show()

print(" SVM-Based Motion Control Prediction Completed Successfully!")
