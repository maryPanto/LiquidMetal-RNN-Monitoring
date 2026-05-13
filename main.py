import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from scipy.io import loadmat

# Custom module imports
from src.data_loader import prepare_data
from src.models import build_model
from src.processing import process_errors, generate_ground_truth

# 1. Configuration & Paths
MODEL_TYPE = 'lstm'
DIRECTORY = "./results"
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)

# 2. Load Data (Replace with your actual relative path)
# zm1 = loadmat('data/zero_mean_entire1.mat')
# vessel_temp = zm1['zero_mean_entire1'][:, 0]
# ground_truth = np.loadtxt('data/anomalies_labelsZ1.csv', delimiter=',', skiprows=1)[:, 1]

# 3. Setup Data & Build Model
x_train, y_train, x_val, y_val, x_test, test_data, test_indices, scaler = prepare_data(vessel_temp)
model = build_model(MODEL_TYPE, input_shape=(x_train.shape[1], 1))

# 4. Training
history = model.fit(
    x_train, y_train, 
    epochs=20, 
    batch_size=4096, 
    validation_data=(x_val, y_val),
    verbose=1
)

# 5. Predictions & Error Processing
train_preds = scaler.inverse_transform(model.predict(x_train))
training_errors = train_preds.flatten() - scaler.inverse_transform(y_train).flatten()

test_preds = scaler.inverse_transform(model.predict(x_test))
# Residuals between predicted and actual test values
errors = np.abs(test_preds.flatten() - test_data[1:])

# Apply your EWMA and Thresholding logic
anomalies, smoothed_errors = process_errors(errors, training_errors, alpha=0.1)

# 6. Evaluation & Metrics
detected_anomalies_idx = test_indices[anomalies]
detected_labels = np.zeros_like(vessel_temp)
detected_labels[detected_anomalies_idx] = 1

# Filter ground truth and detections to the test set only
test_gt = ground_truth[test_indices]
test_det = detected_labels[test_indices]

print(f"\n--- {MODEL_TYPE.upper()} Performance Report ---")
print(classification_report(test_gt, test_det))

# 7. Visualization: Confusion Matrix
conf_matrix = confusion_matrix(test_gt, test_det)
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.savefig(os.path.join(DIRECTORY, f'ConfusionMatrix_{MODEL_TYPE}.png'))
plt.show()

# 8. Saving Results to CSV
# Save Loss History
loss_df = pd.DataFrame({
    'Epoch': range(1, len(history.history['loss']) + 1),
    'Training Loss': history.history['loss'],
    'Validation Loss': history.history['val_loss']
})
loss_df.to_csv(os.path.join(DIRECTORY, f'loss_history_{MODEL_TYPE}.csv'), index=False)

# Save Detected vs Actual Anomalies
results_df = pd.DataFrame({
    'Index': test_indices,
    'Detected': test_det,
    'Actual': test_gt
})
results_df.to_csv(os.path.join(DIRECTORY, f'Anomalies_Comparison_{MODEL_TYPE}.csv'), index=False)

print(f"Results and plots saved to {DIRECTORY}")
