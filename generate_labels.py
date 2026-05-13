import numpy as np
import pandas as pd
from scipy.io import loadmat
import os
from src.processing import generate_ground_truth # Import the logic you already have

# 1. Setup Paths (Using relative paths for GitHub)
input_data_path = 'data/zero_mean_entire1.mat'
output_csv_path = 'data/anomalies_labelsZ1.csv'

# 2. Load the raw METL data
zm1 = loadmat(input_data_path)
vessel_temp = zm1['zero_mean_entire1'][:, 0]

# 3. Generate Labels using your 5th/95th percentile method
# This uses the function we put in processing.py earlier
labels, low_thresh, high_thresh = generate_ground_truth(vessel_temp, lower_p=5, upper_p=95)

# 4. Save to CSV for main.py to use
df = pd.DataFrame({
    'timestamp_index': np.arange(len(vessel_temp)),
    'anomaly_label': labels
})

# Ensure the data directory exists
os.makedirs('data', exist_ok=True)
df.to_csv(output_csv_path, index=False)

print(f"Ground Truth Created!")
print(f"Thresholds: Low={low_thresh:.4f}, High={high_thresh:.4f}")
print(f"File saved to: {output_csv_path}")
