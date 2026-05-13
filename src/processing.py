import numpy as np

def generate_ground_truth(vessel_temp, lower_p=5, upper_p=95):
    """
    Labels anomalies based on 5th and 95th percentile thresholds.
    """
    threshold_low = np.percentile(vessel_temp, lower_p)
    threshold_high = np.percentile(vessel_temp, upper_p)
    
    # 1 for anomaly, 0 for normal
    labels = np.where((vessel_temp > threshold_high) | (vessel_temp < threshold_low), 1, 0)
    return labels, threshold_low, threshold_high

def ewma(errors, alpha=0.2):
    smoothed_errors = [errors[0]]
    for error in errors[1:]:
        smoothed_error = alpha * error + (1 - alpha) * smoothed_errors[-1]
        smoothed_errors.append(smoothed_error)
    return np.array(smoothed_errors)

def process_errors(errors, training_errors, alpha, z_factor=3):
    """
    Error processing using EWMA and dynamic thresholding.
    """
    smoothed_errors = ewma(errors, alpha=alpha)
    smoothed_training_errors = ewma(training_errors, alpha=0.1)
    
    # Static threshold based on standard deviation
    thresholds1 = np.mean(np.abs(smoothed_errors)) + z_factor * np.std(np.abs(smoothed_errors))
    
    # IQR-based filtering for training errors to get a clean baseline
    q1 = np.percentile(smoothed_training_errors, 25)
    q3 = np.percentile(smoothed_training_errors, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    filtered_training_errors = training_errors[(smoothed_training_errors > lower_bound) & (smoothed_training_errors < upper_bound)]
    thresholds2 = np.mean(np.abs(filtered_training_errors)) + 2 * np.std(np.abs(filtered_training_errors))
    
    anomalies = np.where((np.abs(smoothed_errors) > thresholds1) | (np.abs(smoothed_errors) > thresholds2))
    return anomalies, smoothed_errors
