import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_data(vessel_temp, train_start=205000, train_end=983800, validation_ratio=0.2):
    # Split indices
    train_val_data = vessel_temp[train_start:train_end]
    validation_index = int((1 - validation_ratio) * len(train_val_data))
    
    training_data = train_val_data[:validation_index]
    validation_data = train_val_data[validation_index:]
    
    test_indices = np.concatenate((np.arange(0, train_start), np.arange(train_end, len(vessel_temp))))
    test_data = vessel_temp[test_indices]
    
    scaler = StandardScaler()
    training_data_norm = scaler.fit_transform(training_data.reshape(-1, 1))
    validation_data_norm = scaler.transform(validation_data.reshape(-1, 1))
    test_data_norm = scaler.transform(test_data.reshape(-1, 1))
    
    # Input-Output pairs (predicting the next time step)
    x_train, y_train = training_data_norm[:-1], training_data_norm[1:]
    x_val, y_val = validation_data_norm[:-1], validation_data_norm[1:]
    x_test = test_data_norm[:-1]
    
    return x_train, y_train, x_val, y_val, x_test, test_data, test_indices, scaler
