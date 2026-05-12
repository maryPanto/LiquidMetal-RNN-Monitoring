import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional

def build_model(model_type, input_shape, dropout_rate=0.2):
    """
    Builds and compiles specialized RNN architectures for reactor monitoring.
    """
    model = Sequential()
    
    if model_type == 'lstm':
        model.add(LSTM(55, input_shape=input_shape, return_sequences=False))
    elif model_type == 'gru':
        model.add(GRU(80, input_shape=input_shape, return_sequences=False))
    elif model_type == 'bilstm':
        model.add(Bidirectional(LSTM(55, input_shape=input_shape)))
    elif model_type == 'bigru':
        model.add(Bidirectional(GRU(80, input_shape=input_shape)))
        
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                  loss='mean_absolute_error')
    return model
