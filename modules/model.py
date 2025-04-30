from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
import numpy as np

def build_model(time_step):
    model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model,X_train,Y_train):
    #Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
    history = model.fit(X_train, Y_train, epochs=100, batch_size=32,validation_split=0.1, callbacks=[early_stop, reduce_lr])
    return history

def predict(model, X_test):
    return model.predict(X_test)


def forecast_future(model, scaled_data, predict_days, time_step):
    input_seq = scaled_data[-time_step:]
    forecast = []

    for _ in range(predict_days):
        pred = model.predict(input_seq.reshape(1, time_step, 1))
        forecast.append(pred[0][0])
        input_seq = np.append(input_seq[1:], pred).reshape(time_step, 1)

    return np.array(forecast).reshape(-1, 1)
