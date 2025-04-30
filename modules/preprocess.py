import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()#df.dropna() (without inplace=True), it returns a new DataFrame that doesn't contain any rows with NaN values. The original DataFrame (df) remains unchanged.

def scale_data(data: pd.DataFrame):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))#reshape the data to 2D array
    return scaled_data, scaler  # Return both scaled data and the scaler for inverse transformation later

def create_sequences(data,time_step):
    X,Y=[],[]
    for i in range(len(data)-time_step):
        X.append(data[i:i+time_step])
        Y.append(data[i+time_step])
    return np.array(X),np.array(Y)

def split_data(X, Y, test_size=0.2):
    return train_test_split(X, Y, test_size=test_size, shuffle=False)