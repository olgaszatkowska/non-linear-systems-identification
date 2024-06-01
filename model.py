import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


class SequentialModel:
    def __init__(self, window_size: int) -> None:
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
        self.add(LSTM(50))
        self.add(Dense(1))
