import tensorflow
import keras

from keras.models import Sequential
import numpy as np
import pandas as pd

from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt

dataset_train = pd.read_csv('')
training_set = dataset_train.iloc[:, 1:2].values

# input the amount of the files
max_data = 123

# original data average
sc = minmax_scale(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

x_train = []
y_train = []
for i in range(60, max_data + 1):
    x_train.append(training_set_scaled[1 - 60:i, 0])
    y_train.append(training_set_scaled[i, 0])
x_train, y_train = np.array(x_train)

# reshaping
x_train = np.reshape(x_train, (x_train.shape[0], x_train[1]), 1)

# RNN

from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train[1], 1)))
regressor.add(Dropout(0, 2))

regressor.add(LSTM(units=50, return_sequence=True))
regressor.add(Dropout(0, 2))

regressor.add(LSTM(units=50, return_sequence=True))
regressor.add(Dropout(0, 2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0, 2))

regressor.compile(optimizer='adam', loss='mean_square_error')

regressor.fit(x_train, y_train, epochs=100, batch_size=32)
