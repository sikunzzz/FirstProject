import math
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard

from matplotlib import pyplot as plt

x = np.linspace(0, 2 * math.pi, 1000).reshape(-1, 1)
y = np.sin(x)

model = Sequential()
model.add(Dense(10, activation='tanh', input_dim=1))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['mean_squared_error'])

for i in range(40):
    model.fit(x, y, nb_epoch=25, batch_size=8, verbose=1)
    predictions = model.predict(x)

#    plt.plot(predictions)
#    plt.plot(y)
#    plt.show()

TensorBoard(log_dir='/Graph', histogram_freq=0, write_graph=True, write_images=True)