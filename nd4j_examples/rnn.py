from keras.layers import *
from keras.models import Model

import jumpy as jp

arr = jp.zeros((100, 3, 2))

x = Input((3, 2))
y = LSTM(5)(x)
y = Dense(10)(y)

model = Model(x, y)

model.predict(arr)


