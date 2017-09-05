'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.constraints import max_norm, non_neg, unit_norm, min_max_norm

batch_size = 128
num_classes = 10
epochs = 1

# the data, shuffled and split between train and test sets
(x_train, y_train), (X_test, Y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
x_train = x_train.astype('float32')
X_test = X_test.astype('float32')
x_train /= 255
X_test /= 255
print(x_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,), kernel_constraint=max_norm(2.)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', bias_constraint=non_neg()))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=unit_norm()))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax', bias_constraint=min_max_norm(0.0, 1.0)))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
