'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from scipy.misc import imread

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

batch_size = 128
nb_classes = 10
nb_epoch = 10

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

#early stop parameters
early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)



#New test model
X_test_temp = []
image_data = imread('numbers/0.jpg',flatten=1).astype(np.float32)
image_data /= 255
X_test_temp.append(image_data)
image_data = imread('numbers/1.jpg',flatten=1).astype(np.float32)
image_data /= 255
X_test_temp.append(image_data)
image_data = imread('numbers/2.jpg',flatten=1).astype(np.float32)
image_data /= 255
X_test_temp.append(image_data)
image_data = imread('numbers/3.jpg',flatten=1).astype(np.float32)
image_data /= 255
X_test_temp.append(image_data)
image_data = imread('numbers/4.jpg',flatten=1).astype(np.float32)
image_data /= 255
X_test_temp.append(image_data)
image_data = imread('numbers/5.jpg',flatten=1).astype(np.float32)
image_data /= 255
X_test_temp.append(image_data)
image_data = imread('numbers/6.jpg',flatten=1).astype(np.float32)
image_data /= 255
X_test_temp.append(image_data)
image_data = imread('numbers/7.jpg',flatten=1).astype(np.float32)
image_data /= 255
X_test_temp.append(image_data)
image_data = imread('numbers/8.jpg',flatten=1).astype(np.float32)
image_data /= 255
X_test_temp.append(image_data)
image_data = imread('numbers/9.jpg',flatten=1).astype(np.float32)
image_data /= 255
X_test_temp.append(image_data)

X_test = np.array(X_test_temp)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

y_test = np.arange(10)
y_test = np.array(y_test)
Y_test = np_utils.to_categorical(y_test, nb_classes)



#Model construction
model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test), callbacks=[early_stopping])

result = model.predict(X_test, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])