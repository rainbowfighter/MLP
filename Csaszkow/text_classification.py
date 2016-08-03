'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
import os
import re

#Reading poems in and preparing dictionary
text_all = ''
folder_name = "Poems"
poems = os.listdir(folder_name)
for poem_name in poems:
    path = folder_name +'/' + poem_name
    file = open(path)
    text = file.read().lower()
    text_all += text
    file.close()
    
regex = re.compile('[0-9?,.:()"!-;\']')
text_all = regex.sub('', text_all)
words = sorted(list(set(text_all.split())))
dictionary = dict((c, i) for i, c in enumerate(words))


X_train = []
y_train = []
X_test = []
y_test = []
thir_party_test = []
petofi_cntr = 0
vajda_cntr = 0

for poem_name in poems:
    lst_temp = []
    path = folder_name +'/' + poem_name
    file = open(path)
    text = file.read().lower()
    text = regex.sub('', text)
    text_lst = list(text.split())
    for element in text_lst:
        element = dictionary[element]
        lst_temp.append(element)
    
    if "petofi" in poem_name:
        petofi_cntr += 1
        if petofi_cntr <= 10:
            y_train.append(0)
            X_train.append(lst_temp)
        else:
            y_test.append(0)
            X_test.append(lst_temp)
    elif "csokonai" in poem_name:
        thir_party_test.append(lst_temp)
    else:
        vajda_cntr += 1
        if vajda_cntr <= 10:
            y_train.append(1)
            X_train.append(lst_temp)
        else:
            y_test.append(1)
            X_test.append(lst_temp)
    
    file.close()


max_words = len(dictionary)
#max_words = 1000
batch_size = 2
nb_epoch = 500

print('Loading data...')
#(X_train, y_train), (X_test, y_test) = reuters.load_data(nb_words=max_words, test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

nb_classes = np.max(y_train)+1
print(nb_classes, 'classes')

print('Vectorizing sequence data...')
tokenizer = Tokenizer(nb_words=max_words)
X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
thir_party_test =  tokenizer.sequences_to_matrix(thir_party_test, mode='binary')
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Convert class vector to binary class matrix (for use with categorical_crossentropy)')
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)

print('Building model...')
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    nb_epoch=nb_epoch, batch_size=batch_size,
                    verbose=1, validation_split=0.1)
score = model.evaluate(X_test, Y_test,
                       batch_size=batch_size, verbose=1)
result = model.predict(X_test, verbose = 1)
result_third_party = model.predict(thir_party_test, verbose = 1)
print('Test score:', score[0])
print('Test accuracy:', score[1])