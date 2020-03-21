#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle

from tensorflow.keras.layers import LSTM, Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np

from translate import to_categorical
from translate import from_categorical
from translate import one_hot_to_text
from translate import data_to_timesteps

name       = 'divine_comedy'
steps      = 100
train_size = int(10000)

data = np.load('data/' + name + '.npy')[:train_size]
X, y = data_to_timesteps(data, steps=steps)

size, steps, features = X.shape

inp    = Input(shape=X.shape[1:])
lstm1  = LSTM(units=128)(inp)
drop   = Dropout(rate=0.2)(lstm1)
dense1 = Dense(units=features, activation='softmax')(drop)
model  = Model(inputs=[inp], outputs=[dense1])

model.summary()

optimizer = RMSprop(lr=0.01, rho=.9 )
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

filepath       = './cfg/weights.{epoch:02d}.{loss:.2f}.hdf5'
checkpoint     = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False)
callbacks_list = [checkpoint]

print('********START TRAINING*********')
model.fit(X[:], y[:], batch_size=100, epochs=20, verbose=1, callbacks=callbacks_list)
print('********END TRAINING***********')
