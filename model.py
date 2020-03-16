#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle

from tensorflow.keras.layers import LSTM, Input, Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np

from translate import to_categorical
from translate import from_categorical
from translate import one_hot_to_text

def data_to_timesteps(data, steps, shift=1):

  X = data.reshape(data.shape[0], -1)

  Npoints, features = X.shape
  stride0, stride1  = X.strides

  shape   = (Npoints - steps*shift, steps, features)
  strides = (shift*stride0, stride0, stride1)

  X = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
  y = data[steps:]

  return X, y


name       = 'divine_comedy'
steps      = 50
train_size = int(1e4)

data = np.load('data/' + name + '.npy')
X, y = data_to_timesteps(data, steps=steps)

size, steps, features = X.shape

inp       = Input(shape=X.shape[1:])
lstm1     = LSTM(units=128)(inp)
dense1    = Dense(units=features, activation='softmax')(lstm1)
model     = Model(inputs=[inp], outputs=[dense1])
optimizer = RMSprop(lr=0.01, rho=.9)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

filepath       = 'cfg/weights.hdf5'
checkpoint     = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False)
callbacks_list = [checkpoint]

print('********START TRAINING*********')
model.fit(X[:train_size], y[:train_size], batch_size=32, epochs=5, verbose=1, callbacks=callbacks_list)
print('********END TRAINING***********')
