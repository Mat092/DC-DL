#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tensorflow.keras.models import load_model

from translate import from_categorical
from translate import one_hot_to_text

import pickle

savefile = 'data/divine_comedy_out.txt'

model = load_model('cfg/weights.hdf5')

model.summary()

steps = 100

int2char = pickle.load(open('data/int2char.pickle', 'rb'))

features = len(int2char)
n_seed = steps

start = np.load('data/divine_comedy.npy')[:steps]
# start = np.zeros(shape=(n_seed, n_char))
seed  = np.random.randint(low=0, high=features, size=n_seed)

text = start
text_string = ''.join([int2char[i] for i in from_categorical(text)])

for i in range(10000):
  x_pred = np.reshape(text[-steps:], (1, -1, features))
  preds = model.predict(x_pred, verbose=0)[0]
  index = np.argmax(preds)
  code  = np.zeros(shape=features)
  code[index] = 1
  text = np.vstack([text, code])
  print('number : ', i, flush=True)

one_hot_to_text(text, int2char, savefile)

print('Finished')
