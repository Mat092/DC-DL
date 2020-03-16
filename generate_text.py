#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tensorflow.keras.models import Model, load_model

model = load_model('cfg/weights.hdf5')

model.summary()

for i, idx in enumerate(seed):
start[i, idx] = 1

int2char = pickle.load(open('data/int2char.pickle', 'rb'))

text = start
string = ''.join()

for i in range(15000):
  x_pred = np.reshape(text[-steps:], (1, steps, features))

  preds = model.predict(x_pred, verbose=0)[0]
  index = np.argmax(preds)
  code  = np.zeros(shape=features)
  code[index] = 1
  text = np.vstack([text, code])

  string = string.join()

with
