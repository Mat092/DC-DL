#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tensorflow.keras.models import load_model

from translate import from_categorical
from translate import one_hot_to_text

import random

import sys

import pickle


def generate_text_from_model (modelname, out_filename=None, int2char=None, n_char=5000, seed=None):
  '''
  given a model, generate text and save it

  seed is a tensor of characters, if None, a random passage of lenght steps from the first 3e5 divine comedy
    is considered.
  '''

  model = load_model(modelname)

  _, steps, features = model.input_shape

  if seed is None :
    n = random.randint(0, int(3e5))
    text = np.load('./data/divine_comedy.npy')[n:n+steps]
  else :
    assert seed.shape[0] == steps
    text = seed

  print('Generating text')
  for i in range(n_char):

    # reshape to (1, steps, features) and predict
    x_pred = np.reshape(text[-steps:], (1, -1, features))
    preds  = model.predict(x_pred, verbose=0)[0]

    # selection based only on higher values TODO: temperature
    index = np.argmax(preds)

    # one hot encoded version
    code  = np.zeros(shape=features)
    code[index] = 1

    # stacking with previous text
    text = np.vstack([text, code])

  # save to out_filename if given
  if out_filename is not None and int2char is not None:
    one_hot_to_text(text, int2char, out_filename)

  return text

if __name__ == '__main__':

  savefile  = 'data/divine_comedy_out.txt'
  modelname = 'cfg/weights.20.0.92.hdf5'

  with open('data/int2char.pickle', 'rb') as f:
    int2char = pickle.load(open('data/int2char.pickle', 'rb'))

  text = generate_text_from_model(modelname, n_char=5000)
  one_hot_to_text(text, int2char, savefile)
