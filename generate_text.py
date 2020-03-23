#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tensorflow.keras.models import load_model

from translate import from_categorical
from translate import one_hot_to_text

import random

import sys

import pickle


def sample(preds, temperature=1.0):
  '''
  lower temperature == lower diversity
  from :
  https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
  '''

  # helper function to sample an index from a probability array
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

def generate_text_from_model (modelname, out_filename=None, int2char=None, n_char=5000, seed=None, temperature=1e-5):
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

    # selection based only on higher values, now with temperature!
    index = sample(preds, temperature=temperature)

    # one hot encoded version
    code  = np.zeros(shape=features)
    code[index] = 1

    # stacking with previous text
    text = np.vstack([text, code])

  print(f'Finished generation of {n_char} characters')

  # save to out_filename if given
  if out_filename is not None and int2char is not None:
    one_hot_to_text(text, int2char, out_filename)

  return text

if __name__ == '__main__':

  name = 'weights.10.1.48'

  savefile  = 'data/' + name + '.txt'
  modelfile = 'cfg/' + name + '.hdf5'

  with open('data/int2char.pickle', 'rb') as f:
    int2char = pickle.load(open('data/int2char.pickle', 'rb'))

  text = generate_text_from_model(modelfile, n_char=10000, temperature=0.5)

  print('saving')
  one_hot_to_text(text, int2char, savefile)
