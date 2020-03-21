#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from collections import Counter
import codecs
import pickle

import numpy as np

def data_to_timesteps(data, steps, shift=1):

  X = data.reshape(data.shape[0], -1)

  Npoints, features = X.shape
  stride0, stride1  = X.strides

  shape   = (Npoints - steps*shift, steps, features)
  strides = (shift*stride0, stride0, stride1)

  X = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
  y = data[steps:]

  return X, y

def to_categorical (arr):
  '''
  Converts a vector of labels into one-hot encoding format

  Parameters
  ----------
    arr : array-like 1D
      array of integer labels (without holes)

  Returns
  -------
  2D matrix in one-hot encoding format
  '''

  n = len(arr)
  uniques, index = np.unique(np.asarray(arr), return_inverse=True)

  categorical = np.zeros(shape=(n, uniques.size), dtype=float)
  categorical[range(0, n), index] = 1.

  return categorical

def from_categorical (categoricals):
  '''
  Convert a one-hot encoding format into a vector of labels

  Parameters
  ----------
    categoricals : array-like 2D
      one-hot encoding format of a label set

  Returns
  -------
  Corresponding labels in 1D array
  '''

  return np.argmax(categoricals, axis=-1)

def find_unique (filename):
  '''
  Find unique characters inside the file
  '''

  with codecs.open(filename,'r','utf8') as fin:
    x = Counter(fin.read())

  return x

def translations (filename):
  '''
  Create dicts {char : int} and {int : char} for every unique character
  '''

  counter = find_unique(filename)

  unique_characters = counter.keys()

  char2int = {c : i for i,c in enumerate(unique_characters)}
  int2char = {i : c for i,c in enumerate(unique_characters)}

  return char2int, int2char

def text_to_one_hot (filename, dictionaries=None):
  '''
  Translate the file into a one hot encoded one
  '''

  if dictionaries is None:
    char2int, int2char = translations (filename)

  else:
    char2int, int2char = dictionaries

  # TODO: faster method
  encoding = np.array([])
  with open(filename, 'r') as f:
    for line in f:
      for char in line:
        encoding = np.append(encoding, char2int[char])

  translation = to_categorical(encoding)

  return translation, char2int, int2char

def one_hot_to_text (arr, int2char, filename):
  '''
  converts a one hot encoded array into text, saved in the data directory.
  '''

  cat  = from_categorical(arr)
  text = ''.join([int2char[i] for i in cat])

  with open(filename, 'w') as f:
      f.write(text)


if __name__ == '__main__':
  pass
