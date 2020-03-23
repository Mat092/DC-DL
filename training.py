#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

from translate import data_to_timesteps

def continue_training (data_filename, model_name, start_point=None, trainsize=None, epochs=10, batch_size=100):
  '''
  keep training an already existing model.
  '''

  model = load_model(model_name)

  steps = model.input_shape[1]

  if trainsize is None:
    data  = np.load(data_filename)

  elif isinstance(trainsize, int) and start_point is not None:
    data = np.load(data_filename)[start_point:start_point+trainsize]

  else :
    raise ValueError('variable trainsize must be int or None')

  X, y = data_to_timesteps(data, steps)

  filepath       = './cfg/weights.{epoch:02d}.{loss:.2f}.hdf5'
  checkpoint     = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False)
  callbacks_list = [checkpoint]

  history = model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=callbacks_list)

  return model, history


if __name__ == '__main__':

  model_name = 'cfg/weights.20.0.33.hdf5'
  trainsize=100000
  start_point = trainsize
  epochs=10
  batch_size=500
  data_filename = './data/divine_comedy.npy'

  model, history = continue_training(data_filename, model_name, start_point=start_point, trainsize=trainsize, epochs=epochs, batch_size=batch_size)
