# Predicting the Efficiency of Solar Cells From Molecular Structure
# Model
#
# SMILES Parsing using Keras-Molecules https://github.com/maxhodak/keras-molecules
#
# Made by Liam Nakagawa 2016

from keras import backend as K
from keras import objectives
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D

class SolModel():

   sol = None

   def create(self, charset_len, weights = None):
      latent_dim = 500
      max_len = 120

      inputs = Input(shape=(max_len, charset_len))

      x = Convolution1D(9, 9, activation='relu', name='conv_1')(inputs)
      x = Convolution1D(9, 9, activation='relu', name='conv_2')(x)
      x = MaxPooling1D(3)(x)

      x = Convolution1D(9, 9, activation='relu', name='conv_3')(x)
      x = Convolution1D(9, 9, activation='relu', name='conv_4')(x)
      x = Dropout(0.2)(x)
      x = Convolution1D(9, 9, activation='relu', name='conv_5')(x)
      x = Convolution1D(9, 9, activation='relu', name='conv_6')(x)

      x = GlobalMaxPooling1D()(x)

      x = Dropout(0.2)(x)
      x = Dense(64, name='dense_1')(x)
      x = Dense(32, name='dense_2')(x)
      x = Dense(1, name='dense_3')(x)

      predictions = Activation('sigmoid')(x)

      self.sol = Model(input=inputs, output=predictions)

      if weights:
         self.sol.load_weights(weights)

      self.sol.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['mean_absolute_error'])

   def save(self, filename):
      self.sol.save_weights(filename)

   def load(self, charset_len, weights):
      self.create(charset_len, weights = weights)
