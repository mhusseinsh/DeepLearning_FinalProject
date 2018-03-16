import os
import sys
import glob
import json
import itertools
import math
import random as ra

import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import *
from keras.utils import np_utils
from keras import regularizers
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.initializers import *
from keras.regularizers import l1,l2

def rnn(num_epochs, learning_rate, alpha):

	model = Sequential()
	model.add(LSTM(64, return_sequences = True, input_shape = (None, 6), 
		kernel_initializer = RandomUniform(0.01, 0.05),kernel_regularizer=l2(alpha)))
	model.add(GaussianNoise(0.02))
	model.add(LSTM(64, return_sequences = True, 
		kernel_initializer = RandomUniform(0.01, 0.05),kernel_regularizer=l2(alpha)))
	model.add(GaussianNoise(0.02))
	model.add(Dense(64, kernel_initializer = RandomUniform(0.01, 0.05), 
		bias_initializer = Constant(0.1), activation = 'relu', kernel_regularizer=l2(alpha)))
	model.add(GaussianNoise(0.02))
	model.add(Dense(64, kernel_initializer = RandomUniform(0.01, 0.05), 
		bias_initializer = Constant(0.1), activation = 'relu', kernel_regularizer=l2(alpha)))
	#model.add(GaussianNoise(0.02))

	model.add(Dense(1))

	initial_lr = learning_rate[0]
	final_lr = learning_rate[1]
	decay_factor = (initial_lr - final_lr)/num_epochs
	
	adam = Adam(lr=learning_rate, decay = decay_factor)

	model.compile(loss = 'mean_squared_error', optimizer = 'adam')

	return model

def rnn_stateful(num_epochs, learning_rate, alpha):

	model = Sequential()
	model.add(LSTM(64, return_sequences = True, batch_input_shape = (1, None, 6), stateful=True, 
		kernel_initializer = RandomUniform(0.01, 0.05), kernel_regularizer=l2(alpha)))
	model.add(GaussianNoise(0.02))
	model.add(LSTM(64, return_sequences = False, stateful=True,
		kernel_initializer = RandomUniform(0.01, 0.05), kernel_regularizer=l2(alpha)))
	model.add(GaussianNoise(0.02))
	model.add(Dense(64, kernel_initializer = RandomUniform(0.01, 0.05),
		bias_initializer = Constant(0.1), activation = 'relu', kernel_regularizer=l2(alpha)))
	model.add(GaussianNoise(0.02))
	model.add(Dense(64, kernel_initializer = RandomUniform(0.01, 0.05), 
		bias_initializer = Constant(0.1), activation = 'relu', kernel_regularizer=l2(alpha)))

	model.add(Dense(1))

	initial_lr = learning_rate[0]
	final_lr = learning_rate[1]
	decay_factor = (initial_lr - final_lr)/num_epochs
	
	adam = Adam(lr=learning_rate, decay = decay_factor)

	model.compile(loss = 'mean_squared_error', optimizer = 'adam')

	return model

