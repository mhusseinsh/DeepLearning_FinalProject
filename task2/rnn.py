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
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, LSTM
from keras.utils import np_utils
from keras import regularizers
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.initializers import Constant
from keras.regularizers import l1,l2

def rnn(num_epochs, learning_rate, alpha):

	model = Sequential()
	model.add(LSTM(64, return_sequences = True, input_shape = (None, 6)))

	model.add(LSTM(64, return_sequences = True))

	model.add(Dense(64, kernel_initializer = 'random_uniform', 
		bias_initializer = 'zeros', activation = 'relu', kernel_regularizer=l2(alpha)))

	model.add(Dense(64, kernel_initializer = 'random_uniform', 
		bias_initializer = 'zeros', activation = 'relu', kernel_regularizer=l2(alpha)))

	model.add(Dense(1))

	initial_lr = learning_rate[0]
	final_lr = learning_rate[1]
	decay_factor = (initial_lr - final_lr)/num_epochs
	
	adam = Adam(lr=learning_rate, decay = decay_factor)

	model.compile(loss = 'mean_squared_error', optimizer = 'adam')

	return model

def rnn_stateful(num_epochs, learning_rate, alpha):

	model = Sequential()
	model.add(LSTM(64, return_sequences = True, batch_input_shape = (1, None, 6), stateful=True))

	model.add(LSTM(64, return_sequences = False, stateful=True))

	model.add(Dense(64, kernel_initializer = 'random_uniform', 
		bias_initializer = 'zeros', activation = 'relu', kernel_regularizer=l2(alpha)))

	model.add(Dense(64, kernel_initializer = 'random_uniform', 
		bias_initializer = 'zeros', activation = 'relu', kernel_regularizer=l2(alpha)))

	model.add(Dense(1))

	initial_lr = learning_rate[0]
	final_lr = learning_rate[1]
	decay_factor = (initial_lr - final_lr)/num_epochs
	
	adam = Adam(lr=learning_rate, decay = decay_factor)

	model.compile(loss = 'mean_squared_error', optimizer = 'adam')

	return model

def predict(X, raw):
	X = x_copy(np.array(X), time_steps)
	if (raw):
		json_file = open('model_raw.json', 'r')
		model = json_file.read()
		json_file.close()
		ml = model_from_json(model)
		ml.load_weights("model_raw.h5")
		print("Raw Model Loaded")
		y_net = []
		for x_test, y_test in zip(X, y):
			x_test = x_test.reshape(-1, time_steps, 5)
			y_pred = ml.predict(x_test)
			y_net.append(y_pred[0][-1])
		return y_net

	else:
		json_file = open('model_pre.json', 'r')
		model = json_file.read()
		json_file.close()
		ml = model_from_json(model)
		ml.load_weights("model_pre.h5")
		print("Pre Model Loaded")
		y_net = []
		for x_test, y_test in zip(X, y):
			x_test = x_test.reshape(-1, time_steps, 5)
			y_pred = ml.predict(x_test)
			y_net.append(y_pred[0][-1])
		return y_net