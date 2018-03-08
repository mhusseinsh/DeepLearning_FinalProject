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
from keras.regularizers import l1,l2

def rnn(batch_size, num_epochs, learning_rate, time_steps, alpha):

	model = Sequential()
	#model.add(LSTM(64, return_sequences = True, input_shape = (time_steps - 1, 6)))
	#model.add(LSTM(64, return_sequences = True, input_shape = (1, None)))
	#model.add(Masking(mask_value=0., input_shape=(1, 6)))
	model.add(LSTM(64, return_sequences = True, input_shape = (1, 6)))
	
	model.add(LSTM(64, return_sequences = True))
	model.add(Flatten())

	model.add(Dense(64, kernel_initializer = 'random_uniform', 
		bias_initializer = 'zeros', activation = 'relu', kernel_regularizer=l2(alpha)))

	model.add(Dense(64, kernel_initializer = 'random_uniform', 
		bias_initializer = 'zeros', activation = 'relu', kernel_regularizer=l2(alpha)))
	model.add(Dense(1, kernel_initializer = 'random_uniform'))
	
	initial_lr = learning_rate[0]
	final_lr = learning_rate[1]
	decay_factor = (initial_lr - final_lr)/num_epochs
	
	adam = Adam(lr=learning_rate, decay = decay_factor)
	#adam = Adam(lr=learning_rate)
	model.compile(loss = 'mean_squared_error', optimizer = 'adam')

	#history = model.fit(X, y, epochs = num_epochs, batch_size = batch_size, validation_split = 0.1, shuffle = True)

	return model

def save_model(model, select_time):
	model_json = model.to_json()
	with open("time step " + str(select_time)+" model.json", "w") as json_file:
		json_file.write(model_json)

	model.save_weights("time step " + str(select_time)+" model.h5")

"""def predict(X):
	X = x_copy(np.array(X), time_steps)
	
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
	return y_net"""