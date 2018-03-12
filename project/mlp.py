import os
import sys
import glob
import json

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

from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras import regularizers
from keras.optimizers import Adam
from keras.regularizers import l1,l2

def mlp(batch_size, num_epochs, learning_rate, raw, alpha):
	model = Sequential()

	model.add(Dense(64, input_dim = 5, kernel_initializer = 'random_uniform', 
		bias_initializer = 'zeros', activation = 'relu', kernel_regularizer=l2(alpha)))
		#bias_initializer = 'zeros', activation = 'relu', kernel_regularizer=l1(alpha)))

	model.add(Dense(64, kernel_initializer = 'random_uniform', 
		bias_initializer = 'zeros', activation = 'relu', kernel_regularizer=l2(alpha)))
		#bias_initializer = 'zeros', activation = 'relu', kernel_regularizer=l1(alpha)))

	model.add(Dense(1, kernel_initializer = 'random_uniform'))

	#initial_lr = learning_rate[0]
	#final_lr = learning_rate[1]
	#decay_factor = (initial_lr - final_lr)/num_epochs
	#adam = Adam(lr=learning_rate, decay = decay_factor)
	adam = Adam(lr=learning_rate)
	
	model.compile(loss = 'mean_squared_error', optimizer = 'adam')

	return model

def save_model(model, raw):
	if raw == True:
		model_json = model.to_json()
		with open("model_raw.json", "w") as json_file:
			json_file.write(model_json)

		model.save_weights("model_raw.h5")
		#print("raw model is saved")
	else:
		model_json = model.to_json()
		with open("model_pre.json", "w") as json_file:
			json_file.write(model_json)

		model.save_weights("model_pre.h5")
		#print("preprocessed model is saved")

"""def predict(X, raw):
	if (raw):
		json_file = open('model_raw.json', 'r')
		model = json_file.read()
		json_file.close()
		ml = model_from_json(model)
		ml.load_weights("model_raw.h5")
		print("Raw Model Loaded")
		y_net = []
		for x_test, y_test in zip(X, y):
			x_test = x_test.reshape(-1, 5)
			y_pred = ml.predict(x_test)
			y_net.append(y_pred[0])
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
			x_test = x_test.reshape(-1, 5)
			y_pred = ml.predict(x_test)
			y_net.append(y_pred[0])
		return y_net"""