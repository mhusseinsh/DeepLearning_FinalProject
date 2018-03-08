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

import baselines
import load_prepare_data
import rnn
import plotting

if __name__ == "__main__":


	lrs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
	batches = [8, 16, 32, 64, 128, 256]
	batches_used = []
	lr_used = []
	history = []
	history_scaled = []
	Training = True
	y_pred = []
	y_pred_scaled = []
	y_net = []
	y_net_scaled = []
	best_lr = 0
	best_batch = 0
	best_error = 100
	models = 16
	mse_all = []
	mse_all_scaled = []
	epochs = 1000
	time_steps = int(ra.uniform(5,20))
	random_time_steps = []
	X_rnn = x_copy(X, time_steps)
	X_rnn_scaled = x_copy(X_scaled, time_steps)
	y_rnn = y_select(time_steps)
	random_time_steps.append(time_steps)

	for model in range (models):
		print("model no: ",model)
		batch = np.random.choice(batches)
		learningrate = np.random.choice(lrs)
		batches_used.append(batch)

		lr_used.append(learningrate)
	
		print("Start Training for configs: lr = " + str(learningrate) + " , batch size =" + str(batch))
		history.append(mlp(X_rnn, y_rnn, batch, epochs, time_steps, learning_rate = learningrate, raw = True))
		history_scaled.append(mlp(X_rnn_scaled, y_rnn, batch, epochs, time_steps, learning_rate = learningrate, raw = False))
		
		y_sorted = []
		for i in range (len(y)):
			y_sorted.append(y[i][0])

		indices = np.argsort(y_sorted)[::-1]

		X_sorted = [X[i] for i in indices]
		y_sorted = [y[i] for i in indices]
		X_scaled_sorted = [X_scaled[i] for i in indices]

		# RAW DATA EVALUATION
		# Create linear regression object
		lr = linear_model.LinearRegression()

		# Predict using the fitted model (raw data)
		pred = cross_val_predict(lr, X_sorted, y_sorted, cv=3)

		# Predict using the network (raw data)
		raw = True
		net = evaluate(X_sorted, raw)

		print("using baseline to compare: ")
		# The difference between the mlp and the baseline
		y_error = abs(pred - net)
		print("baseline scores predict raw data: ", np.mean(y_error))

		mse_all.append(mean_squared_error(y, net))
		
		if (mean_squared_error(y, net) < best_error):
			best_error = mean_squared_error(y, net)
			best_batch = batch
			best_lr = learningrate
		
		y_pred.append(pred)
		y_net.append(np.array(net))
		
		# SCALED DATA EVALUATION
		# Create linear regression object
		lr = linear_model.LinearRegression()

		# Predict using the fitted model (raw data)
		pred_scaled = cross_val_predict(lr, X_scaled_sorted, y_sorted, cv=3)

		# Predict using the network (raw data)
		raw = False
		net_scaled = evaluate(X_scaled_sorted, raw)

		print("using baseline to compare: ")
		# The difference between the mlp and the baseline
		y_error = abs(pred_scaled - net_scaled)
		print("baseline scores predict scaled data: ", np.mean(y_error))

		mse_all_scaled.append(mean_squared_error(y, net_scaled))
		
		if (mean_squared_error(y, net_scaled) < best_error):
			best_error = mean_squared_error(y, net_scaled)
			best_batch = batch
			best_lr = learningrate

		y_pred_scaled.append(pred_scaled)
		y_net_scaled.append(np.array(net_scaled))

		print("Randomized time-steps: ",time_steps)

	plot(models)