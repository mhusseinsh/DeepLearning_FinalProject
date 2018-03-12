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

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor


from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, LSTM
from keras.utils import np_utils
from keras import regularizers
from keras.optimizers import SGD
from keras.optimizers import Adam

from load_prepare_data import *
from rnn import *
from plotting import *


if __name__ == "__main__":


	decaying_lrs = [[1e-1, 1e-4],[1e-1, 1e-5],[1e-1, 1e-6],[1e-1, 1e-7],[1e-2, 1e-5],[1e-2, 1e-6],[1e-2, 1e-7],[1e-3, 1e-5],[1e-3, 1e-6],[1e-3, 1e-7],
			[1e-4, 1e-6],[1e-4, 1e-7],[1e-4, 1e-8],[1e-5, 1e-7]]
	batches = [8, 16, 32, 64, 128, 256]
	alphas = [1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
	max_depth = [2,4,8,16,32]
	n_estimators = [4,8,16,32]
	bootstrap = [True, False]
	min_samples_leaf = [1,2,4,8]
	models = 3
	epochs = [5]
	select_time = 20
	time_steps = [select_time]
	time = [5, 10, 20, 30]

	# get data
	data, targets, targets_original = prepare_data()
	data_scaled = preprocess_data(data)
	targets_selected = y_select(targets_original, select_time)

	#m_train_data, m_train_scaled_data, m_train_targets, m_valid_data, m_valid_scaled_data, m_valid_targets = get_train_validation_data(data, data_scaled, targets_selected)

	# given time steps, y is prepared 
	targets_selected_for_input = y_select_targets(targets_selected, select_time)
		
	# prepare data for cv
	rnn_input = prepare_rnn_input(data_scaled, targets_selected_for_input, select_time).reshape(-1, select_time-1, 1 + data.shape[1])

	rnn_targets = prepare_rnn_targets(targets_selected, select_time).reshape(-1, select_time-1,1)

	# prepare data for train and validation
	"""train_targets_selected = y_select(m_train_targets, select_time)
	targets_selected_for_input = y_select_targets(targets_selected, select_time)
	
	targets_selected_for_input = y_select_targets(targets_selected, select_time)"""

	"""valid_targets_selected = y_select(m_valid_targets, select_time)
	train_rnn_input = prepare_rnn_input(m_train_scaled_data, train_targets_selected, select_time).reshape(- 1, select_time -1, 1 + data.shape[1])
	train_rnn_targets = prepare_rnn_targets(m_train_scaled_data, select_time).reshape(-1,select_time -1)
	valid_rnn_input = prepare_rnn_input(m_valid_scaled_data, valid_targets_selected, select_time).reshape(-1, select_time -1, 1 + data.shape[1])
	valid_rnn_targets = prepare_rnn_targets(m_valid_scaled_data, select_time).reshape(-1,select_time -1)"""

	# randomness
	seed = 7
	np.random.seed(seed)

	# keras wrapper
	model = KerasRegressor(build_fn = rnn, verbose = 0)

	# parameters to be used in random search cv
	param_dist = dict(learning_rate=decaying_lrs, batch_size=batches, num_epochs = epochs, time_steps = time_steps, alpha = alphas)

	random_search = RandomizedSearchCV(estimator = model, random_state = seed, param_distributions=param_dist, n_iter = models)
	
	random_search.fit(rnn_input, rnn_targets)

	# summarize results
	print("Best: %f using %s" % (random_search.best_score_, random_search.best_params_))
	means = random_search.cv_results_['mean_test_score']
	stds = random_search.cv_results_['std_test_score']
	params = random_search.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))
	exit()	

	"""best_model = rnn(learning_rate=random_search.best_params_.get("learning_rate"), 
		batch_size=random_search.best_params_.get("batch_size"), 
		num_epochs = random_search.best_params_.get("num_epochs"),
		time_steps = random_search.best_params_.get("time_steps"),
		alpha = random_search.best_params_.get("alpha"))"""

	# loss of training after the best model is found
	"""loss = best_model.fit(rnn_input.reshape(-1, select_time-1, 1 + data.shape[1]), 
		rnn_targets.reshape(-1, select_time-1,1),
		epochs = random_search.best_params_.get("num_epochs"),
		batch_size=random_search.best_params_.get("batch_size"))"""

	
	#save_model(best_model, select_time)

	# store best parameter set for plotting
	params = [random_search.best_params_.get("learning_rate"), 
		random_search.best_params_.get("alpha"), 
		random_search.best_params_.get("batch_size"), 
		random_search.best_params_.get("num_epochs"),
		random_search.best_params_.get("time_steps"),
		random_search.best_params_.get("alpha")]	

	#val_predictions = best_model.predict(valid_rnn_input.reshape(-1, select_time -1, 1 + data.shape[1]))

	# predict using given data

	#***********************************************************

	#PREPARE RNN INPUT FOR DIFFERENT LENGTHS

	#***********************************************************
	for t in time:
		targets_selected = y_select(targets_original, t)
		targets_selected_for_input = y_select_targets(targets_selected, t)
		rnn_input_new = prepare_rnn_input(data_scaled, targets_selected_for_input, t).reshape(-1, t-1, 1 + data.shape[1])
						
		predictions = random_search.predict(rnn_input_new)
		
		# prepare predictions for future
		 
		all_predictions = predictions
		# if t = 10, predict next 30 
		for s in range(t , 41):
			# an intermediate array for predictions
			last_predictions = np.zeros((data.shape[0],1))
			# for each data point (265), get the last prediction
			for i in range(data.shape[0]):
				last_predictions[i] = predictions[i][-1]

			# prepare new rnn input using the last prediction
			rnn_input_future = prepare_rnn_input_future(data_scaled, last_predictions).reshape(-1, 1, 1 + data.shape[1])

			# make new predictions
			predictions = random_search.predict(rnn_input_future).reshape(data_scaled.shape[0],1)
			
			all_predictions = np.insert(all_predictions,-1,predictions.T, axis = 1)
		
		mse_all =[]
		for z in range(data.shape[0]):
			mse_all.append(mean_squared_error(targets_original[z], all_predictions[z]))

		mse_min = np.argmin(mse_all)
		mse_max = np.argmax(mse_all)

		mse = mean_squared_error(targets_original, all_predictions)
		plot_learning_curves(all_predictions[mse_max], targets_original[mse_max],
			all_predictions[mse_min], targets_original[mse_min], params, t, mse_all[mse_max], mse_all[mse_min],select_time)

		plot_rnn_vs_true(targets, predictions, params, select_time, t, mse_all[-1])
	"""
				# Baseline as random forest
				rf_raw = RandomForestRegressor()
			
				param_dist_baseline = dict(max_depth=max_depth, n_estimators=n_estimators, bootstrap=bootstrap, min_samples_leaf = min_samples_leaf)
			
				baseline_random_search = RandomizedSearchCV(estimator = rf_raw, random_state = seed, param_distributions=param_dist_baseline, n_iter = models)
				y40 = []
				for y in targets:
					y40.append(y[-1])
				y40_list = [y.tolist() for y in y40]
				baseline_random_search.fit(targets_selected, y40_list)
			
				# summarize results
				print("Best: %f using %s" % (baseline_random_search.best_score_, baseline_random_search.best_params_))
				means = baseline_random_search.cv_results_['mean_test_score']
				stds = baseline_random_search.cv_results_['std_test_score']
				params = baseline_random_search.cv_results_['params']
				for mean, stdev, param in zip(means, stds, params):
					print("%f (%f) with: %r" % (mean, stdev, param))	"""

	"""best_baseline_model_raw = RandomForestRegressor(max_depth=baseline_random_search.best_params_.get("max_depth"), 
		n_estimators=baseline_random_search.best_params_.get("n_estimators"), 
		bootstrap=baseline_random_search.best_params_.get("bootstrap"), 
		min_samples_leaf = baseline_random_search.best_params_.get("min_samples_leaf"))

	baseline_raw_loss = best_baseline_model_raw.fit(train_data, train_targets.reshape(train_targets.shape[0],))
	
	baseline_val_predictions = best_baseline_model_raw.predict(valid_data)
	baseline_train_predictions = best_baseline_model_raw.predict(train_data)"""
	"""
		baseline_predictions = baseline_random_search.predict(targets_selected)
		mse_baseline = mean_squared_error(targets, baseline_predictions)
		plot_baseline_vs_true(targets, baseline_predictions,mse_baseline)

		# Last baseline
		rf_last = RandomForestRegressor()

		param_dist_baseline = dict(max_depth=max_depth, n_estimators=n_estimators, bootstrap=bootstrap, min_samples_leaf = min_samples_leaf)

		baseline_random_search_last = RandomizedSearchCV(estimator = rf_last, random_state = seed, param_distributions=param_dist_baseline, n_iter = models)

		inputs_last_baseline, targets_last_baseline = prepare_last_baseline(targets_original)
		print(inputs_last_baseline.shape, targets_last_baseline.shape)
		
		baseline_random_search_last.fit(inputs_last_baseline,targets_last_baseline)

		baseline_predictions = baseline_random_search_last.predict(inputs_last_baseline)
		#mse_baseline = mean_squared_error(targets, baseline_predictions)
		#plot_baseline_vs_true(targets, baseline_predictions,mse_baseline)
	"""