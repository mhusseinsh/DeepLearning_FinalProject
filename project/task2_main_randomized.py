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
from rnn_randomized import *
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
	models = 2
	mse_all = []
	mse_all_scaled = []
	epochs = [10]
	select_time = 5
	time_steps = [select_time]
	random_time_steps = []
	random_lengths = False
	time = [5, 10, 20, 30]
	shuffle = [False]
	
	
	# get data
	data, targets, targets_original = prepare_data()
	data_scaled = preprocess_data(data)
	#m_train_data, m_train_scaled_data, m_train_targets, m_valid_data, m_valid_scaled_data, m_valid_targets = get_train_validation_data(data, data_scaled, targets_original)

	# prepare data for random sequence length
	data_randomly_replicated, targets_randomly_selected, random_list = data_randomize(data_scaled, targets_original)
	
	# prepare data for cv 
	rnn_input = prepare_rnn_input_random(data_randomly_replicated, targets_randomly_selected).reshape(-1, select_time-1, 1 + data.shape[1])

	rnn_targets = prepare_rnn_targets_random(targets_randomly_selected, rnn_input.shape[0]).reshape(-1,1)

	# prepare data for train and validation splits
	"""train_data_random, train_targets_random = data_randomize(m_train_scaled_data, m_train_targets)
				valid_data_random, valid_targets_random = data_randomize(m_valid_scaled_data, m_valid_targets)
			
				train_rnn_input = prepare_rnn_input_random(train_data_random, train_targets_random).reshape(-1, 1, 1 + data.shape[1])
				train_rnn_targets = prepare_rnn_targets_random(train_targets_random, train_rnn_input.shape[0]).reshape(-1,1)
			
				valid_rnn_input = prepare_rnn_input_random(valid_data_random, valid_targets_random).reshape(-1, 1, 1 + data.shape[1])
				valid_rnn_targets = prepare_rnn_targets_random(valid_targets_random, valid_rnn_input.shape[0]).reshape(-1,1)
			"""
	# randomness
	"""seed = 7
				np.random.seed(seed)
			
				# kreas wrapper
				model = KerasRegressor(build_fn = rnn, verbose = 0)
			
				# parameters to be used in random search cv
				param_dist = dict(learning_rate=decaying_lrs, batch_size=batches, num_epochs = epochs, time_steps = time_steps, alpha = alphas, shuffle = shuffle)
			
				random_search = RandomizedSearchCV(estimator = model, random_state = seed, param_distributions=param_dist, n_iter = models)
				
				random_search.fit(rnn_input, rnn_targets)
			
				# summarize results
				print("Best: %f using %s" % (random_search.best_score_, random_search.best_params_))
				means = random_search.cv_results_['mean_test_score']
				stds = random_search.cv_results_['std_test_score']
				params = random_search.cv_results_['params']
				for mean, stdev, param in zip(means, stds, params):
					print("%f (%f) with: %r" % (mean, stdev, param))	
			"""
	model = rnn(learning_rate=, 
		batch_size=, 
		num_epochs = ,
		time_steps = ,
		alpha=r)

	for x, y in zip(rnn_input, rnn_targets):

		model.fit(x, y, batch_size = 1, shuffle = False)
		model.reset_states()

	# loss of training after the best model is found
	"""loss = best_model.train_on_batch(train_rnn_input.reshape(-1, 1, 1 + data.shape[1]), 
					train_rnn_targets.reshape(-1,1),
					epochs = random_search.best_params_.get("num_epochs"))"""
	"""loss = best_model.fit(train_rnn_input.reshape(-1, 1, 1 + data.shape[1]), 
								train_rnn_targets.reshape(-1,1),
					epochs = random_search.best_params_.get("num_epochs"),
					batch_size=random_search.best_params_.get("batch_size"))
				
				save_model(best_model, select_time)
			
				val_predictions = best_model.predict(valid_rnn_input.reshape(-1, 1, 1 + data.shape[1]))"""
	
	# predict using given data
	predictions = best_model.predict(rnn_input)
	print(rnn_input.shape, len(predictions))
	# prepare predictions for future
	reshaped_predictions = predictions.reshape(rnn_input.shape[0],-1)
	reshaped_transposed_predictions = predictions.reshape(rnn_input.shape[0],-1).T

	# store all previous predictions 
	all_predictions5 = [p for p in reshaped_transposed_predictions]
	all_predictions10 = [p for p in reshaped_transposed_predictions]
	all_predictions20 = [p for p in reshaped_transposed_predictions]
	all_predictions30 = [p for p in reshaped_transposed_predictions]

	# for 5, 10, 20 and 30, predict the future one by one
	for t in time:
		# if t = 10, predict next 30 
		for s in range(t , 41):
			# an intermediate array for predictions
			last_predictions = np.zeros((rnn_input.shape[0],1))
			# for each data point (265), get the last prediction
			for i in range(data.shape[0]):
				last_predictions[i] = reshaped_predictions[i][-1]

			# prepare new rnn input using the last prediction
			rnn_input_future = prepare_rnn_input_future(data_scaled, last_predictions)

			# make new predictions
			predictions = best_model.predict(rnn_input_future.reshape(-1, 1, 1 + data.shape[1]))

			# prepare predictions for future
			reshaped_predictions = predictions.reshape(data.shape[0],-1)
			reshaped_transposed_predictions = predictions.reshape(data.shape[0],-1).T

			# store prediction of each data point(265)
			for r in reshaped_transposed_predictions:
				if t == 5:
					all_predictions5.append(r)
				elif t == 10:
					all_predictions10.append(r)
				elif t == 20:
					all_predictions20.append(r)
				else:
					all_predictions30.append(r)


	# store best parameter set for plotting
	params = [random_search.best_params_.get("learning_rate"),
		random_search.best_params_.get("batch_size"), 
		random_search.best_params_.get("num_epochs"),
		random_search.best_params_.get("time_steps"),
		random_search.best_params_.get("alpha")]	

	mse = mean_squared_error(targets, predictions.reshape(targets.shape[0],))
	#plot_predictions(predictions.reshape(targets.shape[0],),targets,select_time)
	plot_loss_rnn(loss.history['loss'], params, select_time)	
	plot_rnn_vs_true(targets, predictions, params, select_time)
	plot_learning_curves(all_predictions5, targets_original, params, select_time, 5)
	plot_learning_curves(all_predictions10, targets_original, params, select_time, 10)
	plot_learning_curves(all_predictions20, targets_original, params, select_time, 20)
	plot_learning_curves(all_predictions30, targets_original, params, select_time, 30)
	