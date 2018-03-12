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
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import np_utils
from keras import regularizers
from keras.optimizers import SGD
from keras.optimizers import Adam

#import baselines
from load_prepare_data import *
from mlp import *
from plotting import *

if __name__ == "__main__":

	lrs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7,5e-1, 5e-2, 5e-3, 5e-4, 5e-5, 5e-6, 5e-7]
	#lrs_final = [1e-7, 1e-8]
	#lrs = [[1e-1, 1e-4],[1e-1, 1e-5],[1e-1, 1e-6],[1e-1, 1e-7],[1e-2, 1e-5],[1e-2, 1e-6],[1e-2, 1e-7],[1e-3, 1e-5],[1e-3, 1e-6],[1e-3, 1e-7],
	#		[1e-4, 1e-6],[1e-4, 1e-7],[1e-4, 1e-8],[1e-5, 1e-7]]
	batches = [8, 16, 32]
	alphas = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
	#alphas = [0]
	
	max_depth = [2,4,8,16,32]
	n_estimators = [4,8,16,32]
	bootstrap = [True, False]
	min_samples_leaf = [1,2,4,8]
	models = 36
	epochs = [5000]
	raw = [True]
	scaled = [False]
	

	# Get the data, prepare it for further use
	data, targets, _ = prepare_data()
	
	data_scaled = preprocess_data(data)
	data_sorted, data_scaled_sorted, targets_sorted = sort(data, data_scaled, targets)
	#train_data, train_scaled_data,train_targets,valid_data,valid_scaled_data,valid_targets = get_train_validation_data(data, data_scaled, targets)

	# Random state
	seed = 7
	np.random.seed(seed)

	# Raw model create, cv with random search
	model_raw = KerasRegressor(build_fn = mlp, verbose = 0)

	param_dist = dict(learning_rate=lrs, alpha=alphas, batch_size=batches, num_epochs = epochs, raw = raw)

	random_search_raw = RandomizedSearchCV(estimator = model_raw, random_state = seed, param_distributions=param_dist, n_iter = models)
	
	random_search_raw.fit(data, targets)

	# summarize results of rancom cv 
	print("Best: %f using %s" % (random_search_raw.best_score_, random_search_raw.best_params_))
	means = random_search_raw.cv_results_['mean_test_score']
	stds = random_search_raw.cv_results_['std_test_score']
	params = random_search_raw.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))	

	# create a new model from scratch using the best parameters found
	"""best_model_raw = mlp(learning_rate=random_search.best_params_.get("learning_rate"), 
		alpha=random_search.best_params_.get("alpha"), 
		batch_size=random_search.best_params_.get("batch_size"), 
		num_epochs = random_search.best_params_.get("num_epochs"), 
		raw = random_search.best_params_.get("raw"))

	# train the model
	raw_loss = best_model_raw.fit(train_data, train_targets,
		epochs = random_search.best_params_.get("num_epochs"))

	# train and validation prediction
	val_predictions = best_model_raw.predict(valid_data)
	train_predictions = best_model_raw.predict(train_data)"""
	

	# overall prediction given the whole data
	predictions = random_search_raw.predict(data)
	# MSE
	mse_raw = mean_squared_error(targets, predictions)
				
	# best parameters' list for plotting later
	params_raw = [random_search_raw.best_params_.get("learning_rate"), 
		random_search_raw.best_params_.get("alpha"), 
		random_search_raw.best_params_.get("batch_size"), 
		random_search_raw.best_params_.get("num_epochs"), 
		random_search_raw.best_params_.get("raw")]

	#save_model(best_model_raw, True)


	# Scaled model create, cv with random search
	model_scaled = KerasRegressor(build_fn = mlp, verbose = 0)

	param_dist = dict(learning_rate=lrs, alpha=alphas, batch_size=batches, num_epochs = epochs, raw = scaled)

	random_search_scaled = RandomizedSearchCV(estimator = model_scaled, random_state = seed, param_distributions=param_dist, n_iter = models)
	
	random_search_scaled.fit(data_scaled, targets)

	# summarize results of rancom cv
	print("Best: %f using %s" % (random_search_scaled.best_score_, random_search_scaled.best_params_))
	means = random_search_scaled.cv_results_['mean_test_score']
	stds = random_search_scaled.cv_results_['std_test_score']
	params = random_search_scaled.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))
	
	# create a new model from scratch using the best parameters found
	"""best_model_scaled = mlp(learning_rate=random_search_scaled.best_params_.get("learning_rate"), 
		alpha=random_search_scaled.best_params_.get("alpha"), 
		batch_size=random_search_scaled.best_params_.get("batch_size"), 
		num_epochs = random_search_scaled.best_params_.get("num_epochs"), 
		raw = random_search_scaled.best_params_.get("raw"))

	
	# train the model
	scaled_loss = best_model_scaled.fit(train_scaled_data, train_targets,
		epochs = random_search.best_params_.get("num_epochs"))

	# train and validation prediction
	val_predictions_scaled = best_model_scaled.predict(valid_scaled_data)
	train_predictions_scaled = best_model_scaled.predict(train_scaled_data)
	# overall prediction given the whole data"""
	predictions_scaled = random_search_scaled.predict(data_scaled)
	# MSE
	mse_scaled = mean_squared_error(targets, predictions_scaled)


	# best parameters' list for plotting later
	params_scaled = [random_search_scaled.best_params_.get("learning_rate"), 
		random_search_scaled.best_params_.get("alpha"), 
		random_search_scaled.best_params_.get("batch_size"), 
		random_search_scaled.best_params_.get("num_epochs"), 
		random_search_scaled.best_params_.get("raw")]	
	#save_model(best_model_scaled, False)

	# some plots	
	#plot_loss(raw_loss.history['loss'], scaled_loss.history['loss'], params_raw, params_scaled, mse_raw, mse_scaled)
	plot_network_vs_true(targets, predictions, predictions_scaled, params_raw, params_scaled, mse_raw, mse_scaled)

	# Baseline as random forest
	rf_raw = RandomForestRegressor()

	param_dist_baseline = dict(max_depth=max_depth, n_estimators=n_estimators, bootstrap=bootstrap, min_samples_leaf = min_samples_leaf)

	baseline_random_search = RandomizedSearchCV(estimator = rf_raw, random_state = seed, param_distributions=param_dist_baseline, n_iter = models)
	targets_for_baseline = targets
	baseline_random_search.fit(data, targets_for_baseline.reshape(targets.shape[0],))

	# summarize results
	print("Best: %f using %s" % (baseline_random_search.best_score_, baseline_random_search.best_params_))
	means = baseline_random_search.cv_results_['mean_test_score']
	stds = baseline_random_search.cv_results_['std_test_score']
	params = baseline_random_search.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))	

	"""best_baseline_model_raw = RandomForestRegressor(max_depth=baseline_random_search.best_params_.get("max_depth"), 
		n_estimators=baseline_random_search.best_params_.get("n_estimators"), 
		bootstrap=baseline_random_search.best_params_.get("bootstrap"), 
		min_samples_leaf = baseline_random_search.best_params_.get("min_samples_leaf"))

	baseline_raw_loss = best_baseline_model_raw.fit(train_data, train_targets.reshape(train_targets.shape[0],))
	#print(baseline_raw_loss)
	
	baseline_val_predictions = best_baseline_model_raw.predict(valid_data)
	baseline_train_predictions = best_baseline_model_raw.predict(train_data)"""
	baseline_predictions = baseline_random_search.predict(data)
	mse_baseline = mean_squared_error(targets, baseline_predictions)
	plot_baseline_vs_true(targets, baseline_predictions,mse_baseline)
	