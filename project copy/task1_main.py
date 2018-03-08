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

	lrs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
	batches = [8, 16, 32, 64, 128, 256]
	alphas = [1e-15, 1e-10, 1e-8, 1e-6, 1e-3, 1e-2]
	batches_used = []
	lr_used = []
	alphas_used = []
	history_train = []
	history_train_scaled = []
	history_valid = []
	history_valid_scaled = []
	cvscores_all = []
	cvscores_scaled_all = []
	network_mse_all = []
	network_mse_scaled_all = []
	baseline_mse_all = []
	baseline_mse_scaled_all = []
	Training = True
	y_pred = []
	y_pred_scaled = []
	y_net = []
	y_net_scaled = []
	best_lr = 0
	best_batch = 0
	best_error = 100
	models = 3
	l2norm_all = []
	epochs = [5]
	raw = [True]
	scaled = [False]

	data, targets, _ = prepare_data()
	data_scaled = preprocess_data(data)
	data_sorted, data_scaled_sorted, targets_sorted = sort(data, data_scaled, targets)
	train_data, train_scaled_data,train_targets,valid_data,valid_scaled_data,valid_targets = get_train_validation_data(data, data_scaled, targets)
				

	seed = 7
	np.random.seed(seed)

	keras_model_raw = KerasRegressor(build_fn = mlp, verbose = 0)

	param_dist = dict(learning_rate=lrs, alpha=alphas, batch_size=batches, num_epochs = epochs, raw = raw)

	random_search = RandomizedSearchCV(estimator = keras_model_raw, random_state = seed, param_distributions=param_dist, n_iter = models)
	
	random_search.fit(data, targets)

	# summarize results
	print("Best: %f using %s" % (random_search.best_score_, random_search.best_params_))
	means = random_search.cv_results_['mean_test_score']
	stds = random_search.cv_results_['std_test_score']
	params = random_search.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))	

	print(random_search.get_params(deep = True))

	exit()

	best_keras_model_raw = mlp(learning_rate=random_search.best_params_.get("learning_rate"), 
		alpha=random_search.best_params_.get("alpha"), 
		batch_size=random_search.best_params_.get("batch_size"), 
		num_epochs = random_search.best_params_.get("num_epochs"), 
		raw = random_search.best_params_.get("raw"))
	raw_loss = best_keras_model_raw.fit(train_data, train_targets)
	val_predictions = model_raw.predict(valid_data)

	predictions = model_raw.predict(data)
				


	keras_model_scaled = KerasRegressor(build_fn = mlp, verbose = 0)

	param_dist = dict(learning_rate=lrs, alpha=alphas, batch_size=batches, num_epochs = epochs, raw = scaled)

	random_search_scaled = RandomizedSearchCV(estimator = keras_model_scaled, random_state = seed, param_distributions=param_dist, n_iter = models)
	
	random_search_scaled.fit(data_scaled, targets)

	# summarize results
	print("Best: %f using %s" % (random_search_scaled.best_score_, random_search_scaled.best_params_))
	means = random_search_scaled.cv_results_['mean_test_score']
	stds = random_search_scaled.cv_results_['std_test_score']
	params = random_search_scaled.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))

	random_search_scaled.best_params_.get("raw")
	

	best_keras_model_scaled = mlp(learning_rate=random_search_scaled.best_params_.get("learning_rate"), 
		alpha=random_search_scaled.best_params_.get("alpha"), 
		batch_size=random_search_scaled.best_params_.get("batch_size"), 
		num_epochs = random_search_scaled.best_params_.get("num_epochs"), 
		raw = random_search_scaled.best_params_.get("raw"))
	scaled_loss = best_keras_model_scaled.fit(train_scaled_data, train_targets)
	val_predictions_scaled = model_scaled.predict(valid_scaled_data)

	predictions_scaled = model_scaled.predict(data_scaled)

	#plot_loss(history_train, history_train_scaled, lr_used, batches_used, best_mse, best_mse_scaled, network_mse_all, network_mse_scaled_all)

	lr = linear_model.LinearRegression()
	lr.fit(train_data, train_targets)
	lr_pred = lr.predict(valid_data)

	lr_scaled = linear_model.LinearRegression()
	lr_scaled.fit(train_scaled_data, train_targets)
	lr_pred_scaled = lr_scaled.predict(valid_scaled_data)

	"""kfold = KFold(n_splits = 3, shuffle = True, random_state = seed)
				for model in range (models):
					cvscores = []
					cvscores_scaled = []
			
					loss_train_all = []
					loss_train_scaled_all = []
			
					loss_valid_all = []
					loss_valid_scaled_all = []
			
					network_mse = []
					network_mse_scaled = []
			
					baseline_mse = []
					baseline_mse_scaled = []
			
					batch = np.random.choice(batches)
					alpha = np.random.choice(alphas)
					learningrate = np.random.choice(lrs)
					batches_used.append(batch)
					lr_used.append(learningrate)
					alphas_used.append(alpha)
					
					print("Start Training for Model no: " + str(model) + " with configs: lr = " + str(learningrate) + " , batch size =" + str(batch))
			
					for train, test in kfold.split(data, targets):
			
						# For regularization uncomment the below line
						#alpha = 0
						
						# Network for raw & scaled data
						loss, model_raw, _ = mlp(data[train], targets[train], batch, epochs, 
							learning_rate = learningrate, raw = True, alpha = alpha)
						loss_scaled, model_scaled, _ = mlp(data_scaled[train], targets[train], batch, epochs, 
							learning_rate = learningrate, raw = False, alpha = alpha)
			
						loss_train_all.append(loss.history['loss'])
						#loss_valid_all.append(loss.history['val_loss'])
						loss_train_scaled_all.append(loss_scaled.history['loss'])
						#loss_valid_scaled_all.append(loss_scaled.history['val_loss'])
			
						scores = model_raw.evaluate(data[test], targets[test])
						scores_scaled = model_scaled.evaluate(data_scaled[test], targets[test])
			
						predictions = model_raw.predict(data[test])
						predictions_scaled = model_scaled.predict(data_scaled[test])
			
						network_mse.append(mean_squared_error(targets[test], predictions))
						network_mse_scaled.append(mean_squared_error(targets[test], predictions_scaled))
			
						print("%s: %.2f%%" % (model_raw.metrics_names[0], scores*100))
						print("%s: %.2f%%" % (model_scaled.metrics_names[0], scores_scaled*100))
						
						cvscores.append(scores * 100)
						cvscores_scaled.append(scores_scaled * 100)
						
						# Linear Regression for raw & scaled data
						lr = linear_model.LinearRegression()
						lr.fit(data[train], targets[train])
						lr_pred = lr.predict(data[test])
			
						lr_scaled = linear_model.LinearRegression()
						lr_scaled.fit(data_scaled[train], targets[train])
						lr_pred_scaled = lr_scaled.predict(data_scaled[test])
			
						baseline_mse.append(mean_squared_error(targets[test], lr_pred))
						baseline_mse_scaled.append(mean_squared_error(targets[test], lr_pred_scaled))
			
					print("Training done for model " + str(model))
			
					history_train.append(np.mean(loss_train_all, axis=0))
					history_train_scaled.append(np.mean(loss_train_scaled_all, axis=0))
			
					history_valid.append(np.mean(loss_valid_all, axis=0))
					history_valid_scaled.append(np.mean(loss_valid_scaled_all, axis=0))
			
					cvscores_all.append(np.mean(cvscores))
					cvscores_scaled_all.append(np.mean(cvscores_scaled))
			
					network_mse_all.append(np.mean(network_mse))
					network_mse_scaled_all.append(np.mean(network_mse_scaled))
			
					baseline_mse_all.append(np.mean(baseline_mse))
					baseline_mse_scaled_all.append(np.mean(baseline_mse_scaled))
			
					data_sorted, data_scaled_sorted, targets_sorted = sort(data, data_scaled, targets)
			
					#y_net.append(model_raw.predict(data_sorted))
					#y_net_scaled.append(model_raw.predict(data_scaled_sorted))
			
			
			
				# Using best hyperparameters to evaluate
				best_mse = np.argmin(network_mse_all)
				best_mse_scaled = np.argmin(network_mse_scaled_all)
			
				plot_loss(history_train, history_train_scaled, lr_used, batches_used, best_mse, best_mse_scaled, network_mse_all, network_mse_scaled_all)
			
				train_data, train_scaled_data,train_targets,valid_data,valid_scaled_data,valid_targets = get_train_validation_data(data, data_scaled, targets)
				# Network for raw & scaled data
				loss, model_raw, l_rate = mlp(train_data, train_targets, batches_used[best_mse], epochs, learning_rate = lr_used[best_mse], raw = True)
				loss_scaled, model_scaled, l_rate_scaled = mlp(train_scaled_data, train_targets, batches_used[best_mse_scaled], epochs, learning_rate = lr_used[best_mse_scaled], raw = False)
			
				predictions = model_raw.predict(data)
				predictions_scaled = model_scaled.predict(data_scaled)
			
				plot_network_vs_true(targets, predictions, predictions_scaled, best_mse, best_mse_scaled, lr_used, batches_used)
			
				val_predictions = model_raw.predict(valid_data)
				val_predictions_scaled = model_scaled.predict(valid_scaled_data)
			
				lr = linear_model.LinearRegression()
				lr.fit(train_data, train_targets)
				lr_pred = lr.predict(valid_data)
			
				lr_scaled = linear_model.LinearRegression()
				lr_scaled.fit(train_scaled_data, train_targets)
				lr_pred_scaled = lr_scaled.predict(valid_scaled_data)
			
				baseline = mean_squared_error(valid_targets, lr_pred_scaled)
			"""
	"""# Predict using the network (raw data)
		raw = True
		net = predict(data_sorted, raw)

		
		y_pred.append(pred)
		y_net.append(np.array(net))
		

		# Predict using the network (raw data)
		raw = False
		net = predict(data_scaled_sorted, raw)


		y_pred_scaled.append(pred)
		y_net_scaled.append(np.array(net))"""

	#plot()