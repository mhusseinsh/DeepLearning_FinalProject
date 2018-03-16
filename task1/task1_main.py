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
from mlp import *
from plotting import *


if __name__ == "__main__":


	#decaying_lrs = [[1e-4, 1e-6], [1e-4, 1e-7], [1e-2, 1e-6], [1e-3, 1e-6]]
	decaying_lrs = [1e-3, 1e-4, 1e-5, 1e-3, 1e-6] 
	#alphas = [1e-7, 1e-6, 1e-5, 1e-4]
	alphas = [0]
	max_depth = [2, 4, 8, 16, 32]
	n_estimators = [4, 8, 16, 32]
	min_samples_leaf = [1, 2, 4, 8]
	models = 5
	num_epochs = 1000


	# randomness
	kfold = get_folds()

	if not os.path.exists("./Plots/Train/Task1"):
		os.makedirs("./Plots/Train/Task1")

	# Get the data, prepare it for further use
	data, targets, targets_original = prepare_data()
	
	data_scaled = preprocess_data(data)
	data_sorted, data_scaled_sorted, targets_sorted = sort(data, data_scaled, targets)

	overall_mse = []
	s_overall_mse = []
	b_overall_mse = []
	b_s_overall_mse = []

	lr_used = []
	alpha_used = []
	n_estimator_used = []
	min_leaf_used = []
	depth_used = []

	for model in range(models):
		# randomize hyperparams
		idx = np.random.randint(0, len(decaying_lrs))
		learningrate = decaying_lrs[idx]
		alpha = np.random.choice(alphas)
		lr_used.append(learningrate)
		alpha_used.append(alpha)

		n_estimator = np.random.choice(n_estimators)
		min_leaf = np.random.choice(min_samples_leaf)
		depth = np.random.choice(max_depth)

		n_estimator_used.append(n_estimator)
		min_leaf_used.append(min_leaf)
		depth_used.append(depth)
		# get data


		all_split_score = []
		all_split_mse = []
		all_split_loss=[]


		s_all_split_score = []
		s_all_split_mse = []
		s_all_split_loss=[]

		b_all_split_score = []
		b_all_split_mse = []
		b_all_split_loss=[]


		b_s_all_split_score = []
		b_s_all_split_mse = []
		b_s_all_split_loss=[]


		model = mlp(num_epochs = num_epochs, learning_rate = learningrate, alpha = alpha)
		s_model = mlp(num_epochs = num_epochs, learning_rate = learningrate, alpha = alpha)
		b_model = RandomForestRegressor(max_depth=depth, n_estimators=n_estimator, min_samples_leaf = min_leaf)
		b_s_model = RandomForestRegressor(max_depth=depth, n_estimators=n_estimator, min_samples_leaf = min_leaf)

		for train, valid in kfold.split(data, targets):
			
			split_loss = []
			split_score = []
			split_mse = []

			split_loss = model.fit(data[train], targets[train], epochs = num_epochs, batch_size = 28).history['loss']

			all_split_loss.append(np.mean(split_loss))

			split_score = model.evaluate(data[valid], targets[valid]) * 100

			preds = model.predict(data[valid])
			
			split_mse = mean_squared_error(preds, targets[valid])

			all_split_score.append(np.mean(split_score))
			all_split_mse.append(np.mean(split_mse))

			# scaled data part starts
			
			s_split_loss = []
			s_split_score = []
			s_split_mse = []

			
			s_split_loss = s_model.fit(data_scaled[train], targets[train], epochs = num_epochs, batch_size = 28).history['loss']

			s_all_split_loss.append(np.mean(split_loss))

			s_split_score = s_model.evaluate(data_scaled[valid], targets[valid]) * 100
			s_preds = s_model.predict(data_scaled[valid])
			
			s_split_mse = mean_squared_error(s_preds, targets[valid])

			s_all_split_score.append(np.mean(s_split_score))
			s_all_split_mse.append(np.mean(s_split_mse))

			# raw baseline
			
			b_split_loss = []
			b_split_score = []
			b_split_mse = []

			training_data_rf = []
			training_targets_rf = []
			for i in (train):
				training_data_rf.append(data[i])
				training_targets_rf.append(targets[i])

			training_data_rf = np.asarray(training_data_rf)
			training_targets_rf = np.asarray(training_targets_rf).reshape(len(training_targets_rf),)

			valid_data_rf = []
			for i in (valid):
				valid_data_rf.append(data[i])

			valid_data_rf = np.asarray(valid_data_rf)
			
			b_split_loss = b_model.fit(training_data_rf, training_targets_rf)

			#b_all_split_loss.append(np.mean(b_split_loss))

			#b_split_score = b_model.evaluate(data[valid], targets[valid]) * 100
			b_preds = b_model.predict(valid_data_rf)
			
			b_split_mse = mean_squared_error(b_preds, targets[valid])

			b_all_split_score.append(np.mean(b_split_score))
			b_all_split_mse.append(np.mean(b_split_mse))

			# scaled baseline
			
			b_s_split_loss = []
			b_s_split_score = []
			b_s_split_mse = []

			training_data_rf = []
			training_targets_rf = []
			for i in (train):
				training_data_rf.append(data_scaled[i])
				training_targets_rf.append(targets[i])

			training_data_rf = np.asarray(training_data_rf)
			training_targets_rf = np.asarray(training_targets_rf).reshape(len(training_targets_rf),)

			valid_data_rf = []
			for i in (valid):
				valid_data_rf.append(data_scaled[i])

			valid_data_rf = np.asarray(valid_data_rf)
			b_s_split_loss = b_s_model.fit(training_data_rf, training_targets_rf)

			#b_s_all_split_loss.append(np.mean(b_s_split_loss))

			#b_s_split_score = b_s_model.evaluate(valid_data_rf, targets[valid]) * 100
			b_s_preds = b_s_model.predict(valid_data_rf)
			
			b_s_split_mse = mean_squared_error(b_s_preds, targets[valid])

			b_s_all_split_score.append(np.mean(b_s_split_score))
			b_s_all_split_mse.append(np.mean(b_s_split_mse))
			
		overall_score = np.mean(all_split_score)
		overall_mse.append(np.mean(all_split_mse))
		overall_loss = np.mean(all_split_loss)

		s_overall_score = np.mean(s_all_split_score)
		s_overall_mse.append(np.mean(s_all_split_mse))
		s_overall_loss = np.mean(s_all_split_loss)

		b_overall_score = np.mean(b_all_split_score)
		b_overall_mse.append(np.mean(b_all_split_mse))
		b_overall_loss = np.mean(b_all_split_loss)

		b_s_overall_score = np.mean(b_s_all_split_score)
		b_s_overall_mse.append(np.mean(b_s_all_split_mse))
		b_s_overall_loss = np.mean(b_s_all_split_loss)


	best = np.argmin(overall_mse)
	best_lr = lr_used[best]
	best_alpha = alpha_used[best]

	s_best = np.argmin(s_overall_mse)
	s_best_lr = lr_used[s_best]
	s_best_alpha = alpha_used[s_best]

	b_best = np.argmin(b_overall_mse)
	b_n_estimator = n_estimator_used[b_best]
	b_min_leaf = min_leaf_used[b_best]
	b_depth = depth_used[b_best]

	b_s_best = np.argmin(b_s_overall_mse)
	b_s_n_estimator = n_estimator_used[b_s_best]
	b_s_min_leaf = min_leaf_used[b_s_best]
	b_s_depth = depth_used[b_s_best]

	all_preds = []
	s_all_preds = []
	b_s_all_preds = []
	b_all_preds = []
	targets_last = []

	model = mlp(num_epochs = num_epochs, learning_rate = best_lr, alpha = best_alpha)

	s_model = mlp(num_epochs = num_epochs, learning_rate = s_best_lr, alpha = s_best_alpha)
	b_model = RandomForestRegressor(max_depth=b_depth, n_estimators=b_n_estimator, min_samples_leaf = b_min_leaf)
	b_s_model = RandomForestRegressor(max_depth=b_s_depth, n_estimators=b_s_n_estimator, min_samples_leaf = b_s_min_leaf)

	all_split_score = []
	all_split_mse = []
	all_split_loss=[]

	s_all_split_score = []
	s_all_split_mse = []
	s_all_split_loss=[]

	b_all_split_score = []
	b_all_split_mse = []
	b_all_split_loss=[]

	b_s_all_split_score = []
	b_s_all_split_mse = []
	b_s_all_split_loss=[]

	for train, valid in kfold.split(data, targets):
		
		split_loss = []
		split_score = []
		split_mse = []

		split_loss = model.fit(data[train], targets[train], epochs = num_epochs, batch_size = 28).history['loss']

		all_split_loss.append(np.mean(split_loss))

		split_score = model.evaluate(data[valid], targets[valid]) * 100
		preds = model.predict(data[valid])
		
		all_preds.append(preds)

		split_mse = mean_squared_error(preds, targets[valid])

		all_split_score.append(np.mean(split_score))
		all_split_mse.append(np.mean(split_mse))

		# scaled data part starts
		
		s_split_loss = []
		s_split_score = []
		s_split_mse = []

		s_split_loss = s_model.fit(data_scaled[train], targets[train], epochs = num_epochs, batch_size = 28).history['loss']

		s_all_split_loss.append(np.mean(split_loss))

		s_split_score = s_model.evaluate(data_scaled[valid], targets[valid]) * 100
		s_preds = s_model.predict(data_scaled[valid])
		s_all_preds.append(s_preds)
		s_split_mse = mean_squared_error(s_preds, targets[valid])

		s_all_split_score.append(np.mean(s_split_score))
		s_all_split_mse.append(np.mean(s_split_mse))

		# raw baseline
			
		b_split_loss = []
		b_split_score = []
		b_split_mse = []

		training_data_rf = []
		training_targets_rf = []
		for i in (train):
			training_data_rf.append(data[i])
			training_targets_rf.append(targets[i])

		training_data_rf = np.asarray(training_data_rf)
		training_targets_rf = np.asarray(training_targets_rf)

		valid_data_rf = []

		for i in (valid):
			valid_data_rf.append(data[i])

		valid_data_rf = np.asarray(valid_data_rf)

		b_split_loss = b_model.fit(training_data_rf, training_targets_rf)

		#b_all_split_loss.append(np.mean(b_split_loss))

		#b_split_score = b_model.evaluate(data[valid], targets[valid]) * 100
		b_preds = b_model.predict(valid_data_rf)
		b_all_preds.append(b_preds)
		b_split_mse = mean_squared_error(b_preds, targets[valid])

		b_all_split_score.append(np.mean(b_split_score))
		b_all_split_mse.append(np.mean(b_split_mse))

		# scaled baseline
		
		b_s_split_loss = []
		b_s_split_score = []
		b_s_split_mse = []

		training_data_rf = []
		training_targets_rf = []
		for i in (train):
			training_data_rf.append(data_scaled[i])
			training_targets_rf.append(targets[i])

		training_data_rf = np.asarray(training_data_rf)
		training_targets_rf = np.asarray(training_targets_rf)

		valid_data_rf = []
		for i in (valid):
			valid_data_rf.append(data_scaled[i])

		valid_data_rf = np.asarray(valid_data_rf)

		b_s_split_loss = b_s_model.fit(training_data_rf, training_targets_rf)

		#b_s_all_split_loss.append(np.mean(b_s_split_loss))

		#b_s_split_score = b_s_model.evaluate(data_scaled[valid], targets[valid]) * 100
		b_s_preds = b_s_model.predict(valid_data_rf)
		b_s_all_preds.append(b_s_preds)

		b_s_split_mse = mean_squared_error(b_s_preds, targets[valid])

		b_s_all_split_score.append(np.mean(b_s_split_score))
		b_s_all_split_mse.append(np.mean(b_s_split_mse))

		targets_last.append(targets[valid])

	params = [best_lr,best_alpha]
	s_params = [s_best_lr,s_best_alpha]

	b_params = [b_depth, b_n_estimator, b_min_leaf]
	b_s_params = [b_s_depth, b_s_n_estimator, b_s_min_leaf]

	plot_task1_vs_true(targets_last, all_preds, all_split_mse, b_all_preds, b_all_split_mse, params, b_params)
	plot_task1_vs_true2(targets_last, s_all_preds, s_all_split_mse, b_s_all_preds, b_s_all_split_mse, s_params, b_s_params)

