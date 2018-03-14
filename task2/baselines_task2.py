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
from rnn import rnn_stateful 
from plotting import *


if __name__ == "__main__":

	max_depth = [2, 4, 8, 16, 32]
	n_estimators = [4, 8, 16, 32]
	min_samples_leaf = [1, 2, 4, 8]
	models = 5
	train_time = [5, 10, 20]

	# randomness
	kfold = get_folds()

	# get data
	data, targets, targets_original = prepare_data()
	data_scaled = preprocess_data(data)

	y40 = []
	for y in targets:
		y40.append(y[-1])
	y40_list = [y.tolist() for y in y40]

	if not os.path.exists("./Plots/Train/Baselines2"):
			os.makedirs("./Plots/Train/Baselines2")


	for l in train_time:

		if not os.path.exists("./Plots/Train/Baselines2/Test " + str(l)):
				os.makedirs("./Plots/Train/Baselines2/Test " + str(l))

		targets_selected = y_select(targets_original, l)

		overall_mse = []
		n_estimator_used = []
		min_leaf_used = []
		depth_used = []


		for model in range(models):
			# randomize hyperparams
			n_estimator = np.random.choice(n_estimators)
			min_leaf = np.random.choice(min_samples_leaf)
			depth = np.random.choice(max_depth)

			n_estimator_used.append(n_estimator)
			min_leaf_used.append(min_leaf)
			depth_used.append(depth)

			

			all_split_mse = []
			

			for train, valid in kfold.split(targets_selected, y40_list):
				rf_raw = RandomForestRegressor(max_depth=depth, n_estimators=n_estimator, min_samples_leaf = min_leaf)
				split_input = []
				split_targets = []
				for t_index in train:
					split_input.append(targets_selected[t_index])
					split_targets.append(y40_list[t_index])

				rf_raw.fit(split_input, split_targets)

				split_input = []
				split_targets = []
				for v_index in valid:
					split_input.append(targets_selected[v_index])
					split_targets.append(y40_list[v_index])

							
				raw_predictions = rf_raw.predict(split_input)

				all_split_mse.append(np.mean(mean_squared_error(split_targets, raw_predictions)))

			overall_mse.append(np.mean(all_split_mse))


		best = np.argmin(overall_mse)
		n_estimator = n_estimator_used[best]
		min_leaf = min_leaf_used[best]
		depth = depth_used[best]

		
		raw_predictions = []

		for train, valid in kfold.split(targets_selected, y40_list):
			rf_raw = RandomForestRegressor(max_depth=depth, n_estimators=n_estimator, min_samples_leaf = min_leaf)
			split_input = []
			split_targets = []
			for t_index in train:
				split_input.append(targets_selected[t_index])
				split_targets.append(y40_list[t_index])

			rf_raw.fit(split_input, split_targets)

			split_input = []
			split_targets = []
			for v_index in valid:
				split_input.append(targets_selected[v_index])
				split_targets.append(y40_list[v_index])

						
			raw_predictions.append(rf_raw.predict(split_input))

		preds = [item for sublist in raw_predictions for item in sublist]

		plot_baseline_vs_true(y40_list, preds, overall_mse[best], l)

	if not os.path.exists("./Plots/Train/Baselines2/Last"):
		os.makedirs("./Plots/Train/Baselines2/Last")

	inputs_last_baseline, targets_last_baseline = prepare_last_baseline(targets_original)

	overall_mse = []
	n_estimator_used = []
	min_leaf_used = []
	depth_used = []

	for model in range(models):
		# randomize hyperparams
		n_estimator = np.random.choice(n_estimators)
		min_leaf = np.random.choice(min_samples_leaf)
		depth = np.random.choice(max_depth)

		n_estimator_used.append(n_estimator)
		min_leaf_used.append(min_leaf)
		depth_used.append(depth)

		all_split_mse = []

		for train, valid in kfold.split(inputs_last_baseline, targets_last_baseline):
			split_mse = []
			rf_last = RandomForestRegressor(max_depth=depth, n_estimators=n_estimator, min_samples_leaf = min_leaf)
			
			training_data_rf = []
			training_targets_rf = []
			for i in (train):
				training_data_rf.append(inputs_last_baseline[i])
				training_targets_rf.append(targets_last_baseline[i])

			training_data_rf = np.asarray(training_data_rf)
			training_targets_rf = np.asarray(training_targets_rf)

			valid_data_rf = []
			for i in (valid):
				valid_data_rf.append(inputs_last_baseline[i])

			valid_data_rf = np.asarray(valid_data_rf)

			rf_last.fit(training_data_rf, training_targets_rf)

			predictions = rf_last.predict(valid_data_rf)

			split_mse.append(mean_squared_error(targets_last_baseline[valid], predictions))

		overall_mse.append(np.mean(split_mse))

	best = np.argmin(overall_mse)
	n_estimator = n_estimator_used[best]
	min_leaf = min_leaf_used[best]
	depth = depth_used[best]

	
	predictions_last = []
	targets_last = []
	split_mse = []

	for train, valid in kfold.split(inputs_last_baseline, targets_last_baseline):
		rf_last = RandomForestRegressor(max_depth=depth, n_estimators=n_estimator, min_samples_leaf = min_leaf)
		training_data_rf = []
		training_targets_rf = []
		for i in (train):
			training_data_rf.append(inputs_last_baseline[i])
			training_targets_rf.append(targets_last_baseline[i])

		training_data_rf = np.asarray(training_data_rf)
		training_targets_rf = np.asarray(training_targets_rf)

		valid_data_rf = []
		for i in (valid):
			valid_data_rf.append(inputs_last_baseline[i])

		valid_data_rf = np.asarray(valid_data_rf)

		rf_last.fit(training_data_rf, training_targets_rf)

		predictions = rf_last.predict(valid_data_rf)

		split_mse.append(mean_squared_error(targets_last_baseline[valid], predictions))
		targets_last.append(targets_last_baseline[valid])
		predictions_last.append(predictions)

	plot_baseline_vs_true2(targets_last, predictions_last, split_mse)

